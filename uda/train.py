from __future__ import division

import argparse
import os
import os.path as osp
import warnings
import wandb
import torch
import pickle
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from mmcv import Config
from mmcv.runner.dist_utils import master_only

from mmdet import __version__
from mmdet.apis import (get_root_logger, init_dist, set_random_seed,
                        train_detector)
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from imslp import ImslpDataset
from transforms_imslp import *
from core.utils.metric_logger import MetricLogger
from core.models import build_model, build_adversarial_discriminator, build_feature_extractor, build_classifier
from core.solver import adjust_learning_rate
from core.utils.loss_ops import reduce_loss_dict, summarise_loss, soft_label_cross_entropy
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', help='test config file path', default = './configs/deepscoresv2/s2anet_r50_fpn_1x_deepscoresv2_ghos_lowrez.py')
    parser.add_argument('--work_dir', help='the dir to save logs and models', default='./models/uda/')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from', default = './models/epoch_500.pth')
        # '--resume_from', help='the checkpoint file to resume from', default = None)
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=2,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def collate_fn_Custom(batch):
    img_meta = list()
    img = list()
    gt_bboxes = list()
    gt_labels = list()

    for b in batch:

        img_meta.append(b['img_meta'])
        img.append(b['img'])
        gt_bboxes.append(b['gt_bboxes'])

        gt_labels.append(b['gt_labels'])

    img = torch.stack(img, dim=0)

    return [img_meta, img, gt_bboxes, gt_labels]


def main():
    wandb.init(project="uda", entity="adhirajghosh")
    args = parse_args()

    cfg = Config.fromfile(args.config)
    num_epochs = cfg.total_epochs
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
        # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.gpus / 8

    # init distributed env first, since logger depends on the dist info.
    # if args.launcher == 'none':
    #     distributed = False
    # else:
    #     distributed = True
    #     init_dist(args.launcher, **cfg.dist_params)

    num_gpus = 2
    distributed = False
    args.local_rank = 1

    if distributed:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '8001'
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://", rank=1, world_size = 4
        )

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    #DATASETS
    #Takes more time at the start. Doesn't affect training epochs. Probably makes it faster
    #TODO: Make it a part of the dataset call.

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        datasets.append(build_dataset(cfg.data.val))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.text,
            CLASSES=datasets[0].CLASSES)

    trainset = []
    for _, j in enumerate(datasets):

        for i in range(len(j)):
            sample = {}
            sample['img_meta'] = j[i]['img_meta'].data
            sample['img'] = j[i]['img'].data
            sample['gt_bboxes'] = j[i]['gt_bboxes'].data
            sample['gt_labels'] = j[i]['gt_labels'].data
            trainset.append(sample)


    trainloader_src = DataLoader(trainset, batch_size=1, shuffle=True, collate_fn=collate_fn_Custom,
                                 num_workers=4)

    data_transform = transforms.Compose([ToTensor()])

    dataset = ImslpDataset(split_file='./data/imslp_dataset/train_test_split/train_list.txt',
                           root_dir='./data//imslp_dataset/images/', transform=data_transform)

    trainloader_tgt = torch.utils.data.DataLoader(dataset,
                                                  batch_size=1, shuffle=True,
                                                  num_workers=4)

    #MODELS
    device1 = torch.device('cuda:0')
    device2 = torch.device('cuda:1')

    s2anet_det = build_detector(cfg.model, train_cfg=cfg.train_cfg)
    s2anet_det.to(device1)
    # add an attribute for visualization convenience
    s2anet_det.CLASSES = datasets[0].CLASSES

    cfg2 = Config.fromfile('./uda/configs/r50_adv_ghos.yaml')

    feature_extractor = build_feature_extractor(cfg2)
    feature_extractor.to(device1)

    model_D = build_adversarial_discriminator(cfg2)
    model_D.to(device2)

    #TODO: PLACEHOLDER. REPLACE WITH S2ANET PREDICTION TENSOR
    classifier = build_classifier(cfg2)
    classifier.to(device2)

    if distributed:
        feature_extractor = torch.nn.parallel.DistributedDataParallel(
            feature_extractor, device_ids=[0,1])
        classifier = torch.nn.parallel.DistributedDataParallel(
            classifier, device_ids=[0, 1])
        model_D = torch.nn.parallel.DistributedDataParallel(
            model_D, device_ids=[0,1])
        s2anet_det = torch.nn.parallel.DistributedDataParallel(
            s2anet_det, device_ids=[0, 1])
        torch.autograd.set_detect_anomaly(True)
        torch.distributed.barrier()
        # pg1 = torch.distributed.new_group(range(torch.distributed.get_world_size()))

        # feature_extractor = torch.nn.parallel.DistributedDataParallel(
        #         feature_extractor, device_ids=[0,1], output_device=0,
        #     find_unused_parameters=True, #process_group=pg1
        # )

        # pg2 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        # classifier = torch.nn.parallel.DistributedDataParallel(
        #     classifier, device_ids=[0,1], output_device=0,
        #     find_unused_parameters=True, #process_group=pg2
        # )
        # pg3 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        # model_D = torch.nn.parallel.DistributedDataParallel(
        #     model_D, device_ids=[0,1], output_device=0,
        #     find_unused_parameters=True, #process_group=pg3
        # )
        #
        # # pg4 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        # model_D = torch.nn.parallel.DistributedDataParallel(
        #     model_D, device_ids=[0, 1], output_device=0,
        #     find_unused_parameters=True, #process_group=pg4
        # )


    #OPTIMIZERS
    optimizer_fea = torch.optim.SGD(feature_extractor.parameters(), lr=cfg2.SOLVER.BASE_LR, momentum=cfg2.SOLVER.MOMENTUM,
                                    weight_decay=cfg2.SOLVER.WEIGHT_DECAY)
    optimizer_fea.zero_grad()

    optimizer_s2a = torch.optim.SGD(s2anet_det.parameters(), lr=cfg2.SOLVER.BASE_LR * 10, momentum=cfg2.SOLVER.MOMENTUM,
                                    weight_decay=cfg2.SOLVER.WEIGHT_DECAY)
    optimizer_s2a.zero_grad()

    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=cfg2.SOLVER.BASE_LR_D, betas=(0.9, 0.99))
    optimizer_D.zero_grad()

    logger.info(feature_extractor)
    logger.info(model_D)
    logger.info(classifier)

    feature_extractor.train()
    model_D.train()
    classifier.train()
    s2anet_det.train()

    max_iters = cfg2.SOLVER.MAX_ITER
    meters = MetricLogger(delimiter="  ")
    for epoch in range(num_epochs):
        print("For epoch ", epoch)
        for i_batch, ((src), (tgt))  in enumerate(zip(trainloader_src, trainloader_tgt)):

            current_lr = adjust_learning_rate(cfg2.SOLVER.LR_METHOD, cfg2.SOLVER.BASE_LR, epoch, max_iters,
                                              power=cfg2.SOLVER.LR_POWER)
            current_lr_D = adjust_learning_rate(cfg2.SOLVER.LR_METHOD, cfg2.SOLVER.BASE_LR_D, epoch, max_iters,
                                                power=cfg2.SOLVER.LR_POWER)
            for index in range(len(optimizer_fea.param_groups)):
                optimizer_fea.param_groups[index]['lr'] = current_lr
            for index in range(len(optimizer_s2a.param_groups)):
                optimizer_s2a.param_groups[index]['lr'] = current_lr * 10
            for index in range(len(optimizer_D.param_groups)):
                optimizer_D.param_groups[index]['lr'] = current_lr_D

            optimizer_fea.zero_grad()
            optimizer_s2a.zero_grad()
            optimizer_D.zero_grad()

            # Index and meaning for source
            # 0 is img_meta
            # 1 is img
            # 2 is gt_bboxes
            # 3 is gt_labels
            img_meta = src[0]
            # src_input = src[1].cuda(non_blocking=True)
            src_input = src[1].to(device = device1)
            bbox = src[2]
            label = src[3]
            boxes = [b.to(device = device1) for b in bbox]
            # boxes = [b.cuda(non_blocking=True) for b in bbox]
            labels = [l.to(device = device1) for l in label]
            # labels = [l.cuda(non_blocking=True) for l in label]

            #For Target
            tgt_input = tgt['image'].to(device = device1, dtype = torch.float32)
            # tgt_input = tgt['image'].cuda(non_blocking=True).float()

            src_size = src_input.shape[-2:]
            tgt_size = tgt_input.shape[-2:]

            src_fea = feature_extractor(src_input)
            src_fea2 = src_fea.to(device2)
            src_pred = classifier(src_fea2, src_size)
            temperature = 1.8
            src_pred = src_pred.div(temperature)
            src_soft_label = F.softmax(src_pred, dim=1).detach()
            src_soft_label[src_soft_label > 0.9] = 0.9

            loss_dict = s2anet_det(src_input, img_meta, boxes, labels)
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            loss_obj, loss_reduced = summarise_loss(loss_dict_reduced)
            loss_obj.backward()
            #print(loss_obj, "\t", loss_reduced)

            #print(tgt_input)
            tgt_fea = feature_extractor(tgt_input)
            tgt_fea2 = tgt_fea.to(device2)
            tgt_pred = classifier(tgt_fea2, tgt_size)
            tgt_pred = tgt_pred.div(temperature)
            tgt_soft_label = F.softmax(tgt_pred, dim=1)

            tgt_soft_label = tgt_soft_label.detach()
            tgt_soft_label[tgt_soft_label > 0.9] = 0.9


            tgt_D_pred = model_D(tgt_fea2, tgt_size)
            loss_adv_tgt = 0.001 * soft_label_cross_entropy(tgt_D_pred, torch.cat(
                (tgt_soft_label, torch.zeros_like(tgt_soft_label)), dim=1))
            loss_adv_tgt.backward()

            optimizer_fea.step()
            #optimizer_s2a.step()

            optimizer_D.zero_grad()
            # torch.distributed.barrier()

            src_D_pred = model_D(src_fea2.detach(), src_size)
            loss_D_src = 0.5 * soft_label_cross_entropy(src_D_pred,
                                                        torch.cat((src_soft_label, torch.zeros_like(src_soft_label)),
                                                                  dim=1))
            loss_D_src.backward()

            tgt_D_pred = model_D(tgt_fea2.detach(), tgt_size)
            loss_D_tgt = 0.5 * soft_label_cross_entropy(tgt_D_pred,
                                                        torch.cat((torch.zeros_like(tgt_soft_label), tgt_soft_label),
                                                                  dim=1))
            loss_D_tgt.backward()
            optimizer_D.step()

            meters.update(loss_obj=loss_obj.item())
            meters.update(loss_adv_tgt=loss_adv_tgt.item())
            meters.update(loss_D=(loss_D_src.item() + loss_D_tgt.item()))
            meters.update(loss_D_src=loss_D_src.item())
            meters.update(loss_D_tgt=loss_D_tgt.item())


            if i_batch % 100 ==0:
                logger.info(
                    meters.delimiter.join(
                        [
                            "epoch: {epoch}",
                            "{meters}",
                            "lr: {lr:.6f}",
                        ]
                    ).format(
                        epoch=epoch,
                        meters=str(meters),
                        lr=optimizer_fea.param_groups[0]["lr"],
                    )
                )

            if epoch % 10 ==0 or epoch == num_epochs:
                if i_batch % 100 ==0:
                    if not os.path.isdir(args.work_dir):
                        os.makedirs(args.work_dir)
                    logger.info("Saving model")
                    filename = os.path.join(args.work_dir, "model_{:03d}.pth".format(epoch))
                    torch.save({'epoch': epoch, 'detector': s2anet_det.state_dict(),
                                'feature_extractor': feature_extractor.state_dict(),
                                'classifier': classifier.state_dict(), 'model_D': model_D.state_dict(),
                                'optimizer_fea': optimizer_fea.state_dict(), 'optimizer_det': optimizer_s2a.state_dict(),
                                'optimizer_D': optimizer_D.state_dict()}, filename)


if __name__ == '__main__':
    main()
