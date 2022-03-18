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
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet import __version__
from mmdet.apis import (get_root_logger, init_dist, set_random_seed,
                        train_detector)
from mmdet.apis.train import build_optimizer
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models import build_detector
from imslp import ImslpDataset
from transforms_imslp import *
from core.utils.metric_logger import MetricLogger
from core.utils.logger import setup_logger
from core.models import build_model, build_adversarial_discriminator, build_feature_extractor, build_classifier
from core.solver import adjust_learning_rate
from core.utils.loss_ops import reduce_loss_dict, summarise_loss, soft_label_cross_entropy
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', help='test config file path', default = 'configs/deepscoresv2/s2anet_r50_fpn_1x_deepscoresv2_tugg_halfrez_crop.py')
    parser.add_argument('--work_dir', help='the dir to save logs and models', default='models/uda/')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from', default = 'models/deepscoresV2_tugg_halfrez_crop_epoch250.pth')
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

def main():
    wandb.init(project="uda", entity="adhirajghosh")
    args = parse_args()

    cfg = Config.fromfile(args.config)
    num_epochs = cfg.total_epochs
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus

    if args.autoscale_lr:
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.gpus / 8

    num_gpus = 2
    distributed = True
    args.local_rank = 1

    if distributed:
        torch.distributed.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)

    logger = setup_logger('UDA', args.work_dir, args.local_rank)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)
    logger.info('Distributed training: {}'.format(distributed))

    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        datasets.append(build_dataset(cfg.data.val))
    datasets = datasets if isinstance(datasets, (list, tuple)) else [datasets]
    dataloader_src = [
        build_dataloader(
            ds, cfg.data.imgs_per_gpu, cfg.data.workers_per_gpu, dist=True)
        for ds in datasets
    ]

    data_transform = transforms.Compose([ToTensor()])
    imslp_dataset = ImslpDataset(split_file='./data/imslp_dataset/train_test_split/train_list.txt',
                                 root_dir='./data//imslp_dataset/images/', transform=data_transform)
    dataloader_tgt = DataLoader(imslp_dataset,batch_size=2, shuffle=True,num_workers=2)

    # put model on gpus
    s2anet = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    s2anet = MMDistributedDataParallel(s2anet.cuda())
    s2anet.CLASSES = datasets[0].CLASSES


    cfg2 = Config.fromfile('./uda/configs/r50_adv_ghos.yaml')
    feature_extractor = build_feature_extractor(cfg2)
    feature_extractor.cuda()

    model_D = build_adversarial_discriminator(cfg2)
    model_D.cuda()

    classifier = build_classifier(cfg2)
    classifier.cuda()

    if distributed:
        pg1 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        feature_extractor = torch.nn.parallel.DistributedDataParallel(
                feature_extractor, device_ids=[0,1], output_device=0,
            find_unused_parameters=False, process_group=pg1
        )

        pg2 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        classifier = torch.nn.parallel.DistributedDataParallel(
            classifier, device_ids=[0,1], output_device=0,
            find_unused_parameters=False, process_group=pg2
        )
        pg3 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        model_D = torch.nn.parallel.DistributedDataParallel(
            model_D, device_ids=[0,1], output_device=0,
            find_unused_parameters=False, process_group=pg3
        )
        torch.autograd.set_detect_anomaly(True)
        torch.distributed.barrier()

    optimizer_s2anet = build_optimizer(s2anet, cfg.optimizer)
    optimizer_s2anet.zero_grad()

    optimizer_fea = torch.optim.SGD(feature_extractor.parameters(), lr=cfg2.SOLVER.BASE_LR,
                                    momentum=cfg2.SOLVER.MOMENTUM,
                                    weight_decay=cfg2.SOLVER.WEIGHT_DECAY)
    optimizer_fea.zero_grad()

    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=cfg2.SOLVER.BASE_LR_D, betas=(0.9, 0.99))
    optimizer_D.zero_grad()

    optimizer_cls = torch.optim.SGD(classifier.parameters(), lr=cfg2.SOLVER.BASE_LR*10, momentum=cfg2.SOLVER.MOMENTUM, weight_decay=cfg2.SOLVER.WEIGHT_DECAY)
    optimizer_cls.zero_grad()

    logger.info(feature_extractor)
    logger.info(model_D)
    logger.info(classifier)

    s2anet.train()
    feature_extractor.train()
    model_D.train()
    classifier.train()

    max_iters = cfg2.SOLVER.MAX_ITER
    meters = MetricLogger(delimiter="  ")
    logger.info("Start training")
    for epoch in range(num_epochs):
        print("For epoch ", epoch+1)
        for i_batch, ((src), (tgt)) in enumerate(zip(dataloader_src[0], dataloader_tgt)):
            src_img = src['img']
            src_meta = src['img_meta']
            src_gtbboxes = src['gt_bboxes']
            src_gtlabels = src['gt_labels']

            tgt_img = tgt['image'].cuda(non_blocking=True).float()

            current_lr = adjust_learning_rate(cfg2.SOLVER.LR_METHOD, cfg2.SOLVER.BASE_LR, epoch, max_iters,
                                                      power=cfg2.SOLVER.LR_POWER)
            current_lr_D = adjust_learning_rate(cfg2.SOLVER.LR_METHOD, cfg2.SOLVER.BASE_LR_D, epoch, max_iters,
                                                power=cfg2.SOLVER.LR_POWER)
            for index in range(len(optimizer_fea.param_groups)):
                optimizer_fea.param_groups[index]['lr'] = current_lr
            for index in range(len(optimizer_s2anet.param_groups)):
                optimizer_s2anet.param_groups[index]['lr'] = current_lr
            for index in range(len(optimizer_cls.param_groups)):
                optimizer_cls.param_groups[index]['lr'] = current_lr * 10
            for index in range(len(optimizer_D.param_groups)):
                optimizer_D.param_groups[index]['lr'] = current_lr_D

            optimizer_fea.zero_grad()
            optimizer_cls.zero_grad()
            optimizer_D.zero_grad()

            src_size = src_img.data[0].shape[-2:]
            tgt_size = tgt_img.data.shape[-2:]

            outputs = s2anet.train_step(src_img, src_meta, src_gtbboxes, src_gtlabels, optimizer_s2anet)
            # outputs = s2anet(src_img, src_meta, src_gtbboxes, src_gtlabels)
            loss_dict_reduced = reduce_loss_dict(outputs)
            loss_obj, loss_reduced = summarise_loss(loss_dict_reduced)
            # loss_obj.backward() #happens in train_step()
            # print(outputs)

            src_fea = feature_extractor(src_img.data[0])
            # src_fea2 = src_fea.cuda()
            src_pred = classifier(src_fea, src_size)
            temperature = 1.8
            src_pred = src_pred.div(temperature)
            src_soft_label = F.softmax(src_pred, dim=1).detach()
            src_soft_label[src_soft_label > 0.9] = 0.9

            tgt_fea = feature_extractor(tgt_img)
            # tgt_fea2 = tgt_fea.cuda()
            tgt_pred = classifier(tgt_fea, tgt_size)
            tgt_pred = tgt_pred.div(temperature)
            tgt_soft_label = F.softmax(tgt_pred, dim=1)
            tgt_soft_label = tgt_soft_label.detach()
            tgt_soft_label[tgt_soft_label > 0.9] = 0.9

            tgt_D_pred = model_D(tgt_fea, tgt_size)
            loss_adv_tgt = 0.001 * soft_label_cross_entropy(tgt_D_pred, torch.cat(
                (tgt_soft_label, torch.zeros_like(tgt_soft_label)), dim=1))
            loss_adv_tgt.backward()

            optimizer_fea.step()
            optimizer_s2anet.step()
            optimizer_cls.step()

            optimizer_D.zero_grad()
            # torch.distributed.barrier()

            src_D_pred = model_D(src_fea.detach(), src_size)
            loss_D_src = 0.5 * soft_label_cross_entropy(src_D_pred,torch.cat((src_soft_label, torch.zeros_like(src_soft_label)),dim=1))
            loss_D_src.backward()

            tgt_D_pred = model_D(tgt_fea.detach(), tgt_size)
            loss_D_tgt = 0.5 * soft_label_cross_entropy(tgt_D_pred,torch.cat((torch.zeros_like(tgt_soft_label), tgt_soft_label),dim=1))
            loss_D_tgt.backward()
            optimizer_D.step()

            meters.update(loss_obj=loss_obj.item())
            meters.update(loss_adv_tgt=loss_adv_tgt.item())
            meters.update(loss_D=(loss_D_src.item() + loss_D_tgt.item()))
            meters.update(loss_D_src=loss_D_src.item())
            meters.update(loss_D_tgt=loss_D_tgt.item())

            if i_batch % 20 == 0:
                logger.info('Epoch: [{0}][{1}/{2}]\t'
                      '{meters}\t'
                      'lr: {lr:.6f}\t'.format(
                    epoch + 1, i_batch + 1, len(dataloader_src[0]),
                    meters=str(meters),
                    lr=optimizer_fea.param_groups[0]["lr"],
                ))
                wandb.log({
                    "Detection Loss ": loss_obj.item(),
                    "Adversarial Loss (Target)": loss_adv_tgt.item(),
                    "Discriminator Loss (Source)": loss_D_src.item(),
                    "Discriminator Loss (Target)": loss_D_tgt.item()
                })

        if epoch % 10 == 0 or epoch == num_epochs:
            logger.info("Saving model")
            filename = os.path.join(args.work_dir, "model_{:03d}.pth".format(epoch))
            torch.save({'epoch': epoch, 'detector': s2anet.state_dict(),
                        'feature_extractor': feature_extractor.state_dict(),
                        'classifier': classifier.state_dict(), 'model_D': model_D.state_dict(),
                        'optimizer_fea': optimizer_fea.state_dict(), 'optimizer_det': optimizer_s2anet.state_dict(),
                        'optimizer_D': optimizer_D.state_dict()}, filename)


if __name__ == '__main__':
    main()
