#HERE WE TAKE FEATURES EXTRACTOR FROM S2ANET DIRECTLY

from __future__ import division

import argparse
import os
import os.path as osp
import warnings
import wandb
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms, utils
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel

from mmdet.apis import set_random_seed
from mmdet.apis.train import build_optimizer
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models import build_detector
from imslp import ImslpDataset
from transforms_imslp import *
from core.utils.metric_logger import MetricLogger
from core.utils.logger import setup_logger
from core.utils.gl import WarmStartGradientLayer
from core.models import build_binary_discriminator
from core.solver import DomainAdversarialLoss
from core.utils.loss_ops import reduce_loss_dict, summarise_loss
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    # parser.add_argument('--config', help='test config file path', default = 'configs/deepscoresv2/s2anet_r50_fpn_1x_deepscoresv2_sage_halfrez_crop.py')
    parser.add_argument('--config', help='test config file path', default = 'configs/deepscoresv2/ghos_uda.py')
    parser.add_argument('--work_dir', help='the dir to save logs and models', default='models/uda_test_model_june/pretrained')
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
    # wandb.init(project="omr-uda", entity="adhirajghosh")
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
    if not cfg.resume:
        args.resume_from = None
        args.work_dir = 'models/uda_test_model_june/no_pretrained'
    cfg.gpus = args.gpus

    if args.autoscale_lr:
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.gpus / 8

    num_gpus = cfg.num_gpu
    distributed = cfg.distributed
    args.local_rank = 1

    if distributed:
        torch.distributed.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)

    logger = setup_logger('UDA', args.work_dir, args.local_rank)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info("Pretrained: {}".format(cfg.resume))
    logger.info(args)
    logger.info('Distributed training: {}'.format(distributed))

    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    #S2ANet and IMSLP Dataset Loading Happens Here
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        datasets.append(build_dataset(cfg.data.val))
    datasets = datasets if isinstance(datasets, (list, tuple)) else [datasets]
    dataloader_src = [
        build_dataloader(
            ds, cfg.data.imgs_per_gpu, cfg.data.workers_per_gpu, num_gpus = num_gpus, dist=True)
        for ds in datasets
    ]

    data_transform = transforms.Compose([ToTensor()])
    imslp_dataset = ImslpDataset(split_file
                                 ='data/imslp_dataset/train_test_split/train_list_no_landscape.txt',
                                 root_dir='data/imslp_dataset/images/', transform=data_transform)
    dataloader_tgt = DataLoader(imslp_dataset,batch_size=4, shuffle=True,num_workers=2)

    # Model Loading Happens Here
    s2anet = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    s2anet.CLASSES = datasets[0].CLASSES

    s2anet_feat = 128
    s2anet_hidden = 256
    model_D = build_binary_discriminator(num_gpus, s2anet_feat, s2anet_hidden)
    model_D.cuda()

    if distributed:
        s2anet = MMDistributedDataParallel(s2anet.cuda())

        # pg = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        # model_D = torch.nn.parallel.DistributedDataParallel(
        #     model_D, device_ids=[0,1], output_device=1,
        #     find_unused_parameters=False, process_group=pg
        # )
        torch.autograd.set_detect_anomaly(True)
        torch.distributed.barrier()

    #Optimizers are Set Here
    optimizer_s2anet = build_optimizer(s2anet, cfg.optimizer)
    optimizer_s2anet.zero_grad()

    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=cfg.optimizer_d['lr'], betas=(0.9, 0.99))
    optimizer_D.zero_grad()

    #Schedulers are Set Here
    lr_scheduler_s2anet = LambdaLR(optimizer_s2anet, lambda x: cfg.optimizer['lr'] * (1. + cfg.optimizer_d['gamma'] * float(x)) ** (-cfg.optimizer['weight_decay']))
    lr_scheduler_d = LambdaLR(optimizer_D, lambda x: cfg.optimizer_d['lr'] * (1. + cfg.optimizer_d['gamma'] * float(x)) ** (-cfg.optimizer_d['weight_decay']))

    # define loss function
    domain_adv = DomainAdversarialLoss().cuda()
    gl = WarmStartGradientLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)

    # Load from checkpoint
    if cfg.resume:
        checkpoint = torch.load(args.resume_from)
        s2anet.load_state_dict(checkpoint['state_dict'])
        optimizer_s2anet.load_state_dict(checkpoint['optimizer'])

    logger.info(model_D)

    meters = MetricLogger(delimiter="  ")
    logger.info("Start training")
    for epoch in range(num_epochs):
        print("For epoch ", epoch+1)
        print("lr S2ANet:", lr_scheduler_s2anet.get_lr())
        print("lr discriminator:", lr_scheduler_d.get_lr())
        for i_batch, ((src), (tgt)) in enumerate(zip(dataloader_src[0], dataloader_tgt)):
            src_img = src['img']
            src_meta = src['img_meta']
            src_gtbboxes = src['gt_bboxes']
            src_gtlabels = src['gt_labels']

            tgt_img = tgt['image'].cuda(non_blocking=True).float()

            # Step 1: Train S2ANet, freeze the discriminator
            s2anet.train()
            model_D.eval()
            for param in s2anet.parameters():
                param.requires_grad = True

            for param in model_D.parameters():
                param.requires_grad = False
            src = True
            losses, src_fea = s2anet.train_step(src_img, src_meta, src_gtbboxes, src_gtlabels, src)
            loss_obj, loss_reduced = summarise_loss(losses)

            src = False
            tgt_fea = s2anet.train_step(tgt_img, src_meta, src_gtbboxes, src_gtlabels, src)

            optimizer_s2anet.step()
            # print(src_fea[0].shape)
            # print(tgt_fea[0].shape)

            # adversarial training to fool the discriminator
            src_fea = torch.stack(list(src_fea[0]), dim=0)
            tgt_fea = torch.stack(list(tgt_fea[0]), dim=0)

            src_fea_tensor = torch.tensor(src_fea, requires_grad=True)
            tgt_fea_tensor = torch.tensor(tgt_fea, requires_grad=True)

            feat_concat = torch.cat((src_fea_tensor, tgt_fea_tensor), dim=0).cuda()

            d = model_D(gl(feat_concat))
            d_s, d_t = d.chunk(2, dim=0)

            loss_transfer = 0.5 * (domain_adv(d_s, 'target') + domain_adv(d_t, 'source'))
            optimizer_s2anet.zero_grad()

            (loss_transfer+cfg.trade_off).backward()
            optimizer_s2anet.step()
            lr_scheduler_s2anet.step()

            # Step 2: Train the discriminator
            s2anet.eval()
            model_D.train()
            for param in s2anet.parameters():
                param.requires_grad = False

            for param in model_D.parameters():
                param.requires_grad = True

            d = model_D(feat_concat.detach())
            d_s, d_t = d.chunk(2, dim=0)
            loss_discriminator = 0.5 * (domain_adv(d_s, 'source') + domain_adv(d_t, 'target'))

            optimizer_D.zero_grad()
            loss_discriminator.backward()
            optimizer_D.step()
            lr_scheduler_d.step()


            meters.update(loss_obj=loss_obj.item())
            meters.update(loss_transfer=loss_transfer.item())
            meters.update(loss_discriminator=loss_discriminator.item())


            if i_batch % 20 == 0:
                logger.info('Epoch: [{0}][{1}/{2}]\t'
                      '{meters}\t'
                      'lr: {lr:.6f}\t'.format(
                    epoch + 1, i_batch + 1, len(dataloader_src[0]),
                    meters=str(meters),
                    lr=optimizer_s2anet.param_groups[0]["lr"],
                ))

                wandb.log({
                    "Detection Loss ": loss_obj.item(),
                    "Domain Transfer Loss": loss_transfer.item(),
                    "Discriminator Loss": loss_discriminator.item(),
                })

        meta = {}
        meta['epoch'] = epoch
        meta['CLASSES'] = datasets[0].CLASSES
        if epoch % 10 == 0 or epoch == num_epochs:
            logger.info("Saving model")
            filename = os.path.join(args.work_dir, "model_{:03d}.pth".format(epoch))

            torch.save({'meta': meta, 'state_dict': s2anet.state_dict(), 'optimizer': optimizer_s2anet.state_dict()}, filename)


if __name__ == '__main__':
    main()
