import wandb
import argparse
import os
import mmcv
import torch

from mmcv.runner import get_dist_info, load_checkpoint

from mmdet.core import coco_eval, results2json, wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector

from results import strip_prefix_if_present

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('--config', help='test config file path', default = 'configs/deepscoresv2/s2anet_r50_fpn_1x_deepscoresv2_ghos_halfrez_crop.py')
    parser.add_argument('--checkpoint2', help='checkpoint file', default = 'models/uda/model_020.pth')
    parser.add_argument('--checkpoint1', help='checkpoint file', default = 'models/deepscoresV2_tugg_halfrez_crop_epoch250.pth')
    args = parser.parse_args()
    return args

def main():
    wandb.init(project="uda", entity="adhirajghosh")
    args = parse_args()
    args.show = False

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None


    cfg.data.test.test_mode = True


    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)
    # build the model and load checkpoint
    model1 = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    model2 = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model1)
        wrap_fp16_model(model2)

    checkpoint1 = load_checkpoint(model1, args.checkpoint1, map_location='cpu')

    checkpoint_tmp = torch.load(args.checkpoint2, map_location=torch.device('cpu'))
    checkpoint2 = {}
    checkpoint2['meta'] = checkpoint1['meta']
    checkpoint2['state_dict'] = checkpoint_tmp['detector']
    checkpoint2['optimizer'] = checkpoint_tmp['optimizer_det']
    model_weights = strip_prefix_if_present(checkpoint2['state_dict'], 'module.')
    model2.load_state_dict(model_weights)


    print("For model1")
    print(checkpoint1.keys())

    print("For model2")
    print(checkpoint2.keys())

    torch.save(checkpoint2, 'models/uda/epoch_20.pth' )



if __name__ == '__main__':
    main()
