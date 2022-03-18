from __future__ import division

import argparse
import os
import warnings
import pickle
from mmcv import Config
from mmdet.datasets import build_dataset
import resource

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', help='test config file path', default = './configs/deepscoresv2/s2anet_r50_fpn_1x_deepscoresv2_ghos_lowrez.py')
    parser.add_argument('--work_dir', help='the dir to save logs and models', default='./models/uda/')
    parser.add_argument(
        # '--resume_from', help='the checkpoint file to resume from', default = './models/epoch_500.pth')
        '--resume_from', help='the checkpoint file to resume from', default = None)
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
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.gpus = args.gpus

    #DATASETS
    datasets = [build_dataset(cfg.data.train)]
    j = datasets[0]
    size = len(j)
    print(size)

    trainset = []

    for i in range(size):
        print(i)
        sample = {}
        sample['img_meta'] = j[i]['img_meta'].data
        print(sample['img_meta'])
        sample['img'] = j[i]['img'].data.tolist()
        sample['gt_bboxes'] = j[i]['gt_bboxes'].data
        sample['gt_labels'] = j[i]['gt_labels'].data
        trainset.append(sample)
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        del sample
        assert trainset[-1] is not None



    # for _, j in enumerate(datasets):
    #     print(j)
    #
    #     for i in range(len(j)):
    #         # if i <= 140:
    #         #     continue
    #         print(i)
    #         sample = {}
    #         sample['img_meta'] = j[i]['img_meta'].data
    #         print(sample['img_meta'])
    #         sample['img'] = j[i]['img'].data.tolist()
    #         sample['gt_bboxes'] = j[i]['gt_bboxes'].data
    #         sample['gt_labels'] = j[i]['gt_labels'].data
    #         if i == 143:
    #             print(sample['img'])
    #             print(sample['gt_bboxes'])
    #             print(sample['gt_labels'])
    #         trainset.append(sample)
    #         print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    #         del sample
    #         assert trainset[-1] is not None
    #         if i == 400:
    #             break
    #     del j


    print(len(trainset))
    with open("uda/ds2.pkl", "wb") as f:
        pickle.dump(trainset, f, pickle.HIGHEST_PROTOCOL)

    # with open("uda/file.json", "w") as f:
    #     json.dump(trainset, f, indent=2)

if __name__ == '__main__':
    main()