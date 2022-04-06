import argparse
import itertools
import os
import os.path as osp
import shutil
import tempfile
from pathlib import Path
from typing import Tuple

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from dateutil.parser import parse
from matplotlib import pyplot as plt
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, load_checkpoint, _load_checkpoint
from pandas import DataFrame

from mmdet.apis import init_dist
from mmdet.core import coco_eval, results2json, wrap_fp16_model
from mmdet.core import rotated_box_to_poly_np
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector


# Code based on test_BE.py

def single_gpu_test(model, data_loader, show=False, cfg=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result, bbox_list = model(return_loss=False, rescale=not show, **data)
        results.append(result)
        if show:
            print("asdf")
            # for nr, sub_list in enumerate(bbox_list):
            #    bbox_list[nr] = [rotated_box_to_poly_np(sub_list[0].cpu().numpy()), sub_list[1].cpu().numpy()]

            model.module.show_result(data, result, show=show, dataset=dataset.CLASSES,
                                     bbox_transform=rotated_box_to_poly_np, score_thr=cfg.test_cfg['score_thr'])
            # typo in bbox_transorm -> bbox_transform?

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)

    return results


def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN,),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoints', nargs='+',
                        help='checkpoint files', required=True)
    parser.add_argument('--test-sets', nargs='+',
                        help='test set paths')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--json_out',
        help='output result file name without extension',
        type=str)
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    # add dataset type for more dataset eval other than coco
    parser.add_argument(
        '--data',
        choices=['coco', 'dota', 'dota_large', 'dota_hbb', 'hrsc2016', 'voc', 'dota_1024', 'dsv2'],
        default='dota',
        type=str,
        help='eval dataset type')
    parser.add_argument(
        '--cache',
        action='store_true',
        default=False,
        help='Use cached results/metrics/evaluations instead of recalculating'
    )
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.show or args.json_out, \
        ('Please specify at least one operation (save or show the results) '
         'with the argument "--out" or "--show" or "--json_out"')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    # cfg.model.rpn_pretrained = None
    # cfg.model.rcnn_pretrained = None

    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    data_loaders = []
    if args.test_sets is not None:
        for path in map(Path, args.test_sets):
            assert path.exists(), f"Test set does not exist at {str(path)}"
            tds =  cfg.data.test.deepcopy()
            tds.ann_file = str(path)
            data_loaders.append(build_dataloader(
                build_dataset(tds),
                imgs_per_gpu=1,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=distributed,
                shuffle=False
            ))
    else:
        data_loaders.append(build_dataloader(
            build_dataset(cfg.data.test),
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False
        ))

    out_folder = Path('eval')
    proposals_fp = Path(args.out)
    if args.json_out:
        json_out_fp = Path(args.json_out)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    index = []
    stats = {'samples': []}
    stats.update({key: [] for key in data_loaders[0].dataset.CLASSES})

    outputs_m = []
    for i, checkpoint_file in enumerate(map(Path, args.checkpoints)):
        outputs_m.append([])
        checkpoint = load_checkpoint(model, str(checkpoint_file), map_location='cpu')
        cfg_str = checkpoint['meta']['config']
        config_file = Path(cfg_str.splitlines()[0])
        assert config_file.suffix == '.py'
        epoch = checkpoint['meta']['epoch']
        time = parse(checkpoint['meta']['time'])
        print('#' * 30)
        print(f"=> Loaded checkpoint {checkpoint_file.name} ({epoch} epochs, created: {str(time)})")
        chkp_folder = out_folder / f"{config_file.stem}_epoch_{epoch}"
        chkp_folder.mkdir(parents=True, exist_ok=True)
        # Write checkpoint config to file
        with open(chkp_folder / config_file.name, 'w') as fp:
            fp.write(f"# {cfg_str}")
        print(f"==> Extracted original configuration to {config_file.name}")
        new_chkpnt = chkp_folder / checkpoint_file.name
        if new_chkpnt.is_symlink():
            new_chkpnt.unlink()
        new_chkpnt.symlink_to(os.path.relpath(checkpoint_file, new_chkpnt.parent))
        for j, data_loader in enumerate(data_loaders):
            stats['samples'].append(len(data_loader.dataset.img_ids))
            ann_file = Path(data_loader.dataset.ann_file)
            print(f"==> Selecting dataset: {ann_file.stem}")
            index.append(f"{config_file.stem}_epoch_{epoch} - {ann_file.stem}")
            result_folder = chkp_folder / ann_file.stem
            result_folder.mkdir(exist_ok=True)

            # old versions did not save class info in checkpoints, this walkaround is
            # for backward compatibility
            if 'CLASSES' in checkpoint['meta']:
                model.CLASSES = checkpoint['meta']['CLASSES']
            else:
                model.CLASSES = data_loader.dataset.CLASSES

            pkl_fp = result_folder / proposals_fp
            if not pkl_fp.exists() or not args.cache:
                print(f"===> Testing model on {ann_file.stem}")
                if not distributed:
                    model = MMDataParallel(model, device_ids=[0])
                    outputs_m[i].append(single_gpu_test(model, data_loader, args.show, cfg))
                else:
                    model = MMDistributedDataParallel(model.cuda())
                    outputs_m[i].append(multi_gpu_test(model, data_loader, args.tmpdir))
                print()  # The tests use ncurses and don't append a new line at the end

                print(f'===> Writing proposals to {str(pkl_fp)}')
                mmcv.dump(outputs_m[i][j], pkl_fp)
            else:
                print(f'===> Reading proposals from {str(pkl_fp)}')
                outputs_m[i].append(mmcv.load(pkl_fp))
            eval_types = args.eval
            data_name = args.data
            eval_data = {}
            if data_name == 'coco':
                if eval_types:
                    print('Starting evaluate {}'.format(' and '.join(eval_types)))
                    if eval_types == ['proposal_fast']:
                        result_file = args.out
                        coco_eval(result_file, eval_types, data_loader.dataset.coco)
                    else:
                        if not isinstance(outputs_m[i][j][0], dict):
                            result_files = results2json(data_loader.dataset, outputs_m[i][j], args.out)
                            coco_eval(result_files, eval_types, data_loader.dataset.coco)
                        else:
                            for name in outputs_m[i][j][0]:
                                print('\nEvaluating {}'.format(name))
                                outputs_m[i][j] = [out[name] for out in outputs_m[i][j]]
                                result_file = args.out + '.{}'.format(name)
                                result_files = results2json(data_loader.dataset, outputs_m[i][j],
                                                            result_file)
                                coco_eval(result_files, eval_types, data_loader.dataset.coco)
            elif data_name in ['dota', 'hrsc2016']:
                eval_kwargs = cfg.get('evaluation', {}).copy()
                work_dir = osp.dirname(args.out)
                data_loader.dataset.evaluate(outputs_m[i][j], work_dir=work_dir, **eval_kwargs)
            elif data_name in ['dsv2']:
                from mmdet.core import outputs_rotated_box_to_poly_np

                for page in outputs_m[i][j]:
                    page.insert(0, np.array([]))

                outputs_m[i][j] = outputs_rotated_box_to_poly_np(outputs_m[i][j])

                eval_fp = result_folder / "dsv2_metrics.pkl"
                if not eval_fp.exists() or not args.cache:
                    data_loader.dataset.evaluate(
                        outputs_m[i][j],
                        result_json_filename=str(result_folder / "result.json"),
                        work_dir=str(eval_fp.parent)
                    )  # Extremely slow...
                eval_data = mmcv.load(eval_fp)

            overlap = 0.5
            for cls in data_loader.dataset.CLASSES:
                stats[cls].append(eval_data.get(cls, {}).get(overlap, {}).get('ap', np.NaN))

            # Save predictions in the COCO json format
            rank, _ = get_dist_info()
            if args.json_out and rank == 0:
                result_file = result_folder / args.json_out
                if not result_file.with_suffix('.bbox.json').exists() or not args.cache:
                    print(f"===> Saving predictions to {str(result_file.with_suffix('.*'))}")
                    if not isinstance(outputs_m[i][j][0], dict):
                        results2json(data_loader.dataset, outputs_m[i][j], result_file.with_suffix(''))
                    else:
                        for name in outputs_m[i][j][0]:
                            outputs_ = [out[name] for out in outputs_m[i][j]]
                            results2json(data_loader.dataset, outputs_, result_file.with_suffix(f'.{name}{result_file.suffix}'))
    eval_fp = out_folder / 'eval.csv'
    print(f"=> Saving stats to {eval_fp}")
    stat_df = DataFrame(stats, index=index)
    stat_df.to_csv(eval_fp)

    print('=' * 30)
    CLASSES = {
        'clefs': {'clefG', 'clefCAlto', 'clefCTenor', 'clefF', 'clef8', 'clef15'},
        'noteheads': {'noteheadBlackOnLine', 'noteheadBlackInSpace', 'noteheadHalfOnLine', 'noteheadHalfInSpace', 'noteheadWholeOnLine', 'noteheadWholeInSpace', 'noteheadDoubleWholeOnLine','noteheadDoubleWholeInSpace'},
        'accidentals': {'accidentalFlat', 'accidentalNatural', 'accidentalSharp', 'accidentalDoubleSharp', 'accidentalDoubleFlat'},
        'keys': {'keyFlat', 'keyNatural', 'keySharp'},
        'rests': {'restDoubleWhole', 'restWhole', 'restHalf', 'restQuarter', 'rest8th', 'rest16th', 'rest32nd', 'rest64th', 'rest128th'},
        'beams': {'beam'},
        'all classes': set(data_loaders[0].dataset.CLASSES)
    }
    dataset_names = list(map(lambda dsl: dsl.dataset.obb.dataset_info['description'], data_loaders))
    n_datasets = len(dataset_names)
    chkpnt_names = [s.split(' - ')[0] for s in stat_df.index[::n_datasets]]
    for name, classes in CLASSES.items():
        print(f"==> Plotting {name}")
        substats = stat_df[classes]
        arr = substats.to_numpy()
        # Only use columns where there is no NaN values
        arr = arr.T[~np.isnan(arr.sum(axis=0))].T
        if arr.shape[1] == 0:
            print("    - No values to compare")
            continue
        arr = arr.mean(axis=1).reshape((len(chkpnt_names), n_datasets))
        fig, ax = plt.subplots(figsize=(15, 9))
        X = np.arange(len(chkpnt_names))
        incr = 1.0/(len(chkpnt_names)+1)
        center_offset = (incr * (len(dataset_names) - 1))/2
        for i, (ds_name, col) in enumerate(zip(dataset_names, itertools.cycle(['b', 'r', 'g', 'y', 'c', 'm']))):
            r = ax.bar(X + incr * i - center_offset, arr[i], color=col, width=incr, label=f'{ds_name} ({stat_df["samples"][i]} samples)')
            ax.bar_label(r, padding=3)
        ax.set_ylabel('AP')
        ax.set_title(f'AP of {name} by model and training set')
        plt.xticks(X, chkpnt_names, rotation=10, horizontalalignment='right', fontsize='small')
        ax.legend()
        fig.tight_layout()
        im_fp = out_folder / f'AP_{name.replace(" ", "_")}.png'
        plt.savefig(im_fp)
        print(f'===> Saved plot to {str(im_fp)}')
        plt.show()


if __name__ == '__main__':
    main()
