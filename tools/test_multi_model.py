import argparse
import itertools
import json
import math
import os
import os.path as osp
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import List

import cv2
import mmcv
import numpy as np
import pandas
import torch
import torch.distributed as dist
from dateutil.parser import parse
from matplotlib import pyplot as plt
from mmcv import DataLoader, Config
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, load_checkpoint
from pandas import DataFrame

from DeepScoresV2_s2anet.omr_prototype_alignment import render
from mmdet.apis import init_dist
from mmdet.core import coco_eval, results2json, wrap_fp16_model, poly_to_rotated_box_single, bbox2result
from mmdet.core import rotated_box_to_poly_np
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from tools.detection_service import _postprocess_bboxes

def _post_process_bbox_list(img, bbox_list, cfg):
    img = img.cpu().numpy().astype("uint8")
    img = img.transpose(1, 2, 0)
    scale = 1 / cfg['test_pipeline'][1]['img_scale']
    img = cv2.resize(img, dsize=(int(img.shape[1] * scale), int(img.shape[0] * scale)))
    boxes = bbox_list[0][0].cpu().numpy()
    labels = bbox_list[0][1].cpu().numpy()

    boxes_new = _postprocess_bboxes(img, boxes, labels)
    boxes_new = torch.from_numpy(boxes_new)
    return boxes_new


def round_results(result):
    result[:, :4] = torch.round(result[:, :4])
    result[:, 5] = torch.round(result[:, 5] * 1000) / 1000
    return result


def single_gpu_test(model, data_loader, show=False, cfg=None, post_process=False, round_=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result, bbox_list = model(return_loss=False, rescale=not show, **data)

        if post_process:
            img = data['img'][0][0]
            boxes = _post_process_bbox_list(img, bbox_list, cfg)
        else:
            boxes = bbox_list[0][0]

        if round_:
            boxes = round_results(boxes)

        result = bbox2result(boxes, bbox_list[0][1], num_classes=cfg['model']['bbox_head']['num_classes'])

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


def multi_gpu_test(model, data_loader, tmpdir=None, cfg=None, post_process=False, round_=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result, bbox_list = model(return_loss=False, rescale=True, **data)

        if post_process:
            img = data['img'][0][0]
            boxes = _post_process_bbox_list(img, bbox_list, cfg)
        else:
            boxes = bbox_list[0][0]

        if round_:
            boxes = round_results(boxes)

        result = bbox2result(boxes, bbox_list[0][1], num_classes=cfg['model']['bbox_head']['num_classes'])

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
        '--cache', '-c',
        action='store_true',
        default=False,
        help='Use cached results/metrics/evaluations instead of recalculating'
    )
    parser.add_argument(
        '--overlap', '-o',
        choices=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        default=0.5,
        type=float,
        help='Overlap to choose the metrics from (default: 0.5)'
    )
    parser.add_argument(
        '--postprocess',
        action='store_true',
        default=False,
        help='post-process the results'
    )
    parser.add_argument(
        '--round',
        action='store_true',
        default=False,
        help='round the results (similar to detection service)'
    )
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def config_from_str(cfg_str: str, path: Path = None) -> Config:
    if path is None:
        path = Path(cfg_str.splitlines()[0])
    if cfg_str.startswith(str(path)):
        cfg_str = cfg_str[len(str(path)) + 1:]
    with tempfile.NamedTemporaryFile('w', suffix='.py') as fp:
        fp.write(cfg_str)
        fp.seek(0)
        config = Config.fromfile(fp.name)
        cfg_txt = str(config.text).replace(fp.name + "\n", '')
    return Config(getattr(config, '_cfg_dict'), cfg_txt, str(path))


@lru_cache()
def get_test_sets(config: Config, dataset_type: str, distributed: bool, *test_sets: Path) -> List[DataLoader]:
    data_loaders = []
    for path in test_sets:
        assert path.exists(), f"Test set does not exist at {str(path)}"
        tds = config.data.test.deepcopy()
        tds.ann_file = str(path)
        tds.type = dataset_type
        data_loaders.append(build_dataloader(
            build_dataset(tds),
            imgs_per_gpu=1,
            workers_per_gpu=config.data.workers_per_gpu,
            dist=distributed,
            shuffle=False
        ))
    return data_loaders


def main():
    args = parse_args()

    assert args.out or args.show or args.json_out, \
        ('Please specify at least one operation (save or show the results) '
         'with the argument "--out" or "--show" or "--json_out"')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    if args.postprocess:
        render.fill_cache()

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

    if args.test_sets is not None:
        test_sets = list(map(Path, args.test_sets))
    else:
        test_sets = [Path(args.test_sets)]

    out_folder = Path('eval')
    proposals_fp = Path(args.out)
    if args.json_out:
        json_out_fp = Path(args.json_out)

    index = []
    stats = {'samples': []}
    stats.update({key: [] for key in get_test_sets(cfg, cfg.dataset_type, distributed, *test_sets)[0].dataset.CLASSES})

    # Make sure we only use the best epochs for each config
    checkpoints = {}
    for checkpoint_file in map(Path, args.checkpoints):
        if not checkpoint_file.exists():
            print(f"!!! Checkpoint file {str(checkpoint_file)} does not exist")
            continue
        model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)

        try:
            checkpoint = load_checkpoint(model, str(checkpoint_file), map_location='cpu')
        except RuntimeError as e:
            print(f"!!! Failed loading checkpoint {str(checkpoint_file)}: {e}")
            continue

        cfg_str = checkpoint['meta']['config']
        chkp_cfg = config_from_str(cfg_str)
        config_file = Path(chkp_cfg.filename)
        parents = [config_file.name]
        for parent in config_file.parents:
            parents.append(parent.name)
            if parent.name == 'configs':
                break
            if parent.name == '/':
                raise Exception("Did not find configs dir")
        config_file = Path(*reversed(parents))

        other_ckpnt_file, other_ckpnt, other_model = checkpoints.get(str(config_file), (None, None, None))
        if other_ckpnt is not None and int(other_ckpnt['meta']['epoch']) >= int(checkpoint['meta']['epoch']):
            checkpoint_file, checkpoint, model = other_ckpnt_file, other_ckpnt, other_model
        checkpoints[str(config_file)] = checkpoint_file, checkpoint, model

    overlaps = np.arange(0.1, 0.96, 0.05)
    overlap = overlaps[np.isclose(overlaps, args.overlap)].tolist()[0]
    outputs_m = {}
    metrics = {}
    got_all_ds_names = False
    dataset_names = []
    for config_file, (checkpoint_file, checkpoint, model) in checkpoints.items():
        # # build the model and load checkpoint
        # model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        # fp16_cfg = cfg.get('fp16', None)
        # if fp16_cfg is not None:
        #     wrap_fp16_model(model)
        #
        # outputs_m[checkpoint_file] = {}
        # try:
        #     checkpoint = load_checkpoint(model, str(checkpoint_file), map_location='cpu')
        # except RuntimeError as e:
        #     raise Exception(f"Failed loading checkpoint {str(checkpoint_file)}: {e}")
        checkpoint_file = Path(checkpoint_file)
        cfg_str = checkpoint['meta']['config']
        chkp_cfg = config_from_str(cfg_str)
        config_file = Path(chkp_cfg.filename)
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

        for data_loader in get_test_sets(cfg, chkp_cfg.data.test.type, distributed, *test_sets):
            if not got_all_ds_names:
                dataset_names.append(data_loader.dataset.obb.dataset_info['description'])
            stats['samples'].append(len(data_loader.dataset.img_ids))
            ann_file = Path(data_loader.dataset.ann_file)
            print(f"==> Selecting dataset: {ann_file.stem}")
            index.append(f"{config_file.stem}_epoch_{epoch} - {ann_file.stem}")
            result_folder = chkp_folder / ann_file.stem
            result_folder.mkdir(exist_ok=True)

            # old versions did not save class info in checkpoints, this workaround is
            # for backward compatibility
            if 'CLASSES' in checkpoint['meta']:
                model.CLASSES = checkpoint['meta']['CLASSES']
            else:
                model.CLASSES = data_loader.dataset.CLASSES

            pkl_fp = result_folder / proposals_fp
            outputs_m[checkpoint_file] = {}
            if not pkl_fp.exists() or not args.cache:
                pkl_fp.parent.mkdir(exist_ok=True)
                print(f"===> Testing model on {ann_file.stem}")
                if not distributed:
                    model = MMDataParallel(model, device_ids=[0])
                    outputs_m[checkpoint_file][data_loader] = single_gpu_test(model, data_loader, args.show, cfg,
                                                                              args.postprocess, args.round)
                else:
                    model = MMDistributedDataParallel(model.cuda())
                    outputs_m[checkpoint_file][data_loader] = multi_gpu_test(model, data_loader, args.tmpdir, cfg,
                                                                             args.postprocess, args.round)
                print()  # The tests use ncurses and don't append a new line at the end

                print(f'===> Writing proposals to {str(pkl_fp)}')
                mmcv.dump(outputs_m[checkpoint_file][data_loader], pkl_fp)
            else:
                print(f'===> Reading proposals from {str(pkl_fp)}')
                outputs_m[checkpoint_file][data_loader] = mmcv.load(pkl_fp)
            eval_types = args.eval
            data_name = args.data
            if data_name == 'coco':
                if eval_types:
                    print('Starting evaluate {}'.format(' and '.join(eval_types)))
                    if eval_types == ['proposal_fast']:
                        result_file = args.out
                        coco_eval(result_file, eval_types, data_loader.dataset.coco)
                    else:
                        if not isinstance(outputs_m[checkpoint_file][data_loader][0], dict):
                            result_files = results2json(data_loader.dataset, outputs_m[checkpoint_file][data_loader], args.out)
                            coco_eval(result_files, eval_types, data_loader.dataset.coco)
                        else:
                            for name in outputs_m[checkpoint_file][data_loader][0]:
                                print('\nEvaluating {}'.format(name))
                                outputs_m[checkpoint_file][data_loader] = [out[name] for out in outputs_m[checkpoint_file][data_loader]]
                                result_file = args.out + '.{}'.format(name)
                                result_files = results2json(data_loader.dataset, outputs_m[checkpoint_file][data_loader],
                                                            result_file)
                                coco_eval(result_files, eval_types, data_loader.dataset.coco)
            elif data_name in ['dota', 'hrsc2016']:
                eval_kwargs = cfg.get('evaluation', {}).copy()
                work_dir = osp.dirname(args.out)
                data_loader.dataset.evaluate(outputs_m[checkpoint_file][data_loader], work_dir=work_dir, **eval_kwargs)
            elif data_name in ['dsv2']:
                from mmdet.core import outputs_rotated_box_to_poly_np

                for page in outputs_m[checkpoint_file][data_loader]:
                    page.insert(0, np.array([]))

                outputs_m[checkpoint_file][data_loader] = outputs_rotated_box_to_poly_np(outputs_m[checkpoint_file][data_loader])

                metrics_fp = result_folder / "dsv2_metrics.pkl"
                result_file = None
                if args.json_out is not None:
                    result_file = str(result_folder / args.json_out)
                if not metrics_fp.exists() or not args.cache:
                    print(f"### EVALUATE: {str(checkpoint_file)} ON {data_loader.dataset.ann_file} IN {str(result_folder)}")
                    data_loader.dataset.evaluate(
                        outputs_m[checkpoint_file][data_loader],
                        result_json_filename=result_file,
                        work_dir=str(result_folder),
                        iou_thrs=overlaps
                    )  # Extremely slow...
                print(f'===> Reading calculated metrics from {str(metrics_fp)}')
                metrics = mmcv.load(metrics_fp)

                if result_file is not None and not (result_folder / 'proposal_stats.csv').exists():
                    print(f"===> Calculating statistics for results")
                    with open(result_file, 'r') as fp:
                        proposals = json.load(fp)
                    prop_stats = {}
                    for proposal in proposals['proposals']:
                        cat_id = int(proposal['cat_id'])
                        cat = data_loader.dataset.CLASSES[cat_id - 1]
                        x, y, w, h, a = poly_to_rotated_box_single(list(map(float, proposal['bbox'])))
                        a *= 180.0/math.pi
                        prop_stats[cat] = prop_stats.get(cat, []) + [a]
                    prop_data = {}
                    for i, cat in enumerate(data_loader.dataset.CLASSES):
                        angles = prop_stats.get(cat, [])
                        prop_data[cat] = (len(angles), np.mean(angles), np.std(angles))
                        #print(f'[{i + 1}] {cat} ({len(angles)}): avg:{np.mean(angles):.02f}, std: {np.std(angles):.02f}')
                    csv_data = pandas.DataFrame(prop_data, index=('occurrences', 'avg', 'std')).transpose()
                    csv_data.to_csv(result_folder / 'proposal_stats.csv')

            print(f'===> Compiling metrics with overlap {args.overlap}')
            for cls in data_loader.dataset.CLASSES:
                stats[cls].append(metrics.get(cls, {}).get(overlap, {}).get('ap', np.NaN))

            # Save predictions in the COCO json format
            rank, _ = get_dist_info()
            if args.json_out and rank == 0:
                result_file = result_folder / args.json_out
                if not result_file.with_suffix('.bbox.json').exists() or not args.cache:
                    print(f"===> Saving predictions to {str(result_file.with_suffix('.*'))}")
                    if not isinstance(outputs_m[checkpoint_file][data_loader][0], dict):
                        results2json(data_loader.dataset, outputs_m[checkpoint_file][data_loader], result_file.with_suffix(''))
                    else:
                        for name in outputs_m[checkpoint_file][data_loader][0]:
                            outputs_ = [out[name] for out in outputs_m[checkpoint_file][data_loader]]
                            results2json(data_loader.dataset, outputs_, result_file.with_suffix(f'.{name}{result_file.suffix}'))
        got_all_ds_names = True
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
        'all classes': set(data_loader.dataset.CLASSES)
    }
    n_datasets = len(dataset_names)
    chkpnt_names = [s.split(' - ')[0] for s in stat_df.index[::n_datasets]]
    for name, classes in CLASSES.items():
        print(f"==> Plotting {name}")
        substats = stat_df[classes]
        all_aps = substats.to_numpy()
        # Only use columns where there is no NaN values
        non_nan_aps = all_aps.T[~np.isnan(all_aps.sum(axis=0))].T
        if non_nan_aps.shape[1] == 0:
            print("    - No values to compare")
            continue
        mean_aps = non_nan_aps.mean(axis=1).reshape((len(chkpnt_names), n_datasets))
        fig, ax = plt.subplots(figsize=(15, 9))
        X = np.arange(len(chkpnt_names))
        incr = 1.0/(len(chkpnt_names)+1)
        center_offset = (incr * (len(dataset_names) - 1))/2
        for i, (ds_name, col, mean_ap) in enumerate(zip(dataset_names, itertools.cycle(['b', 'r', 'g', 'y', 'c', 'm']), mean_aps.T)):
            r = ax.bar(X + incr * i - center_offset, mean_ap, color=col, width=incr, label=f'{ds_name} ({stat_df["samples"][i]} samples)')
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
