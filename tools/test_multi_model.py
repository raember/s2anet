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
from PIL.Image import Image
from dateutil.parser import parse
from matplotlib import pyplot as plt
from mmcv import DataLoader, Config
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, load_checkpoint
from pandas import DataFrame

from DeepScoresV2_s2anet.omr_prototype_alignment import prototype_alignment, render
from mmdet.apis import init_dist
from mmdet.core import coco_eval, results2json, wrap_fp16_model, poly_to_rotated_box_single, bbox2result
from mmdet.core import rotated_box_to_poly_np
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector


DEEPSCORES_TEST_SET = {
    'type': 'DeepScoresV2Dataset',
    'ann_file': 'data/deep_scores_dense/deepscores_test.json',
    'img_prefix': 'data/deep_scores_dense/images/',
    'pipeline': [
        {'type': 'LoadImageFromFile'},
        {
            'type': 'MultiScaleFlipAug',
            'img_scale': 0.5,
            'flip': False,
            'transforms': [
                {'type': 'RotatedResize', 'img_scale': 0.5, 'keep_ratio': True},
                {'type': 'RotatedRandomFlip'},
                {'type': 'Normalize', 'mean': [240, 240, 240], 'std': [57, 57, 57], 'to_rgb': False},
                {'type': 'Pad', 'size_divisor': 32},
                {'type': 'ImageToTensor', 'keys': ['img']},
                {'type': 'Collect', 'keys': ['img']}
            ]
        }
    ],
    'use_oriented_bboxes': True
}

IMSLP_TEST_SET = {
    'type': 'DeepScoresV2Dataset',
    'ann_file': 'data/deep_scores_dense/imslp_test.json',
    'img_prefix': 'data/deep_scores_dense/images/',
    'pipeline': [
        {'type': 'LoadImageFromFile'},
        {
            'type': 'MultiScaleFlipAug',
            'img_scale': 1.0,
            'flip': False,
            'transforms': [
                {'type': 'RotatedResize', 'img_scale': 1.0, 'keep_ratio': True},
                {'type': 'RotatedRandomFlip'},
                {'type': 'Normalize', 'mean': [240, 240, 240], 'std': [57, 57, 57], 'to_rgb': False},
                {'type': 'Pad', 'size_divisor': 32},
                {'type': 'ImageToTensor', 'keys': ['img']},
                {'type': 'Collect', 'keys': ['img']}
            ]
        }
    ],
    'use_oriented_bboxes': True
}
TEST_SETS = [DEEPSCORES_TEST_SET, IMSLP_TEST_SET]

class_names = (
    'brace', 'ledgerLine', 'repeatDot', 'segno', 'coda', 'clefG', 'clefCAlto', 'clefCTenor', 'clefF',
    'clefUnpitchedPercussion', 'clef8', 'clef15', 'timeSig0', 'timeSig1', 'timeSig2', 'timeSig3', 'timeSig4',
    'timeSig5', 'timeSig6', 'timeSig7', 'timeSig8', 'timeSig9', 'timeSigCommon', 'timeSigCutCommon',
    'noteheadBlackOnLine', 'noteheadBlackOnLineSmall', 'noteheadBlackInSpace', 'noteheadBlackInSpaceSmall',
    'noteheadHalfOnLine', 'noteheadHalfOnLineSmall', 'noteheadHalfInSpace', 'noteheadHalfInSpaceSmall',
    'noteheadWholeOnLine', 'noteheadWholeOnLineSmall', 'noteheadWholeInSpace', 'noteheadWholeInSpaceSmall',
    'noteheadDoubleWholeOnLine', 'noteheadDoubleWholeOnLineSmall', 'noteheadDoubleWholeInSpace',
    'noteheadDoubleWholeInSpaceSmall', 'augmentationDot', 'stem', 'tremolo1', 'tremolo2', 'tremolo3', 'tremolo4',
    'tremolo5', 'flag8thUp', 'flag8thUpSmall', 'flag16thUp', 'flag32ndUp', 'flag64thUp', 'flag128thUp', 'flag8thDown',
    'flag8thDownSmall', 'flag16thDown', 'flag32ndDown', 'flag64thDown', 'flag128thDown', 'accidentalFlat',
    'accidentalFlatSmall', 'accidentalNatural', 'accidentalNaturalSmall', 'accidentalSharp', 'accidentalSharpSmall',
    'accidentalDoubleSharp', 'accidentalDoubleFlat', 'keyFlat', 'keyNatural', 'keySharp', 'articAccentAbove',
    'articAccentBelow', 'articStaccatoAbove', 'articStaccatoBelow', 'articTenutoAbove', 'articTenutoBelow',
    'articStaccatissimoAbove', 'articStaccatissimoBelow', 'articMarcatoAbove', 'articMarcatoBelow', 'fermataAbove',
    'fermataBelow', 'caesura', 'restDoubleWhole', 'restWhole', 'restHalf', 'restQuarter', 'rest8th', 'rest16th',
    'rest32nd', 'rest64th', 'rest128th', 'restHNr', 'dynamicP', 'dynamicM', 'dynamicF', 'dynamicS', 'dynamicZ',
    'dynamicR', 'graceNoteAcciaccaturaStemUp', 'graceNoteAppoggiaturaStemUp', 'graceNoteAcciaccaturaStemDown',
    'graceNoteAppoggiaturaStemDown', 'ornamentTrill', 'ornamentTurn', 'ornamentTurnInverted', 'ornamentMordent',
    'stringsDownBow', 'stringsUpBow', 'arpeggiato', 'keyboardPedalPed', 'keyboardPedalUp', 'tuplet3', 'tuplet6',
    'fingering0', 'fingering1', 'fingering2', 'fingering3', 'fingering4', 'fingering5', 'slur', 'beam', 'tie',
    'restHBar', 'dynamicCrescendoHairpin', 'dynamicDiminuendoHairpin', 'tuplet1', 'tuplet2', 'tuplet4', 'tuplet5',
    'tuplet7', 'tuplet8', 'tuplet9', 'tupletBracket', 'staff', 'ottavaBracket'
)


GREEN = 32
YELLOW = 33
BLUE = 34
RED = 31


def msg(s: str, level: int, color: int):
    if level == 0:
        print(f"\033[1;{color}m=>\033[m {s}\033[m")
    else:
        print(f"  \033[1;{color}m{'-' * level}>\033[m {s}\033[m")

def msg1(s: str):
    msg(s, 0, GREEN)

def msg2(s: str):
    msg(s, 1, BLUE)

def msg3(s: str):
    msg(s, 2, BLUE)

def msg4(s: str):
    msg(s, 3, BLUE)

WARN = f"\033[1;{YELLOW}m"
def warn(s: str, level: int):
    msg(f"{WARN}{s}\033[m", level, YELLOW)

def warn1(s: str):
    warn(s, 0)

def warn2(s: str):
    warn(s, 1)

def warn3(s: str):
    warn(s, 2)

def warn4(s: str):
    warn(s, 3)

ERR = f"\033[1;{RED}m"
def err(s: str, level: int):
    msg(f"{ERR}{s}\033[m", level, RED)

def err1(s: str):
    err(s, 0)

def err2(s: str):
    err(s, 1)

def err3(s: str):
    err(s, 2)

def err4(s: str):
    err(s, 3)


def _postprocess_bboxes(img, boxes, labels):
    img = Image.fromarray(img)
    proposal_list = [{'proposal': np.append(box[:5], class_names[int(label) + 1])} for box, label in zip(boxes, labels)]
    processed_proposals = prototype_alignment._process_single(img, proposal_list,
                                                              whitelist=["key", "clef", "accidental", "notehead"])
    new_boxes = np.zeros(boxes.shape)
    new_boxes[..., :5] = np.stack(processed_proposals)
    if new_boxes.shape[1] == 6:
        # copy scores
        new_boxes[..., 5] = boxes[..., 5]
    return new_boxes

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
def get_test_set(test_config: str) -> DataLoader:
    cfg = json.loads(test_config)
    msg2(f"Loading \033[1m{Path(cfg['ann_file']).name}\033[m ({cfg['type']})")
    return build_dataset(cfg)

def get_test_sets(*test_configs: Config, workers_per_gpu: int = 4, imgs_per_gpu: int = 1, distributed: bool = False) -> List[DataLoader]:
    data_loaders = []
    for test_config in test_configs:
        data_loaders.append(build_dataloader(
            get_test_set(json.dumps(test_config)),
            imgs_per_gpu=imgs_per_gpu,
            workers_per_gpu=workers_per_gpu,
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

    out_folder = Path('eval')
    proposals_fp = Path(args.out)
    if args.json_out:
        json_out_fp = Path(args.json_out)

    index = []
    stats = {'samples': []}
    msg1("Loading basic test sets")
    stats.update({key: [] for key in get_test_sets(*TEST_SETS)[0].dataset.CLASSES})

    # Make sure we only use the best epochs for each config
    msg1("Preprocessing checkpoints")
    checkpoints = {}
    for checkpoint_file in map(Path, args.checkpoints):
        if not checkpoint_file.exists():
            warn2(f"!!! Checkpoint file \033[1m{str(checkpoint_file)}\033[m{WARN} does not exist")
            continue
        if checkpoint_file.is_file():
            msg2(f"Pre-loading checkpoint \033[1m{str(checkpoint_file)}\033[m")
            model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
            fp16_cfg = cfg.get('fp16', None)
            if fp16_cfg is not None:
                wrap_fp16_model(model)

            try:
                checkpoint = load_checkpoint(model, str(checkpoint_file), map_location='cpu')
            except RuntimeError as e:
                err2(f"!!! Failed loading checkpoint \033[1m{str(checkpoint_file)}\033[m{ERR}: {e}")
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
            msg3(f"Original config: \033[1m{config_file.name}\033[m (epoch: \033[1m{checkpoint['meta']['epoch']}\033[m)")

            other_ckpnt_file, other_ckpnt, other_model = checkpoints.get(config_file.name, (None, None, None))
            if other_ckpnt is not None:
                # Select the model of the same config with the highest epoch
                if int(checkpoint['meta']['epoch']) >= int(other_ckpnt['meta']['epoch']):
                    warn4(f"Replaces model with same config: epoch \033[1m{other_ckpnt['meta']['epoch']}\033[m >= \033[1m{checkpoint['meta']['epoch']}\033[m")
                else:
                    checkpoint_file, checkpoint, model = other_ckpnt_file, other_ckpnt, other_model
                    warn4(f"Ignored model with same config: epoch\033[1m {other_ckpnt['meta']['epoch']}\033[m <= \033[1m{checkpoint['meta']['epoch']}\033[m")
            checkpoints[config_file.name] = checkpoint_file, checkpoint, model
        elif checkpoint_file.is_dir():
            # Multimodel
            msg2(f"Pre-loading multi-models in \033[1m{str(checkpoint_file)}\033[m")
            for multi_checkpoint_file in checkpoint_file.glob('*.pth'):
                raise NotImplementedError()
        else:
            raise Exception(f"Checkpoint {str(checkpoint_file)} is neither a file nor a folder")

    overlaps = np.arange(0.1, 0.96, 0.05)
    overlaps_str = str(overlaps).replace("\n", "")
    outputs_m = {}
    metrics = {}
    got_all_ds_names = False
    dataset_names = []
    for config_file, (checkpoint_file, checkpoint, model) in checkpoints.items():
        checkpoint_file = Path(checkpoint_file)
        cfg_str = checkpoint['meta']['config']
        chkp_cfg = config_from_str(cfg_str)
        config_file = Path(chkp_cfg.filename)
        assert config_file.suffix == '.py'
        epoch = checkpoint['meta']['epoch']
        time = parse(checkpoint['meta']['time'])
        print('#' * 30)
        msg1(f"Loaded checkpoint \033[1m{checkpoint_file.name}\033[m (\033[1m{epoch}\033[m epochs, created: {str(time)})")
        chkp_folder = out_folder / f"{config_file.stem}_epoch_{epoch}"
        chkp_folder.mkdir(parents=True, exist_ok=True)
        # Write checkpoint config to file
        with open(chkp_folder / config_file.name, 'w') as fp:
            fp.write(f"# {cfg_str}")
        msg2(f"Extracted original configuration to \033[1m{config_file.name}\033[m")

        new_chkpnt = chkp_folder / checkpoint_file.name
        if new_chkpnt.is_symlink():
            new_chkpnt.unlink()
        new_chkpnt.symlink_to(os.path.relpath(checkpoint_file, new_chkpnt.parent))

        test_sets = []
        for test_set in TEST_SETS:
            new_test_set = test_set.copy()
            new_test_set['type'] = chkp_cfg.data.test.type
            test_sets.append(new_test_set)

        for data_loader in get_test_sets(
                *test_sets,
                workers_per_gpu=chkp_cfg.data.workers_per_gpu,
                # imgs_per_gpu=chkp_cfg.data.imgs_per_gpu,
                distributed=distributed):
            if not got_all_ds_names:
                dataset_names.append(data_loader.dataset.obb.dataset_info['description'])
            stats['samples'].append(len(data_loader.dataset.img_ids))
            ann_file = Path(data_loader.dataset.ann_file)
            msg2(f"Selecting dataset: \033[1m{ann_file.stem}\033[m")
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
                msg3(f"Testing model on \033[1m{ann_file.stem}\033[m")
                if not distributed:
                    model = MMDataParallel(model, device_ids=[0])
                    outputs_m[checkpoint_file][data_loader] = single_gpu_test(model, data_loader, args.show, cfg,
                                                                              args.postprocess, args.round)
                else:
                    model = MMDistributedDataParallel(model.cuda())
                    outputs_m[checkpoint_file][data_loader] = multi_gpu_test(model, data_loader, args.tmpdir, cfg,
                                                                             args.postprocess, args.round)
                print()  # The tests use ncurses and don't append a new line at the end

                msg3(f'Writing proposals to \033[1m{str(pkl_fp)}\033[m')
                mmcv.dump(outputs_m[checkpoint_file][data_loader], pkl_fp)
            else:
                msg3(f'Reading proposals from \033[1m{str(pkl_fp)}\033[m')
                outputs_m[checkpoint_file][data_loader] = mmcv.load(pkl_fp)
            eval_types = args.eval
            data_name = args.data
            if data_name == 'coco':
                if eval_types:
                    if eval_types == ['proposal_fast']:
                        result_file = args.out
                        coco_eval(result_file, eval_types, data_loader.dataset.coco)
                    else:
                        if not isinstance(outputs_m[checkpoint_file][data_loader][0], dict):
                            result_files = results2json(data_loader.dataset, outputs_m[checkpoint_file][data_loader], args.out)
                            coco_eval(result_files, eval_types, data_loader.dataset.coco)
                        else:
                            for name in outputs_m[checkpoint_file][data_loader][0]:
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
                    msg3(f"Evaluating: \033[1m{str(checkpoint_file)}\033[m on \033[1m{data_loader.dataset.ann_file}\033[m in \033[1m{str(result_folder)}\033[m")
                    data_loader.dataset.evaluate(
                        outputs_m[checkpoint_file][data_loader],
                        result_json_filename=result_file,
                        work_dir=str(result_folder),
                        iou_thrs=overlaps
                    )  # Extremely slow...
                msg3(f'Reading calculated metrics from \033[1m{str(metrics_fp)}\033[m')
                metrics = mmcv.load(metrics_fp)

                if result_file is not None and not (result_folder / 'proposal_stats.csv').exists():
                    msg3(f"Calculating statistics for results")
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

            msg3(f'Compiling metrics with overlaps \033[1m{overlaps_str}\033[m')
            for cls, overlap_metrics in metrics.items():
                for overlap in overlaps:
                    overlap_metrics[overlap] = overlap_metrics[overlap].get('ap', np.NaN)
                metrics[cls] = overlap_metrics
            for cls in data_loader.dataset.CLASSES:
                stats[cls].append(metrics.get(cls, {}))

            # Save predictions in the COCO json format
            rank, _ = get_dist_info()
            if args.json_out and rank == 0:
                result_file = result_folder / args.json_out
                if not result_file.with_suffix('.bbox.json').exists() or not args.cache:
                    msg3(f"Saving predictions to \033[1m{str(result_file.with_suffix('.*'))}\033[m")
                    if not isinstance(outputs_m[checkpoint_file][data_loader][0], dict):
                        results2json(data_loader.dataset, outputs_m[checkpoint_file][data_loader], result_file.with_suffix(''))
                    else:
                        for name in outputs_m[checkpoint_file][data_loader][0]:
                            outputs_ = [out[name] for out in outputs_m[checkpoint_file][data_loader]]
                            results2json(data_loader.dataset, outputs_, result_file.with_suffix(f'.{name}{result_file.suffix}'))
        got_all_ds_names = True
    print('#' * 30)
    print('#' * 30)
    print('#' * 30)
    msg1(f"Evaluating stats")
    for overlap in overlaps:
        msg2(f"Processing stats for overlap = \033[1m{overlap:.2f}\033[m")
        eval_fp = out_folder / f'eval_{overlap:.2f}.csv'
        overlap_stats = {}
        for cls, overlap_aps in stats.items():
            if cls == 'samples':
                overlap_stats['samples'] = stats['samples']
            else:
                overlap_stats[cls] = []
                for overlap_ap in overlap_aps:
                    overlap_stats[cls].append(overlap_ap.get(overlap, np.NaN))
        stat_df = DataFrame(overlap_stats, index=index)
        msg2(f"Saving stats to \033[1m{eval_fp}\033[m")
        stat_df.to_csv(eval_fp)

        print('=' * 30)
        CLASSES = {
            'clefs': {'clefG', 'clefCAlto', 'clefCTenor', 'clefF', 'clef8', 'clef15'},
            'noteheads': {'noteheadBlackOnLine', 'noteheadBlackInSpace', 'noteheadHalfOnLine', 'noteheadHalfInSpace', 'noteheadWholeOnLine', 'noteheadWholeInSpace', 'noteheadDoubleWholeOnLine','noteheadDoubleWholeInSpace'},
            'accidentals': {'accidentalFlat', 'accidentalNatural', 'accidentalSharp', 'accidentalDoubleSharp', 'accidentalDoubleFlat'},
            'keys': {'keyFlat', 'keyNatural', 'keySharp'},
            'rests': {'restDoubleWhole', 'restWhole', 'restHalf', 'restQuarter', 'rest8th', 'rest16th', 'rest32nd', 'rest64th', 'rest128th'},
            'beams': {'beam'},
            'all classes': set(class_names)
        }
        n_datasets = len(dataset_names)
        chkpnt_names = [s.split(' - ')[0] for s in stat_df.index[::n_datasets]]
        for name, classes in CLASSES.items():
            msg3(f"Plotting \033[1m{name}\033[m")
            substats = stat_df[classes]
            all_aps = substats.to_numpy()
            # Only use columns where there is no NaN values
            non_nan_aps = all_aps.T[~np.isnan(all_aps.sum(axis=0))].T
            if non_nan_aps.shape[1] == 0:
                msg4("No values to compare")
                continue
            mean_aps = non_nan_aps.mean(axis=1).reshape((len(chkpnt_names), n_datasets))
            fig, ax = plt.subplots(figsize=(15, 9))
            X = np.arange(len(chkpnt_names))
            incr = 0.4
            center_offset = (incr * (len(dataset_names) - 1))/2
            for i, (ds_name, col, mean_ap) in enumerate(zip(dataset_names, itertools.cycle(['b', 'r', 'g', 'y', 'c', 'm']), mean_aps.T)):
                r = ax.bar(X + incr * i - center_offset, mean_ap, color=col, width=incr, label=f'{ds_name} ({stat_df["samples"][i]} samples)')
                ax.bar_label(r, padding=3)
            ax.set_ylabel('AP')
            ax.set_title(f'AP of {name} by model and training set (overlap = {overlap:.2f})')
            plt.xticks(X, chkpnt_names, rotation=10, horizontalalignment='right', fontsize='small')
            ax.legend()
            fig.tight_layout()
            im_fp = out_folder / f'AP_{name.replace(" ", "_")}_{overlap:.2f}.png'
            plt.savefig(im_fp)
            msg3(f'Saved plot to \033[1m{str(im_fp)}\033[m')
            #plt.show()


if __name__ == '__main__':
    main()
