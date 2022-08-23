from argparse import ArgumentParser, Namespace
import itertools
import json
import math
import os
import os.path as osp
import shutil
import tempfile
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Generator

import cv2
import mmcv
import numpy as np
import pandas
import pandas as pd
import torch
import torch.distributed as dist
from PIL import Image as PImage
from PIL.Image import Image
from dateutil.parser import parse
from matplotlib import pyplot as plt
from mmcv import DataLoader, Config
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, load_checkpoint
from pandas import DataFrame
from torch.nn import Sequential

from DeepScoresV2_s2anet.analyze_ensembles.draw_WBF_for_multi_model import BboxHelper, load_proposals
from DeepScoresV2_s2anet.analyze_ensembles.wbf_rotated_boxes import rotated_weighted_boxes_fusion
from DeepScoresV2_s2anet.omr_prototype_alignment import prototype_alignment, render
from mmdet.apis import init_dist
from mmdet.core import coco_eval, results2json, wrap_fp16_model, poly_to_rotated_box_single, bbox2result, \
    outputs_rotated_box_to_poly_np
from mmdet.core import rotated_box_to_poly_np
from mmdet.datasets import build_dataloader, build_dataset, DeepScoresV2Dataset
from mmdet.models import build_detector


# models/dsv2_no_augment/DS_2022_04_26/epoch_2000.pth
# models/dsv2/DS_2022_04_26/epoch_2000.pth
# models/dsv2_finalize/DS_2022_05_27/epoch_200.pth
# models/dsv2_finalize_snp/DS_2022_05_27/epoch_250.pth
# models/dsv2hybrid/DS_2022_05_27/epoch_600.pth
# models/dsv2hybrid_finalize_snp/DS_2022_05_27/epoch_150.pth
# models/uda_aug/halfrez_crop_lr_d_0.001_pretrained/model_030.pth
# models/uda_aug/halfrez_crop_lr0.0025/model_140.pth
# models/uda_aug/halfrez_crop_lr_d_0.001_no-pretrained/model_030.pth
# models/uda_aug/halfrez_crop_lr0.0075/model_det_900.pth
# models/uda_aug/halfrez_crop_adam/epoch_200.pth
# models/uda_aug/imslp_04_11/model_030.pth
# models/ensemble_cycle_length_20

# models/uda_aug/halfrez_crop_lr0.0075/model_det_800.pth
# models/uda_aug/halfrez_crop_lr0.0075/model_det_300.pth
# models/uda_aug/halfrez_crop_lr0.0075/model_det_700.pth
# models/uda_aug/halfrez_crop_lr0.0075/model_det_600.pth
# models/uda_aug/halfrez_crop_lr0.0075/model_det_200.pth
# models/uda_aug/halfrez_crop_lr0.0075/model_det_400.pth
# models/uda_aug/halfrez_crop_lr0.0075/model_det_500.pth
# models/uda_aug/halfrez_crop_lr0.0075/model_det_100.pth


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
    img = PImage.fromarray(img)
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
    parser = ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoints', nargs='+',
                        help='checkpoint files', required=True)
    parser.add_argument('--out', help='output result file', default='eval.pkl')
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
        '--postprocess', '-p',
        action='store_true',
        default=False,
        help='post-process the results'
    )
    parser.add_argument(
        '--round', '-r',
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


def create_plots(stats: DataFrame, dataset_names: List[str], overlap: np.float, folder: Path):
    msg3("Plotting classes")
    CLASSES = {
        'clefs': {'clefG', 'clefCAlto', 'clefCTenor', 'clefF', 'clef8', 'clef15'},
        'noteheads': {'noteheadBlackOnLine', 'noteheadBlackInSpace', 'noteheadHalfOnLine', 'noteheadHalfInSpace',
                      'noteheadWholeOnLine', 'noteheadWholeInSpace', 'noteheadDoubleWholeOnLine',
                      'noteheadDoubleWholeInSpace'},
        'accidentals': {'accidentalFlat', 'accidentalNatural', 'accidentalSharp', 'accidentalDoubleSharp',
                        'accidentalDoubleFlat'},
        'keys': {'keyFlat', 'keyNatural', 'keySharp'},
        'rests': {'restDoubleWhole', 'restWhole', 'restHalf', 'restQuarter', 'rest8th', 'rest16th', 'rest32nd',
                  'rest64th', 'rest128th'},
        'beams': {'beam'},
        'all classes': set(class_names)
    }
    n_datasets = len(dataset_names)
    chkpnt_names = [s.split(' - ')[0] for s in stats.index[::n_datasets]]
    for name, classes in CLASSES.items():
        im_fp = folder / f'AP_{name.replace(" ", "_")}_{overlap:.2f}.png'
        im_fp.unlink(missing_ok=True)
        sub_stats = stats[list(classes)]
        all_aps = sub_stats.to_numpy()
        # Only use columns where there is no NaN values
        non_nan_aps = all_aps.T[~np.isnan(all_aps.sum(axis=0))].T
        if non_nan_aps.shape[1] == 0:
            err4(f"{name}: No values to compare")
            continue
        mean_aps = non_nan_aps.mean(axis=1).reshape((len(chkpnt_names), n_datasets))
        fig, ax = plt.subplots(figsize=(15, 9))
        X = np.arange(len(chkpnt_names))
        incr = 0.4
        center_offset = (incr * (len(dataset_names) - 1)) / 2
        for i, (ds_name, col, mean_ap) in enumerate(
                zip(dataset_names, itertools.cycle(['b', 'r', 'g', 'y', 'c', 'm']), mean_aps.T)):
            r = ax.bar(X + incr * i - center_offset, mean_ap, color=col, width=incr,
                       label=f'{ds_name} ({stats["samples"][i]} samples)')
            ax.bar_label(r, padding=3)
        ax.set_ylabel('AP')
        ax.set_title(f'AP of {name} by model and training set (overlap = {overlap:.2f})')
        plt.xticks(X, chkpnt_names, rotation=10, horizontalalignment='right', fontsize='small')
        ax.legend()
        fig.tight_layout()
        plt.savefig(im_fp)
        # msg3(f'Saved plot to \033[1m{str(im_fp)}\033[m')
        # plt.show()
        plt.close(fig)


def compile_stats(stats: dict, overlap: np.float, index: list) -> DataFrame:
    overlap_stats = {}
    for cls, overlap_aps in stats.items():
        if cls == 'samples':
            overlap_stats['samples'] = stats['samples']
        else:
            overlap_stats[cls] = []
            for overlap_ap in overlap_aps:
                overlap_stats[cls].append(overlap_ap.get(overlap, np.NaN))
    return DataFrame(overlap_stats, index=index)


def plot_stats(overlaps: np.ndarray, folder: Path, stats: dict, index: list, dataset_names: list):
    for overlap in overlaps:
        msg2(f"Processing stats for overlap = \033[1m{overlap:.2f}\033[m")
        eval_fp = folder / f'eval_{overlap:.2f}.csv'
        stat_df = compile_stats(stats, overlap, index)
        stat_df.to_csv(eval_fp)
        create_plots(stat_df, dataset_names, overlap, folder)


def from_checkpoint(checkpoint_file: Path, cfg: Config) -> Tuple[str, dict, Sequential]:
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    try:
        checkpoint = load_checkpoint(model, str(checkpoint_file), map_location='cpu')
    except RuntimeError as e:
        err2(f"!!! Failed loading checkpoint \033[1m{str(checkpoint_file)}\033[m{ERR}: {e}")
        return

    if 'config' in checkpoint['meta'].keys():
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
        msg3(
            f"Original config: \033[1m{config_file.name}\033[m (epoch: \033[1m{checkpoint['meta']['epoch']}\033[m)")
        checkpoint_id = config_file.name
    else:
        warn3("No original config found")
        checkpoint_id = f"{checkpoint_file.parent.parent.name}_{checkpoint_file.parent.name}"
    return checkpoint_id, checkpoint, model


def preprocess_checkpoints(checkpoint_list: List[str], cfg: Config) -> dict:
    msg1("Preprocessing checkpoints")
    checkpoints = {}
    for checkpoint_file in map(Path, checkpoint_list):
        if not checkpoint_file.exists():
            warn2(f"!!! Checkpoint file \033[1m{str(checkpoint_file)}\033[m{WARN} does not exist")
            continue
        if checkpoint_file.is_file():
            msg2(f"Pre-loading checkpoint \033[1m{str(checkpoint_file)}\033[m")
            checkpoint_id, checkpoint, model = from_checkpoint(checkpoint_file, cfg)

            other_ckpnt_file, other_ckpnt, other_model = checkpoints.get(checkpoint_id, (None, None, None))
            if other_ckpnt is not None:
                # Select the model of the same config with the highest epoch
                if int(checkpoint['meta']['epoch']) >= int(other_ckpnt['meta']['epoch']):
                    warn4(
                        f"Replaces model with same config: epoch \033[1m{other_ckpnt['meta']['epoch']}\033[m >= \033[1m{checkpoint['meta']['epoch']}\033[m")
                else:
                    checkpoint_file, checkpoint, model = other_ckpnt_file, other_ckpnt, other_model
                    warn4(
                        f"Ignored model with same config: epoch\033[1m {other_ckpnt['meta']['epoch']}\033[m <= \033[1m{checkpoint['meta']['epoch']}\033[m")
            checkpoints[checkpoint_id] = checkpoint_file, checkpoint, model
        elif checkpoint_file.is_dir():
            # Multimodel
            msg2(f"Pre-loading multi-models from \033[1m{str(checkpoint_file)}\033[m")
            checkpoint_id = checkpoint_file.name
            checkpoint_data = []
            for multi_checkpoint_file in checkpoint_file.glob('*.pth'):
                checkpoint_sub_id, checkpoint, model = from_checkpoint(multi_checkpoint_file, cfg)
                checkpoint_data.append((checkpoint_sub_id, multi_checkpoint_file, checkpoint, model))
            checkpoints[checkpoint_id] = checkpoint_data
        else:
            raise Exception(f"Checkpoint {str(checkpoint_file)} is neither a file nor a folder")
    return checkpoints


def prepare_folder(checkpoint_id: str, checkpoint, checkpoint_file: Path, out_folder: Path) -> Tuple[str, Path, dict, list]:
    cfg_str, config_file = None, None
    kwargs = {'workers_per_gpu': 4}
    if 'config' in checkpoint['meta'].keys():
        cfg_str = checkpoint['meta']['config']
        chkp_cfg = config_from_str(cfg_str)
        kwargs['workers_per_gpu'] = chkp_cfg.data.workers_per_gpu
        config_file = Path(chkp_cfg.filename)
        assert config_file.suffix == '.py'
        epoch = checkpoint['meta']['epoch']
        time = parse(checkpoint['meta']['time'])
        print('#' * 30)
        msg1(
            f"Loaded checkpoint \033[1m{checkpoint_file.name}\033[m (\033[1m{epoch}\033[m epochs, created: {str(time)})")
        checkpoint_id = f"{config_file.stem}_epoch_{epoch}"
    else:
        msg1(f"Loaded checkpoint \033[1m{checkpoint_file.name}\033[m")
        chkp_cfg = None
        suffix = checkpoint_file.stem.split('_')[-1]
        epoch_str = ''
        if suffix.isdigit():
            epoch = int(suffix)
            warn2(f"Assuming epoch to be {epoch}")
            epoch_str = f"_epoch_{epoch}"
        checkpoint_id = checkpoint_id + epoch_str

    chkp_folder = out_folder / checkpoint_id
    chkp_folder.mkdir(parents=True, exist_ok=True)
    if cfg_str is not None and config_file is not None:
        with open(chkp_folder / config_file.name, 'w') as fp:
            fp.write(f"# {cfg_str}")
        msg2(f"Extracted original configuration to \033[1m{config_file.name}\033[m")

    test_sets = []
    for test_set in TEST_SETS:
        new_test_set = test_set.copy()
        if chkp_cfg is not None:
            new_test_set['type'] = chkp_cfg.data.test.type
        test_sets.append(new_test_set)

    # Link original checkpoint
    new_chkpnt = chkp_folder / checkpoint_file.name
    if new_chkpnt.is_symlink():
        new_chkpnt.unlink()
    new_chkpnt.symlink_to(os.path.relpath(checkpoint_file, new_chkpnt.parent))

    return checkpoint_id, chkp_folder, kwargs, test_sets


def save_proposal_stats(proposals: dict, prop_stat_fp: Path, data_loader: DataLoader):
    prop_stats = {}
    for proposal in proposals['proposals']:
        cat_id = int(proposal['cat_id'])
        cat = data_loader.dataset.CLASSES[cat_id - 1]
        x, y, w, h, a = poly_to_rotated_box_single(list(map(float, proposal['bbox'])))
        a *= 180.0 / math.pi
        prop_stats[cat] = prop_stats.get(cat, []) + [a]
    prop_data = {}
    for i, cat in enumerate(data_loader.dataset.CLASSES):
        angles = prop_stats.get(cat, [])
        prop_data[cat] = (len(angles), np.mean(angles), np.std(angles))
        # print(f'[{i + 1}] {cat} ({len(angles)}): avg:{np.mean(angles):.02f}, std: {np.std(angles):.02f}')
    csv_data = pandas.DataFrame(prop_data, index=('occurrences', 'avg', 'std')).transpose()
    csv_data.to_csv(prop_stat_fp)


def get_proposals(checkpoint: dict, cfg: Config, model: Sequential, data_loader: DataLoader, proposals_fp: Path, args: Namespace) -> list:
    # old versions did not save class info in checkpoints, this workaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = data_loader.dataset.CLASSES

    if not proposals_fp.exists() or not args.cache:
        proposals_fp.parent.mkdir(exist_ok=True)
        msg3(f"Testing model on \033[1m{Path(data_loader.dataset.ann_file).stem}\033[m")
        if isinstance(model, MockModel):
            output = model.get_gt(data_loader.dataset)
        else:
            distributed = args.launcher != 'none'
            if not distributed:
                model = MMDataParallel(model, device_ids=[0])
                output = single_gpu_test(model, data_loader, args.show, cfg, args.postprocess, args.round)
            else:
                model = MMDistributedDataParallel(model.cuda())
                output = multi_gpu_test(model, data_loader, args.tmpdir, cfg, args.postprocess, args.round)
            print()  # The tests use ncurses and don't append a new line at the end

        msg3(f'Writing proposals to \033[1m{str(proposals_fp)}\033[m')
        mmcv.dump(output, proposals_fp)
        #save_predictions(output, proposals_fp.with_name(args.json_out), data_loader, args)
    else:
        msg3(f'Reading proposals from \033[1m{str(proposals_fp)}\033[m')
        output = mmcv.load(proposals_fp)
    return output


def save_predictions(predictions, result_file: Path, data_loader: DataLoader, args):
    # Save predictions in the COCO json format
    rank, _ = get_dist_info()
    if args.json_out and rank == 0:
        #if not result_file.with_suffix('.bbox.json').exists() or not args.cache:
        msg3(f"Saving predictions to \033[1m{str(result_file.with_suffix('.*'))}\033[m")
        if not isinstance(predictions[0], dict):
            result_file = result_file.with_suffix('')
            if not result_file.with_suffix('.bbox.json').exists() or not args.cache:
                results2json(data_loader.dataset, predictions, result_file)
            result_file = result_file.with_suffix('.bbox.json')  # function sets custom suffix
        else:
            for name in predictions[0]:
                result_file = result_file.with_suffix(f'.{name}{result_file.suffix}')
                outputs_ = [out[name] for out in predictions]
                if not result_file.exists() or not args.cache:
                    results2json(data_loader.dataset, outputs_, result_file)
        standard_results_fp = result_file.with_name(args.json_out)
        if not standard_results_fp.exists() or not args.cache:
            data_loader.dataset.write_results_json(outputs_rotated_box_to_poly_np(predictions), filename=str(standard_results_fp))


def evaluate_results(outputs: list, result_folder: Path, data_loader: DataLoader, checkpoint_id: str, overlaps: np.ndarray, stats: dict, args: Namespace):
    overlaps_str = str(overlaps).replace("\n", "")
    result_folder = result_folder / Path(data_loader.dataset.ann_file).stem
    result_file = None
    if args.json_out is not None:
        result_file = result_folder / args.json_out
    metrics_fp = result_folder / "dsv2_metrics.pkl"
    needs_to_be_exported = False
    if not metrics_fp.exists() or not args.cache:
        msg3(
            f"Evaluating: \033[1m{str(checkpoint_id)}\033[m on \033[1m{data_loader.dataset.ann_file}\033[m in \033[1m{str(result_folder)}\033[m")
        data_loader.dataset.evaluate(
            ensure_8_tuple(outputs),
            result_json_filename=str(result_file) if result_file is not None else None,
            work_dir=str(result_folder),
            iou_thrs=overlaps
        )  # Extremely slow...
        needs_to_be_exported = True
    msg3(f'Reading calculated metrics from \033[1m{str(metrics_fp)}\033[m')
    metrics = mmcv.load(metrics_fp)

    prop_stat_fp = result_folder / 'proposal_stats.csv'
    if (result_file is not None and not prop_stat_fp.exists()) or needs_to_be_exported:
        msg3(f"Calculating statistics for results -> {prop_stat_fp.name}")
        with open(result_file, 'r') as fp:
            save_proposal_stats(json.load(fp), prop_stat_fp, data_loader)

    msg3(f'Compiling metrics with overlaps \033[1m{overlaps_str}\033[m')
    for cls, overlap_metrics in metrics.items():
        for overlap in overlaps:
            overlap_metrics[overlap] = overlap_metrics[overlap].get('ap', np.NaN)
        metrics[cls] = overlap_metrics
    for cls in data_loader.dataset.CLASSES:
        stats[cls].append(metrics.get(cls, {}))
    # for cls, metrics in stats.items():
    #     stats[cls].append(metrics[-1])
    save_predictions(outputs, result_folder / args.json_out, data_loader, args)


def infer_checkpoint(checkpoint: dict, main_config: Config, model: Sequential, data_loader: DataLoader, folder: Path,
                     args: Namespace) -> Tuple[list, Path]:
    ann_file = Path(data_loader.dataset.ann_file)
    msg2(f"Running checkpoint on dataset: \033[1m{ann_file.stem}\033[m")
    result_folder = folder / ann_file.stem
    result_folder.mkdir(exist_ok=True)
    eval_fp = result_folder / args.out
    outputs = get_proposals(checkpoint, main_config, model, data_loader, eval_fp, args)
    return outputs, eval_fp

def wbf_proposals_to_output(proposals: pd.DataFrame) -> List[List[np.ndarray]]:
    # output = []
    # last_img_idx = 0
    # def new_sample():
    #     sample_output = []
    #     for _ in range(135):
    #         sample_output.append(np.zeros((0, 6)))
    #     output.append(sample_output)
    #     return sample_output
    # sample_output = new_sample()
    # for bbox, cat_id, img_idx, score in proposals.values:
    #     if img_idx != last_img_idx:
    #         sample_output = new_sample()
    #         last_img_idx = img_idx
    #     bboxes: np.ndarray = sample_output[cat_id - 1]
    #     #bbox = rotated_box_to_poly_np(bbox[:5])
    #     bbox = np.concatenate([bbox, np.array([score])])
    #     sample_output[cat_id - 1] = np.concatenate([bboxes.reshape((-1, 9)), bbox.reshape((1, 9))])
    # return output

    # TODO: @embe: not sure, I still have to check this one -> I just copied it from Urs Code because something didn't work. Not sure if this is still required though

    proposals_WBF_per_img = []
    for img_idx in sorted(set(list(proposals.img_idx))):
        props = proposals[proposals.img_idx == img_idx]
        result_prop = [np.empty((0, 9))] * 136

        for cat_id in sorted(set(list(props.cat_id))):
            props_cat = props[props.cat_id == cat_id]
            result_prop[cat_id - 1] = np.concatenate((np.array(list(props_cat.bbox)),
                                                      np.array(list(props_cat.score)).reshape(
                                                          (np.array(list(props_cat.score)).shape[0], 1))), axis=1)

        proposals_WBF_per_img.append(result_prop)

    return proposals_WBF_per_img

def ensure_8_tuple(outputs: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
    for sample in outputs:
        for i, cls in enumerate(sample):
            if cls.shape[0] == 0:
                cls = cls.reshape((0, 9))
            elif cls.shape[1] == 6:
                cls = np.concatenate([rotated_box_to_poly_np(cls[:,:5]), cls[:,5].reshape(-1, 1)], axis=1)
            sample[i] = cls
    return outputs


class MockModel(Sequential):
    def __init__(self, acc: float):
        super(MockModel, self).__init__()
        self.acc = min(max(acc, 0.0), 1.0)

    def get_gt(self, dataset: DeepScoresV2Dataset) -> List[List[np.ndarray]]:
        msg3(f"Constructing {self.acc*100:.1f}% of ground truth as outputs")
        output = []
        prog_bar = mmcv.ProgressBar(len(dataset))
        for meta, gt in dataset.obb:
            num_selected = int(len(gt) * self.acc)
            new_gt = gt[:num_selected].copy()
            new_gt = new_gt.assign(cat_id=new_gt['cat_id'].map(lambda x: x[0]))
            sample = []
            for cls in dataset.CLASSES[1:]:  # Since the model swallows the braces, and we add blank proposals for those, we have to omit braces here as well.
                cls_id = class_names.index(cls) + 1  # Class ids start at 1
                bboxes = np.array(new_gt[new_gt['cat_id'] == cls_id]['o_bbox'].tolist()).reshape((-1, 8))
                sample.append(bboxes)
            output.append(sample)
            prog_bar.update()
        print()  # Progress bar doesn't break line
        return output

    def __call__(self, *args, **kwargs):
        print(args)
        for k, v in kwargs.items():
            print(k, v)
        return super(MockModel, self).__call__(*args, **kwargs)


def create_mockup_model(acc: float) -> Tuple[Path, dict, Sequential]:
    return Path(f"model_{acc:.2f}.pth"), {
        'meta': {}
    }, MockModel(acc)


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
    tmp_test_sets = get_test_sets(*TEST_SETS)
    stats.update({key: [] for key in tmp_test_sets[0].dataset.CLASSES})
    dataset_names = []
    for tmp_test_set in tmp_test_sets:
        dataset_names.append(tmp_test_set.dataset.obb.dataset_info['description'])

    # Make sure we only use the best epochs for each config
    checkpoints = preprocess_checkpoints(args.checkpoints, cfg)
    checkpoints = {
        'mockup_50': create_mockup_model(0.5),
        'mockup_100': create_mockup_model(1.0),
        **checkpoints,
    }

    overlaps = np.arange(0.1, 0.96, 0.05)
    outputs_m = defaultdict(dict)
    metrics = {}
    for checkpoint_id, checkpoint_data in checkpoints.items():
        if isinstance(checkpoint_data, tuple):
            checkpoint_file, checkpoint, model = checkpoint_data
            outputs = {}
            sub_stats = defaultdict(list)
            checkpoint_id, chkp_folder, test_set_kwargs, test_sets = prepare_folder(checkpoint_id, checkpoint, checkpoint_file, out_folder)
            test_set_kwargs['distributed'] = distributed
            for data_loader in get_test_sets(*test_sets, **test_set_kwargs):
                ann_file = Path(data_loader.dataset.ann_file)
                index.append(f"{checkpoint_id} - {ann_file.stem}")
                output, _ = infer_checkpoint(checkpoint, cfg, model, data_loader, chkp_folder, args)
                for page in output:
                    page.insert(0, np.array([]))

                outputs[data_loader] = output
                sub_stats['samples'].append(len(data_loader.dataset.img_ids))
                # Evaluate results
                evaluate_results(output, chkp_folder, data_loader, checkpoint_id, overlaps, sub_stats, args)
                outputs_m[checkpoint_id][data_loader] = output
            for cls, aps in sub_stats.items():
                stats[cls].extend(aps)
            compile_stats(sub_stats, 0.5, index[-len(test_sets):]).to_csv(chkp_folder / "stats.csv")
        elif isinstance(checkpoint_data, list):
            mm_index = []
            mm_stats = {'samples': []}
            sub_stats = defaultdict(list)
            ensemble_folder = out_folder / checkpoint_id
            ensemble_folder.mkdir(parents=True, exist_ok=True)
            test_set_kwargs = {
                'workers_per_gpu': 4,
                'distributed': distributed,
            }
            for data_loader in get_test_sets(*TEST_SETS, **test_set_kwargs):
                ann_file = Path(data_loader.dataset.ann_file)
                wbf_glob_dir = ensemble_folder / ann_file.stem
                wbf_glob_dir.mkdir(exist_ok=True)
                for checkpoint_sub_id, checkpoint_file, checkpoint, model in checkpoint_data:
                    checkpoint_sub_id, chkp_folder, _, test_sets = prepare_folder(checkpoint_sub_id, checkpoint, checkpoint_file, ensemble_folder)
                    idx = f"{checkpoint_sub_id} - {ann_file.stem}"
                    output, eval_fp = infer_checkpoint(checkpoint, cfg, model, data_loader, chkp_folder, args)

                    # When using the WBF load_proposals function, copy the results.json file into a special folder for
                    # recursively finding the results.json files
                    folder = wbf_glob_dir / checkpoint_sub_id
                    folder.mkdir(exist_ok=True)
                    shutil.copyfile(chkp_folder / ann_file.stem / args.json_out, folder / 'result.json')
                sub_stats['samples'].append(len(data_loader.dataset.img_ids))
                mm_index.append(f"{checkpoint_id} - {ann_file.stem}")
                index.append(f"{checkpoint_id} - {ann_file.stem}")
                proposal_fp = wbf_glob_dir / f"wbf_proposals.pkl"
                if not (proposal_fp.exists() and args.cache):
                    msg2("Running weighted box fusion")
                    wbf_proposals: DataFrame = load_proposals(Namespace(inp=wbf_glob_dir), data_loader.dataset, iou_thr=0.3)
                    wbf_proposals.to_pickle(proposal_fp)
                    # Delete so the evaluation runs through
                    (wbf_glob_dir / "dsv2_metrics.pkl").unlink(missing_ok=True)
                    shutil.rmtree(wbf_glob_dir / "visualized_proposals", ignore_errors=True)
                else:
                    msg2("Loading weighted box fusion results")
                    wbf_proposals: DataFrame = pd.read_pickle(proposal_fp)
                output2 = wbf_proposals_to_output(wbf_proposals)
                for page in output2:
                    page.insert(0, np.array([]))
                outputs_m[checkpoint_id][data_loader] = output2
                evaluate_results(outputs_m[checkpoint_id][data_loader], ensemble_folder, data_loader, checkpoint_id, overlaps, sub_stats, args)
            for cls, aps in sub_stats.items():
                stats[cls].extend(aps)
            compile_stats(sub_stats, 0.5, index[-len(TEST_SETS):]).to_csv(ensemble_folder / "stats.csv")
    print('#' * 30)
    print('#' * 30)
    print('#' * 30)
    msg1(f"Evaluating stats")
    plot_stats(overlaps, out_folder, stats, index, dataset_names)


if __name__ == '__main__':
    main()
