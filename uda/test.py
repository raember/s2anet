import pickle
import argparse
import os
import os.path as osp
import shutil
import tempfile
import wandb
import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, load_checkpoint

from mmdet.apis import init_dist
from mmdet.core import coco_eval, results2json, wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.core import rotated_box_to_poly_np

from results import get_pickles, create_dframe, add_averages, store_csv, strip_prefix_if_present

def single_gpu_test(model, data_loader, show=False, cfg = None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
            # result, bbox_list = model(return_loss=False, rescale=not show, **data)
        results.append(result)
        if show:
            print("asdf")
            #for nr, sub_list in enumerate(bbox_list):
            #    bbox_list[nr] = [rotated_box_to_poly_np(sub_list[0].cpu().numpy()), sub_list[1].cpu().numpy()]

            model.module.show_result(data, result, show=show, dataset=dataset.CLASSES,
                                     bbox_transorm=rotated_box_to_poly_np, score_thr=cfg.test_cfg['score_thr'])

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
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
    parser.add_argument('--config', help='test config file path', default = 'configs/deepscoresv2/s2anet_r50_fpn_1x_deepscoresv2_ghos_halfrez_crop.py')
    # parser.add_argument('--checkpoint', help='checkpoint file', default = 'models/uda/model_020.pth')
    # parser.add_argument('--checkpoint', help='checkpoint file', default = 'models/deepscoresV2_tugg_halfrez_crop_epoch250.pth')
    parser.add_argument('--checkpoint', help='checkpoint file', default = 'models/uda_test_model/model_000.pth')
    parser.add_argument('--out', help='output result file', default = 'results/test_imslp/s2anet_no_aug/test_imslp.pkl')
    parser.add_argument('--eval_folder', help='Evaluation folder', default = 'results/test_imslp/s2anet_no_aug')
    parser.add_argument('--model_type', help='Type of model (Basic s2anet or UDA)', default = 's2anet')
    parser.add_argument(
        '--json_out',
        help='output result file name without extension',
        default='models/output',
        type=str)
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        default='bbox',
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--tmpdir', default='results/test/', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    # add dataset type for more dataset eval other than coco
    parser.add_argument(
        '--data',
        choices=['coco', 'dota', 'dota_large', 'dota_hbb', 'hrsc2016', 'voc', 'dota_1024','dsv2'],
        default='dsv2',
        type=str,
        help='eval dataset type')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    wandb.init(project="uda", entity="adhirajghosh")
    args = parse_args()
    args.show = False
    assert args.out or args.show or args.json_out, \
        ('Please specify at least one operation (save or show the results) '
         'with the argument "--out" or "--show" or "--json_out"')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    if args.json_out is not None and args.json_out.endswith('.json'):
        args.json_out = args.json_out[:-5]
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

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)
    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = checkpoint['meta']['CLASSES']
    # print(checkpoint)
    # if args.model_type == 's2anet':
    #
    #     checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    #     model.CLASSES = checkpoint['meta']['CLASSES']
    # elif args.model_type == 'uda':
    #     checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    #     print(checkpoint.keys())
    #     model_weights = strip_prefix_if_present(checkpoint['detector'], 'module.')
    #     model.load_state_dict(model_weights)
    #     model.CLASSES = dataset.CLASSES

    model = MMDataParallel(model, device_ids=[0])
    outputs = single_gpu_test(model, data_loader, args.show, cfg)
    # print(outputs)
    rank, _ = get_dist_info()
    if args.out and rank == 0:
        print('\nwriting results to {}'.format(args.out))
        mmcv.dump(outputs, args.out)
        eval_types = args.eval
        data_name = args.data
        if data_name == 'coco':
            if eval_types:
                print('Starting evaluate {}'.format(' and '.join(eval_types)))
                if eval_types == ['proposal_fast']:
                    result_file = args.out
                    coco_eval(result_file, eval_types, dataset.coco)
                else:
                    if not isinstance(outputs[0], dict):
                        result_files = results2json(dataset, outputs, args.out)
                        coco_eval(result_files, eval_types, dataset.coco)
                    else:
                        for name in outputs[0]:
                            print('\nEvaluating {}'.format(name))
                            outputs_ = [out[name] for out in outputs]
                            result_file = args.out + '.{}'.format(name)
                            result_files = results2json(dataset, outputs_,
                                                        result_file)
                            coco_eval(result_files, eval_types, dataset.coco)

        elif data_name in ['dota', 'hrsc2016']:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            work_dir = osp.dirname(args.out)
            dataset.evaluate(outputs, work_dir, **eval_kwargs)
        elif data_name in ['dsv2']:
            from mmdet.core import outputs_rotated_box_to_poly_np
            # TODO: fix ugly hack to make the labels match
            import numpy as np
            for page in outputs:
                page.insert(0, np.array([]))

            outputs = outputs_rotated_box_to_poly_np(outputs)
            work_dir = osp.dirname(args.out)
            print("Printing work dir",work_dir)
            metrics = dataset.evaluate(outputs, work_dir=None, iou_thrs=[0.3])


    # Save predictions in the COCO json format
    if args.json_out and rank == 0:
        if not isinstance(outputs[0], dict):
            results2json(dataset, outputs, args.json_out)
        else:
            for name in outputs[0]:
                outputs_ = [out[name] for out in outputs]
                result_file = args.json_out + '.{}'.format(name)
                results2json(dataset, outputs_, result_file)

    # output = open(os.path.join(args.eval_folder, 'ds2_metrics.pkl'), 'wb')
    output = open(args.out, 'wb')
    pickle.dump(metrics, output)
    output.close()
    error_metrics = get_pickles(args.eval_folder, args.out.split('/')[-1])
    dframes = create_dframe(error_metrics)

    # add averages
    dframes = add_averages(dframes)

    # store as csv
    store_csv(dframes, args.eval_folder)




if __name__ == '__main__':
    main()
