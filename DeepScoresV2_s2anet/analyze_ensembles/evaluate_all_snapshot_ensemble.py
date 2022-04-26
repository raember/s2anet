import argparse
import copy
from argparse import Namespace
from pathlib import Path
import re
from DeepScoresV2_s2anet import analyze_errors
from DeepScoresV2_s2anet.analyze_ensembles import snapshot_overlap, draw_WBF_for_multi_model, \
    compare_dsv2_metrics_multi_model
from tools import test_multi_model


parser = argparse.ArgumentParser(description='Evaluate Snapshot Ensemble')
parser.add_argument('config', help='test config file path')
parser.add_argument('--checkpoints', nargs='+',
                    help='checkpoint files (use like --checkpoints file1 file2 file3 ...', required=True)

parser.add_argument(
    '--out',
    type=str,
    default="eval",
    help="Pth to the output folder")
parser.add_argument(
    '--data',
    choices=['coco', 'dota', 'dota_large', 'dota_hbb', 'hrsc2016', 'voc', 'dota_1024', 'dsv2'],
    default='dsv2',
    type=str,
    help='eval dataset type')
args = parser.parse_args()

BASE_PATH = Path(args.out)

def run_test_multi_model():
    def get_args():
        args_ = copy.deepcopy(args)
        args_.data = 'dsv2'
        args_.out = str(BASE_PATH / 'multi_model_test.pkl')
        args_.json_out = None
        args_.launcher = 'none'
        args_.show = False
        args_.eval = []
        args_.test_sets = None
        args_.cache = None
        return args_

    test_multi_model.parse_args = get_args
    test_multi_model.main()


def _get_result_jsons():
    def sort_str_with_int(l):
        convert = lambda text: float(text) if text.isdigit() else text
        alphanum = lambda key: [convert(c) for c in re.split('([-+]?[0-9]*\.?[0-9]*)', key)]
        l.sort(key=alphanum)
        return l

    return sort_str_with_int([str(x) for x in list(BASE_PATH.rglob("result.json"))])

def run_snapshot_overlap():
    result_jsons = _get_result_jsons()

    def get_args():
        args_dict = {
            'config': args.config,
            'jsons_gt': result_jsons,
            'jsons_pr': result_jsons,
            'out': str(BASE_PATH / "overlap")
        }
        return Namespace(**args_dict)

    snapshot_overlap.parse_args = get_args
    snapshot_overlap.main()


def run_snapshot_overlap_reduced():
    # only compare every 5th model (method not feasible for too many snapshots)
    result_jsons = _get_result_jsons()[0::5]

    def get_args():
        args_dict = {
            'config': args.config,
            'jsons_gt': result_jsons,
            'jsons_pr': result_jsons,
            'out': str(BASE_PATH / "overlap")
        }
        return Namespace(**args_dict)

    snapshot_overlap.parse_args = get_args
    snapshot_overlap.main()



def run_WBF():
    def get_args():
        args_ = copy.deepcopy(args)
        args_.inp = str(BASE_PATH)
        args_.out = str(BASE_PATH / "wbf")
        args_.iou_thr = 0.1
        args_.vis_thr = 0.0001
        args_.plot_proposals = False
        args_.s_cache = None
        args_.l_cache = None
        return args_

    draw_WBF_for_multi_model.parse_args = get_args
    draw_WBF_for_multi_model.main()


def run_compare_metrics():
    def get_args():
        args_ = copy.deepcopy(args)
        args_.inp = str(BASE_PATH)
        args_.out = str(BASE_PATH / "compare_metrics")
        args_.wbf = str(BASE_PATH / "wbf" / "deepscores_ensemble_metrics.pkl")
        return args_

    compare_dsv2_metrics_multi_model.parse_args = get_args
    compare_dsv2_metrics_multi_model.main()


def run_analyze_errors():
    def get_args():
        args_ = copy.deepcopy(args)
        args_.ev_folder = str(BASE_PATH)
        args_.filename = "dsv2_metrics.pkl"
        args_.create_overview = True
        return args_

    analyze_errors.parse_args = get_args
    analyze_errors.main()


if __name__ == '__main__':
    if BASE_PATH.exists() and any(BASE_PATH.iterdir()):
        r = input("'eval' folder is not empy - continue? [y/N]")
        if str(r) != "y" or str(r) != "yes":
            exit()

    run_test_multi_model()
    if len(_get_result_jsons()) <= 20:
        run_snapshot_overlap()
    else:
        run_snapshot_overlap_reduced()
    run_WBF()
    run_compare_metrics()
    run_analyze_errors()

