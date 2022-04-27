import argparse
import os
import pickle
from collections import OrderedDict

import numpy as np
import pandas as pa


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze errors')
    parser.add_argument(
        '--ev_folder',
        type=str,
        default="work_dirs/s2anet_r50_fpn_1x_deepscoresv2_sage_halfrez_crop/",
        help="Path to the folder to evaluate")
    parser.add_argument(
        '--filename',
        type=str,
        default="dsv2_metrics.pkl",
        help="Name of the file(s) to evaluate (must be inside the folder defined by --ev_folder)")
    parser.add_argument('--create_overview', action='store_true',
                        help='Create one csv containing the results from all files')
    return parser.parse_args()


def get_pickles(evaluations_folder, filename):
    error_metrics = dict()
    for base_i, folders_i, files_i in os.walk(evaluations_folder):
        pickles = [x for x in files_i if filename in x]
        if len(pickles) == 1:
            f = open(os.path.join(base_i, pickles[0]), "rb")
            metrics = pickle.load(f)
            f.close()
            error_metrics[base_i] = metrics
        elif len(pickles) > 1:
            print("multiple pickles found")
    return error_metrics


def create_dframe(error_metrics):
    dframes = dict()
    for name, values in error_metrics.items():
        row_names = list(values.keys())
        column_names = list(values[row_names[0]].keys())

        metrics_df = pa.DataFrame(np.zeros((len(row_names), len(column_names))))
        metrics_df.index = row_names
        metrics_df.columns = column_names

        for symbol, metrics in values.items():
            for overlap, ap in metrics.items():
                if isinstance(ap, dict):
                    metrics_df[overlap][symbol] = ap['ap']
                else:
                    metrics_df[overlap][symbol] = ap

        dframes[name] = metrics_df

    return dframes


def add_averages(dframes):
    for key, dframe in dframes.items():
        overall_mean = dframe.mean()
        try:
            variable_mean = dframe.loc[['slur', 'beam', 'tie', 'dynamicCrescendoHairpin', 'dynamicDiminuendoHairpin'],
                            :].mean()
        except:
            variable_mean = overall_mean * 0
        dframe = dframe.append([overall_mean, variable_mean])
        dframe = dframe.rename(index={0: "overall_mean", 1: "variable_mean"})
        dframes[key] = dframe

    return dframes


def store_csv(dframes):
    for key, dframe in dframes.items():
        path = os.path.join(key + "_metrics.csv")
        print(path)
        dframe.to_csv(path)
    return None


def merge_dataframes(dframes):
    threshold = 0.5
    columns = []

    for name, df in OrderedDict(sorted(dframes.items())).items():
        col = df.loc[:, [threshold]]
        col.rename(columns={threshold: name + f" (th={threshold})"}, inplace=True)
        columns.append(col)

    overview_df = pa.concat(columns, axis=1, join='outer')
    return overview_df


def main():
    args = parse_args()
    evaluations_folder = args.ev_folder
    error_metrics = get_pickles(evaluations_folder, args.filename)
    dframes = create_dframe(error_metrics)

    # add averages
    dframes = add_averages(dframes)

    # store as csv
    store_csv(dframes)

    # create overview: store all dataframes in single file
    if args.create_overview:
        overview = merge_dataframes(dframes)
        store_csv({f'{args.ev_folder}/overview': overview})


if __name__ == '__main__':
    main()
