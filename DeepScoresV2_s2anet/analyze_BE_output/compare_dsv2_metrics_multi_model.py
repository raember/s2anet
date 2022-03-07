import os
import pickle

import numpy as np
import pandas as pa


# Includes code snippets adapted from analyze_errors.py

def get_pickle(evaluations_folder):
    error_metrics = dict()
    for base_i, _, files_i in os.walk(evaluations_folder):
        if 'dsv2_metrics.pkl' not in files_i:
            continue
        pickles = [x for x in files_i if f"dsv2_metrics.pkl" in x]
        if len(pickles) == 1:
            f = open(os.path.join(base_i, pickles[0]), "rb")
            metrics = pickle.load(f)
            f.close()
            name = base_i.split("/")[-1]
            error_metrics[name] = metrics
        elif len(pickles) > 1:
            print("multiple pickles found")
        error_metrics = error_metrics[list(error_metrics.keys())[0]]
        break
    return error_metrics


def get_np_arrays(evaluations_folder):
    # Load pickles
    error_metrics = []
    for f in evaluations_folder:
        error_metrics_i = get_pickle(f)
        error_metrics.append(error_metrics_i)
    all_classes = set().union(*(d.keys() for d in error_metrics))

    # Create dummy dict for missing classes
    dummy_dict = {0.5: {'ap': 0.0, 'precision': 0.0, 'recall': 0.0},
                  0.55: {'ap': 0.0, 'precision': 0.0, 'recall': 0.0},
                  0.6000000000000001: {'ap': 0.0, 'precision': 0.0,
                                       'recall': 0.0},
                  0.6500000000000001: {'ap': 0.0, 'precision': 0.0,
                                       'recall': 0.0},
                  0.7000000000000002: {'ap': 0.0, 'precision': 0.0,
                                       'recall': 0.0},
                  0.7500000000000002: {'ap': 0.0, 'precision': 0.0,
                                       'recall': 0.0},
                  0.8000000000000003: {'ap': 0.0, 'precision': 0.0,
                                       'recall': 0.0},
                  0.8500000000000003: {'ap': 0.0, 'precision': 0.0,
                                       'recall': 0.0},
                  0.9000000000000004: {'ap': 0.0, 'precision': 0.0,
                                       'recall': 0.0},
                  0.9500000000000004: {'ap': 0.0, 'precision': 0.0,
                                       'recall': 0.0},
                  'no_occurences': 0
                  }

    # Get np arrays for each ensemble member
    np_arrays = []
    row_names = all_classes
    column_names = list(dummy_dict.keys())

    for i in range(len(error_metrics)):
        metrics_df = pa.DataFrame(np.zeros((len(row_names), len(column_names))))
        metrics_df.index = row_names
        metrics_df.columns = column_names
    
        for symbol in all_classes:
            
            if symbol in error_metrics[i].keys():
                metrics = error_metrics[i][symbol]
                for overlap, ap in metrics.items():
                    if isinstance(ap, dict):
                        metrics_df[overlap][symbol] = ap['ap']

                    else:
                        metrics_df[overlap][symbol] = ap

            else:
                metrics = dummy_dict

                for overlap, ap in metrics.items():
                    if isinstance(ap, dict):
                        metrics_df[overlap][symbol] = ap['ap']
                    else:
                        metrics_df[overlap][symbol] = ap

        np_arrays.append(metrics_df.to_numpy())

    # Stack all member outputs
    np_arrays = np.dstack(np_arrays)

    iou_thr = list(dummy_dict.keys())
    classes = list(all_classes)
    metrics_d = {}

    for cls in range(np_arrays.shape[0]):
        out_d = {}

        for thr in range(np_arrays.shape[1] - 1):
            q1 = np.percentile(np_arrays[cls, thr, :], 25, interpolation='midpoint')
            q3 = np.percentile(np_arrays[cls, thr, :], 75, interpolation='midpoint')
            mini = np_arrays[cls, thr, :].min()
            maxi = np_arrays[cls, thr, :].max()
            av = np_arrays[cls, thr, :].mean()
            out = {'mean': av, '0.25': q1, '0.75': q3, 'min': mini, 'max': maxi}
            out_d[iou_thr[thr]] = out

        out_d['nr_occurences'] = max(np_arrays[cls, thr + 1, :])  # max() because
        # if class was not found at least 1 time nr_occurrences is set to 0.
        metrics_d[classes[cls]] = out_d

    # Generate output for iou_thr = 0.5, only:
    iou_thr = iou_thr[0]
    column_names = list(metrics_d[classes[0]][iou_thr].keys())
    column_names.append('nr_occurrences')
    metrics_df = pa.DataFrame(
        np.zeros((len(row_names), len(column_names)))
    )
    metrics_df.index = row_names
    metrics_df.columns = column_names

    for cls in list(metrics_d.keys()):
        tmp_d = metrics_d[cls][iou_thr]
        tmp_d['nr_occurrences'] = metrics_d[cls]['nr_occurences']
        metrics_df.loc[cls] = tmp_d

    return metrics_df


def main():
    input_folder_p = '/s2anet/work_dirs/s2anet_r50_fpn_1x_deepscoresv2_sage_halfrez_crop/'
    input_folder = sorted([input_folder_p + x for x in os.listdir(input_folder_p) if "result_" in x])
    output_folder = '/s2anet/work_dirs/s2anet_r50_fpn_1x_deepscoresv2_sage_halfrez_crop/analyze_BE_output/'
    metrics_df = get_np_arrays(input_folder)
    metrics_df = round(metrics_df, 2)
    metrics_df.astype({'nr_occurrences': np.float})
    metrics_df = metrics_df.sort_values('nr_occurrences', ascending=False)
    metrics_df.rename(columns={'nr_occurrences': 'nr_occur'}, inplace=True)
    spread = metrics_df.drop(
        metrics_df[(metrics_df['min'] == 0) & (metrics_df['max'] == 0)].index)
    x = spread['max'] - spread['min']
    x = np.mean(x)
    print(f"The mean range between min and max class-wise AP is: {x}")
    path = os.path.join(output_folder, "IOU_0-5_class_wise_APs.csv")
    metrics_df.to_csv(path)


if __name__ == '__main__':
    main()
