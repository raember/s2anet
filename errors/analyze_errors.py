import os
import pickle
import pandas as pa
import numpy as np

def get_pickles(evaluations_folder):
    error_metrics = dict()
    for base_i, folders_i, files_i in os.walk(evaluations_folder):
        pickles = [x for x in files_i if "imslp_metrics.pkl" in x]
        if len(pickles) == 1:
            f = open(os.path.join(base_i, pickles[0]), "rb")
            metrics = pickle.load(f)
            f.close()
            name = base_i.split("/")[-1]
            error_metrics[name] = metrics
        elif len(pickles) > 1:
            print("multiple pickles found")
    return error_metrics



def create_dframe(error_metrics):

    dframes = dict()
    for name, values in error_metrics.items():
        row_names = list(values.keys())
        column_names = list(values[row_names[0]].keys())
        print(column_names)

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
            variable_mean = dframe.loc[['slur', 'beam', 'tie','dynamicCrescendoHairpin', 'dynamicDiminuendoHairpin'],:].mean()
        except:
            variable_mean = overall_mean * 0
        dframe = dframe.append([overall_mean,variable_mean])
        dframe = dframe.rename(index={0: "overall_mean", 1: "variable_mean"})
        dframes[key] = dframe

    return dframes

def store_csv(dframes, evaluations_folder):
    for key, dframe in dframes.items():
        path = os.path.join(evaluations_folder, key, key+"_metrics.csv")
        dframe.to_csv(path)
    return None


def main():

    evaluations_folder = "models/test_ds2/"
    error_metrics = get_pickles(evaluations_folder)
    dframes = create_dframe(error_metrics)

    # add averages
    dframes = add_averages(dframes)

    # store as csv
    store_csv(dframes, evaluations_folder)


if __name__ == '__main__':
    main()