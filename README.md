# Real World Music Object Recognition (under construction)

The *RealScores* test dataset can be found in `RealScores.zip` in this repository.
All necessary scripts are available in the `dev` branch, unless stated otherwise.
All scripts have to be executed from the project directory.

### ScoreAug

For more details, please have a look at branch [`dev-embe`](../../tree/dev-embe/).
This method requires the `blanks.tar` tarball to be extracted into the deepscores folder (`data/deep_scores_dense`).

#### Train the model

The model can be trained using the following command:

```bash
# Don't forget to adjust WandB settings
python tools/train.py configs/deepscoresv2/s2anet_r50_fpn_1x_deepscoresv2_tugg_halfrez_crop.py
```

### UDA

### Confidence Ratings

For more details, please have a look at branch [`dev-sage`](../../tree/dev-sage/).

##### Train the Model

The model can be trained using the following command:

```bash
python tools/train.py configs/deepscoresv2/s2anet_r50_fpn_1x_deepscoresv2_sage_halfrez_crop.py
```

##### Test the Model

```bash
python DeepScoresV2_s2anet/analyze_ensembles/evaluate_all_snapshot_ensemble.py
  configs/deepscoresv2/s2anet_r50_fpn_1x_deepscoresv2_sage_halfrez_crop.py
--checkpoints
  <...>
```

### Test all model checkpoints

Testing on our Realscores test set requires the `imslp_test.tar` tarball to be extracted into the deepscores folder (`data/deep_scores_dense`).
Our results presented in our paper can be obtained by training all the variations of our methods and then running our `test_multi_model.py` test script (on the `dev` branch) on them:

```bash
python tools/test_multi_model.py \
    configs/deepscoresv2/s2anet_r50_fpn_1x_deepscoresv2_tugg_halfrez_crop.py \
    --checkpoints \
        models/dsv2/DS_2022_03_23/deepscoresV2_tugg_halfrez_crop_epoch250.pth \
        models/dsv2/DS_2022_04_26/epoch_2000.pth \
        models/dsv2_no_augment/DS_2022_04_26/epoch_2000.pth \
        models/dsv2_finalize_snp/DS_2022_05_27/epoch_250.pth \
        models/dsv2_finalize/DS_2022_05_27/epoch_200.pth \
        models/dsv2hybrid/DS_2022_05_27/epoch_600.pth \
        models/dsv2hybrid_finalize_snp/DS_2022_05_27/epoch_150.pth \
    --eval bbox \
    --json_out result.json \
    --data dsv2 \
    -c
```

The script will cache intermediate results (`-c`), so if there is an error, restarting the script will not needlessly retest checkpoints.

### Other useful scripts

[`tools/deepscores_stats.py`](../../tree/dev/tools/deepscores_stats.py) - This script is used to compile statistics about the dataset (`-c`), plot the statistics (`-p`), find outliers (`-f`/`-cf`), and fix them (`-a`/`-ag`).
To help fix outliers, the script will output javascript code to be entered into the web console of a [GeoGebra session](https://mat.geogebra.org/classic).
To create the necessary GeoGebra configuration, execute the script with the `-g` flag to javascript code which constructs the necessary skeleton in a new GeoGebra session.
Once an annotation is to be loaded, also load the source image where the annotation is from as a 1:1 background image, such that it can be used to visually guide the correction of the annotation in question.
Check the output of the script with the `-h`/`--help` flag/option for more information.
Please be mindful of the fact that the script has been adapted for our original deepscores dataset.

[`tools/merge_ds.py`](../../tree/dev/tools/merge_ds.py) - This script merges two datasets.
To adjust the datasets to be merged, edit lines 8-11 and execute the script.

[`tools/visualize_ds.py`](../../tree/dev/tools/visualize_ds.py) - This script visualizes the annotations in a dataset.
To adjust the dataset to be visualized, edit line 81 and execute the script.

[`tools/inference_imslp.py`](../../tree/dev/tools/inference_imslp.py) - This script runs an inference step on a given checkpoint and visualizes the result.
To adjust, edit lines 9 and 15-16 and execute the script.

[`tools/ds_dataset.py`](../../tree/dev/tools/ds_dataset.py) - This script creates a new dataset.
To adjust the dataset to be visualized, edit lines 14-35 and execute the script.
The annotation input can be either a JSON file or a CSV file.
Refer to the code for further information.

[`tools/detection_service.py`](../../tree/dev/tools/detection_service.py) - This script boots up a Flask server for a simple GUI to interact with a given checkpoint for inference.
To adjust, edit lines 19 and 20.
