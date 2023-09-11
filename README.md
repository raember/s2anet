# Real World Music Object Recognition (under construction)

The *RealScores* test dataset can be found in RealScores.zip in this repository.

### ScoreAug

### UDA

### Confidence Ratings

For more details, please have a look at branch `dev-sage`

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
