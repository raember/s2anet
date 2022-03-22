# Snapshot Ensemble

## Train Snapshot Ensemble
This ensemble method is based on a "cosine restart" LR scheduling. The used MMCV library implements this LR strategy and therefore no customization is required.
However, no constant learning rate can be defined for the warmup (even if `warmup=constant` is inserted in the config, the LR decreases during the warmup). 
This limitation can be overcome by performing two subsequent trainings.

For this purpose, the config of the first training cycle can be set to:

```python
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=260,
    warmup_ratio=1.0 / 3,
    gamma = 0.5,
    step=[300, 700])
```
And the config for the second training cycle:
```python
optimizer = dict(type='SGD', lr=0.0075, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineRestart',
    warmup='constant',
    warmup_iters=0,
    warmup_ratio=1. / 3,
    periods=[n_warmup_epochs * n_steps] + ([snapshot_epoch_interval * n_steps] * n_snapshots),
    by_epoch=False,  # cannot use epochs, lr is only updated after epoch -> cosine annealing needs update per step
    restart_weights=[1] * (n_snapshots + 1),
    min_lr_ratio=1e-5,
)
```

The training can then be run with the command:
````bash
python tools/train.py config_1 --work_dir <working dir> && python tools/train.py config_2 --work_dir <working dir> --resume_from <working dir>/latest.pth
````

## Evaluate Snapshot Ensembles
Run the script `test_BE_multi_model.py` to create predictions
````bash
python tools/test_BE_multi_model.py config_2 --checkpoints <path to checkpoints> --data dsv2 --out <out filepath>
````

(Optional): Calculate the overlap between multiple snapshots. The predictions (json file) created with the previous command are used as ground truth
or as prediction. 
````bash
python DeepScoresV2_s2anet/analyze_BE_output/snapshot_overlap.py config_2 <path to json used as gt> --jsons <path to jsons used as proposal> ----out_dir <out filepath 2>
````

Finally, run the script `analyze_errors.py` to create a .csv file with the predictions.
````bash
python DeepScoresV2_s2anet/analyze_errors.py --ev_folder <out filepath> --filename dsv2_metrics.pkl
````

(Optional if the overlaps were calculated):
````bash
python DeepScoresV2_s2anet/analyze_errors.py --ev_folder <out filepath 2> --filename .pkl
````

Calculate the AP of the ensemble (based on multiple models)
````bash
python DeepScoresV2_s2anet/analyze_BE_output/compare_dsv2_metrics_multi_model.py
````

## Visualize Predictions
````bash
python DeepScoresV2_s2anet/analyze_BE_output/draw_WBF_for_multi_model.py
````