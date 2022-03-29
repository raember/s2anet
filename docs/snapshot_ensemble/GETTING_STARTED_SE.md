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
Run the script `DeepScoresV2/analyze_ensembles/evaluate_all_snapshot_ensemble.py` to run a complete evaluation of the snapshots
````bash
python DeepScoresV2/analyze_ensembles/evaluate_all_snapshot_ensemble.py <path to config file> --checkpoints <path to checkpoints> -out <out filepath>
````

For a quick overview, the following files are the most important ones:
- `<out filepath>/compare metrics/IOU_0-5_class_wise_APs.csv`: Provides an overview about all the models. It shows the average/min/max ap per class as well as the performance of weighted box fusion (wbf)
- `<out filepath>/overlap/overlap_matrix.csv`: The overlap between ensembles - the models per columns were used as ground truth, the models per row as prediction
- `<out filepath>/overview_metrics.csv`: Shows the ap per class for all models as well as for wbf
