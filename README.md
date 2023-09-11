# Confidence Rating

This branch contains the code to reproduce the results from the confidence ratings.

### Changes

The following changes are made to the configuration in order to create `SnapshotEnsembles`:

# Training Ensembles: Changes in the Config File



1. Increase the learning rate

   Old:

   ```python
   optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
   ```

   New:

   ```python
   optimizer = dict(type='SGD', lr=0.0075, momentum=0.9, weight_decay=0.0001)
   ```

   

2. Change the LR Policy

   Old:

   ```python
   lr_config = dict(
       policy='step',
       warmup='linear',
       warmup_iters=500,
       warmup_ratio=1.0 / 3,
       gamma = 0.5,
       step=[300, 700])
   ```

   New:

   ```python
   n_warmup_epochs = 500
   n_snapshots = 25
   snapshot_epoch_interval = 20
   
   n_steps = -(-1362 // data['imgs_per_gpu']) # -(-numerator // denominator) is a way to round up an integer without importing a module like math
   lr_config = dict(
       policy='CosineRestart',
       warmup='constant',
       warmup_iters=n_warmup_epochs * n_steps,  # 1 epoch = 1362 steps
       warmup_ratio=1. / 3,
       periods=[n_warmup_epochs * n_steps] + ([snapshot_epoch_interval * n_steps] * n_snapshots),
       by_epoch=False,  # cannot use epochs, lr is only updated after epoch -> cosine annealing needs update per step
       warmup_by_epoch=False,
       restart_weights=[1] * (n_snapshots + 1),
       min_lr_ratio=1e-5,
   )
   ```

3. Create a checkpoint every 10 epochs:

   ```python
   checkpoint_config = dict(interval=10)
   ```

   

4. Change the total number of epochs:

   ```python
   total_epochs = n_warmup_epochs + n_snapshots * snapshot_epoch_interval
   ```

### Train the Model

The model can be trained using the following command:

```bash
   python tools/train.py configs/deepscoresv2/s2anet_r50_fpn_1x_deepscoresv2_sage_halfrez_crop.py
```

### Test the Model

```bash
python DeepScoresV2_s2anet/analyze_ensembles/evaluate_all_snapshot_ensemble.py
  configs/deepscoresv2/s2anet_r50_fpn_1x_deepscoresv2_sage_halfrez_crop.py
--checkpoints
  <...>
```
