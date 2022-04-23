export NGPUS=2
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
# train on source data
#python -m torch.distributed.launch --nproc_per_node=$NGPUS uda/train.py
python uda/train.py