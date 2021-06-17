#! /usr/bin/bash


### hangzhou
## od
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 python dcrnn_train_pytorch.py \
    --config
## do
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=1 python dcrnn_train_pytorch.py \
    --config data/config/dcrnn_do_hz_96.yaml

### shanghai
## od
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python dcrnn_train_pytorch.py \
    --config data/model/dcrnn_od_sh.yaml
## do
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=1 python dcrnn_train_pytorch.py \
    --config data/config/dcrnn_do_sh_96.yaml