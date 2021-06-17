#! /usr/bin/bash

### hangzhou OD
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 python train1.py \
    --data data/hangzhou/OD/OD_26 \
    --adjdata data/hangzhou/graph_hz_conn.pkl \
    --exp_base "data" \
    --runs "hz_od" \
    --dropout 0.001 \
    --epochs 400 \
    --learning_rate 0.008 \
    --nhid 96 \
    --out_dim 26 \
    --in_dim 26 \
    --seq_length 4 \
    --num_nodes 80 \
    --gcn_bool \
    --addaptadj \
    --train_tyep "od" \
    --randomadj

### hangzhou DO
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python train1.py \
    --data data/hangzhou/DO/DO_26 \
    --adjdata data/hangzhou/graph_hz_conn.pkl \
    --exp_base "data" \
    --runs "hz_do" \
    --dropout 0.001 \
    --epochs 400 \
    --learning_rate 0.008 \
    --nhid 96 \
    --out_dim 26 \
    --in_dim 26 \
    --seq_length 4 \
    --num_nodes 80 \
    --gcn_bool \
    --addaptadj \
    --train_type "do" \
    --randomadj


###shanghai OD
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=1 python train1.py \
    --data data/shanghai/OD/OD_76 \
    --adjdata data/shanghai/graph_sh_conn.pkl \
    --exp_base "data" \
    --runs "sh" \
    --dropout 0.001 \
    --epochs 300 \
    --learning_rate 0.01 \
    --nhid 96 \
    --in_dim 76 \
    --out_dim 76 \
    --num_nodes 288 \
    --seq_length 4 \
    --gcn_bool \
    --addaptadj \
    --randomadj

###shanghai DO
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 python train1.py \
    --data data/shanghai/DO/DO_76 \
    --adjdata data/shanghai/graph_sh_conn.pkl \
    --exp_base "data" \
    --runs "sh_do" \
    --dropout 0.001 \
    --epochs 300 \
    --learning_rate 0.01 \
    --nhid 96 \
    --in_dim 76 \
    --out_dim 76 \
    --num_nodes 288 \
    --seq_length 4 \
    --gcn_bool \
    --addaptadj \
    --train_type "do" \
    --randomadj

tar --exclude='./data' -zcvf /backup/filename.tgz .