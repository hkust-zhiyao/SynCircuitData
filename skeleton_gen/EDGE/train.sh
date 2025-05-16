#!/bin/bash

python train.py \
        --epochs 20000 \
        --num_generation 64 \
        --diffusion_dim 64 \
        --diffusion_steps 128 \
        --device cuda:0 \
        --dataset skeleton \
        --batch_size 4 \
        --clip_value 1 \
        --lr 1e-4 \
        --optimizer adam \
        --final_prob_edge 1 0 \
        --sample_time_method importance \
        --check_every 50 \
        --eval_every 200 \
        --noise_schedule linear \
        --dp_rate 0.1 \
        --loss_type vb_ce_xt_prescribred_st \
        --arch TGNN_degree_guided \
        --parametrization xt_prescribed_st \
        --degree \
        --num_heads 8 8 8 8 1 