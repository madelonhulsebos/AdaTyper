#!/usr/bin/env bash

data_identifier=17

# --device cuda \

python typetabert/train.py \
    --train_set_id ${data_identifier} \
    --models_dir typetabert/models/ \
    --num_epoch 3 \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --device cpu \
    --descr typetabert