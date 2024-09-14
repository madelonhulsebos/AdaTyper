#!/usr/bin/env bash

topics=("id" "object" "whole" "thing")
python adatyper/preprocess.py \
    --dataset "gittables" \
    --dirs ${topics[@]}
