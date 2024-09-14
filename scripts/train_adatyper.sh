#!/usr/bin/env bash

# To lookup from the preprocessed data
data_identifier=19
# To be set by yourself
pipeline_identifier=19

python adatyper/pipeline.py \
    --train_set_id ${data_identifier} \
    --test_set_id ${data_identifier} \
    --pipeline_identifier ${pipeline_identifier}