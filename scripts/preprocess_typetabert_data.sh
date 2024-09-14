#!/usr/bin/env bash

# This is not needed for the current implementation of TypeTaBERT
# This will preprocess data to be compliant with TaBERT input.

# The below can be a sequence of topics or single topic
topics=("object" "id" "abstraction")

python typetabert/preprocess.py \
    --table_topics ${topics[@]} \
    --input_data_dir data/ \
    --output_data_dir typetabert/data/preprocessed/