#!/usr/bin/env bash

topic='id'
wget https://zenodo.org/record/6517052/files/${topic}_tables_licensed.zip?download=1 -O data/${topic}.zip
unzip -d data/${topic}/ -q data/${topic}.zip
rm data/${topic}.zip
