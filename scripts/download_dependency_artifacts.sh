#!/usr/bin/env bash

# Download (vertical) TaBERT model
cd typetabert/tabert/
gdown https://drive.google.com/uc?id=1xKoqw7Jx-nN81xjzY1ccBO8B4Nl4NuBr
unzip -j tabert.zip
rm tabert.zip


# Download finetuned TypeTaBERT model
cd ../typetabert/models/
gdown https://drive.google.com/uc?id=1Q0KwxU1R8TgZU1fB9007lHX7W28tbhEo
unzip -j coltype_tabert.zip
rm coltype_tabert.zip


# Download AdaTyper estimators
cd ../../../models/
gdown https://drive.google.com/uc?id=1ZUQpMNhBDJUFtQmdR3X_6ay118kUyB_n


# Download training data for adaptation
mkdir -p ../data/training_data/raw/
mkdir -p ../data/adaptation/generated_training_data/
cd ../data/training_data/raw/

gdown https://drive.google.com/uc?id=1nD-_-vkEakuJeQmK-ks_f43mJ-3qB9Fo
