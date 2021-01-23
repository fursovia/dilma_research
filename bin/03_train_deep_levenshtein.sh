#!/usr/bin/env bash

# 1. train deep levenshtein model
# 2. save to ./presets/deep_levenshtein


# allennlp train configs/models/deep_levenshtein.jsonnet

CONFIG_PATH=./configs/models/deep_lev.jsonnet
DATE=$(date +%H%M%S-%d%m)
EXP_NAME=${DATE}-deep_levenshtein
LOG_DIR=./logs/${EXP_NAME}

TRAIN_DATA_PATH=./data/deeplev/train.json \
  VALID_DATA_PATH=./data/deeplev/valid.json \
  allennlp train ${CONFIG_PATH} \
  --serialization-dir ${LOG_DIR} \
  --include-package dilma


cp ${LOG_DIR}/model.tar.gz ./presets/deeplev.tar.gz