#!/usr/bin/env bash

CONFIG_PATH=$1
DATA_DIR=$2

TRAIN_PATH=${DATA_DIR}/train.json
VALID_PATH=${DATA_DIR}/valid.json

DATASET_NAME=$(basename ${DATA_DIR})
CONFIG_NAME=$(basename ${CONFIG_PATH})
CONFIG_NAME="${CONFIG_NAME%.*}"
DATE=$(date +%H%M%S-%d%m)
EXP_NAME=${DATE}-${DATASET_NAME}-${CONFIG_NAME}
LOG_DIR=./logs/${EXP_NAME}


TRAIN_DATA_PATH=${TRAIN_PATH} \
    VALID_DATA_PATH=${VALID_PATH} \
    allennlp train ${CONFIG_PATH} \
    --serialization-dir ${LOG_DIR} \
    --include-package dilma