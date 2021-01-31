#!/usr/bin/env bash

# 0. for each text classification dataset and each attack
# 1. create dataset: take adversarial examples
## from 04, 05 steps and corresponding original examples
# 2. train a LSTM classifier to detect adversarial-vs-non-adversarial
# 3. calculate ROC AUC, Accuracy metrics. Save table to ./results

DATE=$(date +%H%M%S-%d%m)
CONFIG_PATH=./configs/models/clf_lstm.jsonnet

for path in ./data/detection/*; do

  name=$(basename ${path})
  name="${name%.*}"

  EXP_NAME="detection-${name}"
  LOG_DIR=./logs/${DATE}/${EXP_NAME}

  TRAIN_DATA_PATH=${path}/train.json \
      VALID_DATA_PATH=${path}/valid.json \
      allennlp train ${CONFIG_PATH} \
      --serialization-dir ${LOG_DIR} \
      --include-package dilma
done
