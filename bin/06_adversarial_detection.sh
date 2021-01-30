#!/usr/bin/env bash

# 0. for each text classification dataset and each attack
# 1. create dataset: take adversarial examples
## from 04, 05 steps and corresponding original examples
# 2. train a LSTM classifier to detect adversarial-vs-non-adversarial
# 3. calculate ROC AUC, Accuracy metrics. Save table to ./results

for dataset in "rotten_tomatoes" "sst2" "ag_news" "dstc"; do
  PYTHONPATH=. python scripts/convert_attack_output_for_adv_detection_trainuing.py --dataset-dir ./data/${dataset}

  TRAIN_DATA_PATH=./data/${dataset}/ad_train.json \
      VALID_DATA_PATH=./data/${dataset}/ad_valid.json \
      allennlp train ${CONFIG_PATH} \
      --serialization-dir ${LOG_DIR} \
      --include-package dilma
done