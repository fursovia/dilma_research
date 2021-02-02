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

RESULT_DIR='RESULTS'
DISCRIMINATOR_MODEL_PATH='discriminator_save'

for dataset in "sst2" "ag_news" "rotten_tomatoes" "dstc"; do
  for attacker in "dilma" "dilma_with_deep_levenshtein" "sampling_fool"; do
      PYTHONPATH=. python scripts/train_discrimanor.py \
          --model lstm \
          --output-dir ${DISCRIMINATOR_MODEL_PATH} \
          --metrics-output-path ${RESULT_DIR}/discriminator__${dataset}__${attacker}.json \
          --dataset-folder new_data/${dataset}/${attacker}/data.json \
          --epochs 50 \
          --batch-size 250 \
          --learning-rate 5e-4 \
          --early-stopping-epochs 50

  done
        
  for attacker in "deepwordbug" "hotflip" "textbugger" "pwws"; do
      
      PYTHONPATH=. python scripts/train_discrimanor.py \
          --model lstm \
          --output-dir ${DISCRIMINATOR_MODEL_PATH} \
          --metrics-output-path ${RESULT_DIR}/discriminator__${dataset}__${attacker}.json \
          --dataset-folder ${RESULT_DIR}/lstm_${dataset}_${attacker}.csv \
          --epochs 50 \
          --batch-size 250 \
          --learning-rate 5e-4 \
          --early-stopping-epochs 50
  done
done

rm -r ${DISCRIMINATOR_MODEL_PATH}