#!/usr/bin/env bash

# for each dataset train (SUBSTITUTE CLASSIFIERS) [x% of the train set]
## 1) lstm classifier (allennlp, textattack) [check if metrics are the same]
## 2) freezed bert + dense layers classifier
# copy all `model.tar.gz` to ./presets folder
# create a table with metrics on the valid/test splits (save to ./results)

# textattack train --model lstm --dataset glue^cola --max-length 32 --batch-size 64 --epochs 1 --pct-dataset .01
## [--dataset-path instead of --dataset]
# allennlp train configs/models/clf_lstm.jsonnet

# bert-base-uncased-qqp for QQP

DATE=$(date +%H%M%S-%d%m)
CONFIG_PATH=./configs/models/clf_lstm.jsonnet
CONFIG_NAME=$(basename ${CONFIG_PATH})
CONFIG_NAME="${CONFIG_NAME%.*}"

#now don't work with qqp
for dataset in "rotten_tomatoes" "ag_news" "dstc" "sst2"; do
    PYTHONPATH=. python dilma/commands/train_textattack.py \
        --model lstm \
        --output-dir models/${dataset}/lstm \
        --dataset-folder data/${dataset}/ \
        --epochs 50 \
        --batch-size 128 \
        --learning-rate 5e-4

    EXP_NAME=${DATE}-${dataset}-${CONFIG_NAME}
    LOG_DIR=./logs/${EXP_NAME}

    TRAIN_DATA_PATH=./data/${dataset}/substitute_train.json \
      VALID_DATA_PATH=./data/${dataset}/valid.json \
      allennlp train ${CONFIG_PATH} \
      --serialization-dir ${LOG_DIR} \
      --include-package dilma

    cp ${LOG_DIR}/model.tar.gz ./presets/${dataset}.tar.gz
done
