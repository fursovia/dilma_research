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
CONFIG_PATH=./configs/models/clf_bcn.jsonnet
CONFIG_NAME=$(basename ${CONFIG_PATH})
CONFIG_NAME="${CONFIG_NAME%.*}"

mkdir -p ./presets/textattack_models


for dataset in "rotten_tomatoes" "ag_news" "dstc" "sst2"; do

    EXP_NAME=-${dataset}-${CONFIG_NAME}
    LOG_DIR=./logs/${DATE}/${EXP_NAME}

    if [ $dataset == "dstc" ]; then
      num_labels=46
    elif [ $dataset == "ag_news" ]; then
      num_labels=4
    else
      num_labels=2
    fi

    NUM_CLASSES=num_labels \
      TRAIN_DATA_PATH=./data/${dataset}/train.json \
      VALID_DATA_PATH=./data/${dataset}/valid.json \
      allennlp train ${CONFIG_PATH} \
      --serialization-dir ${LOG_DIR} \
      --include-package dilma

    cp ${LOG_DIR}/model.tar.gz ./presets/models/target_${dataset}.tar.gz
done
