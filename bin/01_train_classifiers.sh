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
# 1) TRAIN SUBSTITUTE CLASSIFIERS
# TODO: early stopping?
for dataset in "rotten_tomatoes" "ag_news" "dstc" "sst2"; do
    PYTHONPATH=. python dilma/commands/train_textattack.py \
        --model lstm \
        --output-dir models/${dataset}/lstm \
        --dataset-folder data/${dataset}/ \
        --epochs 50 \
        --batch-size 128 \
        --learning-rate 5e-4
done


for dataset in "rotten_tomatoes" "ag_news" "dstc" "sst2"; do

    EXP_NAME=${DATE}-${dataset}-${CONFIG_NAME}
    LOG_DIR=./logs/${EXP_NAME}

    TRAIN_DATA_PATH=./data/${dataset}/substitute_train.json \
      VALID_DATA_PATH=./data/${dataset}/valid.json \
      allennlp train ${CONFIG_PATH} \
      --serialization-dir ${LOG_DIR} \
      --include-package dilma

    cp ${LOG_DIR}/model.tar.gz ./presets/${dataset}.tar.gz
done


# 2) TRAIN TARGET CLASSIFIER (THE ONES TEXTATTACK DOESNT HAVE)

for dataset in "rotten_tomatoes" "ag_news" "dstc" "sst2"; do
    PYTHONPATH=. python scripts/train_model.py \
        --model_name_or_path 'roberta-base' \
        --config_name 'roberta-base' \
        --tokenizer_name 'roberta-base' \
        --task_name ${dataset} \
        --do_train \
        --do_eval \
        --logging_steps 100 \
        --per_device_train_batch_size 64 \
        --per_device_eval_batch_size 64 \
        --learning_rate 2e-5 \
        --num_train_epochs 5 \
        --save_steps -1 \
        --evaluation_strategy "epoch" \
        --save_total_limit 0 \
        --evaluate_during_training \
        --output_dir models/${dataset}/bert/ \
        --use_custom_trainer \
        --use_early_stopping
done