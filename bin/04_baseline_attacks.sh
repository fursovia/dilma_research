#!/usr/bin/env bash

# 1. for each dataset
## attack first K examples of the test (!) split (not validation) using `textattack` package
## and two attacks (fgsm and textfool) from our repository
RESULTS_DIR='results'
NUM_EXAMPLES=1000

for dataset in "rotten_tomatoes" "ag_news" "sst2" "dstc"; do
    for model in "lstm"; do
        for attacker in "deepwordbug" "hotflip" "textbugger" "pwws"; do
            textattack attack \
                --recipe ${attacker} \
                --model-from-file models/${dataset}/lstm/load_lstm.py \
                --dataset-from-file data/${dataset}/load_valid.py \
                --num-examples ${NUM_EXAMPLES} \
                --log-to-csv ${RESULTS_DIR}/${model}_${dataset}_${attacker}.csv \
                --disable-stdout
        done
    done
done


# 2. convert output files from textattack to our format (save to ./results folder) [!]
# 3. evaluate attacks using `dilma/commands/evaluate.py` script
## (in white-box/black-box scenario. SOTA models to fool)
# 4. save table with metrics to ./results folder
# 5*. attack M examples of the train (!) set (will be needed for adversarial training and detection)
ADV_TRAIN_DIR='adv_training_data'
NUM_EXAMPLES_TRAIN=10000

for dataset in "rotten_tomatoes" "ag_news" "sst2" "dstc"; do
    for model in "lstm"; do
        for attacker in "deepwordbug" "hotflip" "textbugger" "pwws"; do
            textattack attack \
                --recipe ${attacker} \
                --model-from-file models/${dataset}/lstm/load_lstm.py \
                --dataset-from-file data/${dataset}/load_substitute_train.py \
                --num-examples ${NUM_EXAMPLES_TRAIN} \
                --log-to-csv ${ADV_TRAIN_DIR}/${model}_${dataset}_${attacker}.csv \
                --disable-stdout
        done
    done
done