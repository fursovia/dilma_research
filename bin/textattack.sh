#!/usr/bin/env bash

RESULTS_DIR=./results
NUM_EXAMPLES=500


echo "AG news dataset"

for dataset in "ag-news" "mr" "sst2"; do
    for model in "lstm" "roberta-base"; do
        for attacker in "deepwordbug" "hotflip" "textbugger" "pwws"; do
            textattack attack --recipe ${attacker} \
                --model ${model}-${dataset} \
                --num-examples ${NUM_EXAMPLES} \
                --log-to-csv ${RESULTS_DIR}/${model}_${dataset}_${attacker}.csv
        done
    done
done
