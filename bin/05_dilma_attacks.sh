#!/usr/bin/env bash

# 1. for each dataset
## attack first K examples of the test (!) split (not validation) with DILMA, DILMA+, SamplingFool
# 2. covert output files from textattack to our format (save to ./results folder)
# 3. evaluate attacks using `dilma/commands/evaluate.py` script
## (in white-box scenario. SOTA models to fool)
# 4. save table with metrics to ./results folder
# 5*. attack M examples of the train (!) set (will be needed for adversarial training and detection)


NUM_SAMPLES=${1:-"500"}

DATA_DIR="./data"
CONFIG_DIR="./configs/attacks"
PRESETS_DIR="./presets"
RESULTS_DIR="./results"
DATE=$(date +%H%M%S-%d%m)


for dataset_name in "rotten_tomatoes" "sst2" "ag_news" "dstc"; do
    data_path=${DATA_DIR}/${dataset_name}/test.json

    for attack_name in "dilma" "dilma_with_deep_levenshtein" "sampling_fool" "fgsm"; do
        config_path=./configs/attacks/${attack_name}.jsonnet

        echo ">>> Attacking ${dataset_name} with ${attack_name}"

        RESULTS_PATH=${RESULTS_DIR}/${DATE}/${dataset_name}/${attack_name}
        mkdir -p ${RESULTS_PATH}

        CLF_PATH=${PRESETS_DIR}/models/${dataset_name}.tar.gz \
          PYTHONPATH=. python dilma/commands/attack.py \
          ${config_path} \
          ${data_path} \
          --out-dir ${RESULTS_PATH} \
          --samples ${NUM_SAMPLES}

        if [ $dataset_name == "dstc" ]; then
        num_labels=46
        elif [ $dataset_name == "ag_news" ]; then
        num_labels=4
        else
        num_labels=2
        fi

        PYTHONPATH=. python dilma/commands/evaluate.py \
          ${RESULTS_PATH}/data.json \
          --save-to ${RESULTS_PATH}/metrics.json \
          --target-clf-path ./presets/transformer_models/${dataset_name} \
          --num-labels ${num_labels}
    done
done


for dataset_name in "qqp"; do
    data_path=${DATA_DIR}/${dataset_name}/test.json

    for attack_name in "pair_dilma"; do
        config_path=./configs/attacks/${attack_name}.jsonnet

        echo ">>> Attacking ${dataset_name} with ${attack_name}"

        RESULTS_PATH=${RESULTS_DIR}/${DATE}/${dataset_name}/${attack_name}
        mkdir -p ${RESULTS_PATH}

        CLF_PATH=${PRESETS_DIR}/models/${dataset_name}.tar.gz \
          PYTHONPATH=. python dilma/commands/attack.py \
          ${config_path} \
          ${data_path} \
          --out-dir ${RESULTS_PATH} \
          --samples ${NUM_SAMPLES}

        PYTHONPATH=. python dilma/commands/evaluate.py \
          ${RESULTS_PATH}/data.json \
          --save-to ${RESULTS_PATH}/metrics.json \
          --target-clf-path ./presets/transformer_models/${dataset_name} \
          --num-labels ${num_labels}
    done

PYTHONPATH=. python scripts/parse_attack_metrics.py ${RESULTS_DIR}/${DATE}
