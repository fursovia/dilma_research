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

    for config_path in ${CONFIG_DIR}/*.jsonnet; do
        attack_name=$(basename ${config_path})
        attack_name="${attack_name%.*}"

        echo ">>> Attacking ${dataset_name} with ${attack_name}"

        RESULTS_PATH=${RESULTS_DIR}/${DATE}/${dataset_name}/${attack_name}
        mkdir -p ${RESULTS_PATH}

        CLF_PATH=${PRESETS_DIR}/models/${dataset_name}.tar.gz \
          PYTHONPATH=. python dilma/commands/attack.py \
          ${config_path} \
          ${data_path} \
          --out-dir ./results/${DATE}__${dataset_name}__${attack_name} \
          --samples ${NUM_SAMPLES}

        if [ $dataset_name == "dstc" ]; then
        num_labels=46
        elif [ $dataset_name == "ag_news" ]; then
        num_labels=4
        else
        num_labels=2
        fi

        PYTHONPATH=. python dilma/commands/evaluate.py \
          ./results/${DATE}__${dataset_name}__${attack_name}/data.json \
          --save-to ./results/${DATE}__${dataset_name}__${attack_name}/metrics.json \
          --target-clf-path ./presets/transformer_models/${dataset_name} \
          --num-labels ${num_labels}
    done
done

python dilma/commands/aggregate.py ${RESULTS_DIR}/${DATE}
