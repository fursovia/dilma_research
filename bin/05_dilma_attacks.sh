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


for dataset_name in "sst2" "ag_news" "rotten_tomatoes"; do
    data_path=${DATA_DIR}/${dataset_name}/test.json

    for config_path in ${CONFIG_DIR}/*.jsonnet; do
        attack_name=$(basename ${config_path})
        attack_name="${attack_name%.*}"

        RESULTS_PATH=${RESULTS_DIR}/${DATE}/${dataset_name}/${attack_name}
        mkdir -p ${RESULTS_PATH}

        CLF_PATH=${PRESETS_DIR}/models/${dataset_name}.tar.gz \
          PYTHONPATH=. python dilma/commands/attack.py ${config_path} ${data_path} --samples ${NUM_SAMPLES}

        PYTHONPATH=. python dilma/commands/evaluate.py ${RESULTS_PATH}/output.json \
            --save-to=${RESULTS_PATH}/metrics.json \
            --target-clf-path=${PRESETS_DIR}/${dataset_name}/models/target_clf/${TARGET_CONFIG_NAME}.tar.gz
    done
done

python dilma/commands/aggregate.py ${RESULTS_DIR}/${DATE}