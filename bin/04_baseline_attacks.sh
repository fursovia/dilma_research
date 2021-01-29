#!/usr/bin/env bash

# 1. for each dataset
## attack first K examples of the test (!) split (not validation) using `textattack` package
## and two attacks (fgsm and textfool) from our repository
RESULTS_DIR='results'
NUM_EXAMPLES=1000

if [ ! -d ${RESULTS_DIR} ]; then
  mkdir -p ${RESULTS_DIR};
fi

for dataset in "qqp" "rotten_tomatoes" "ag_news" "sst2" "dstc"; do
    for model in "lstm"; do
        for attacker in "deepwordbug" "hotflip" "textbugger" "pwws"; do
            
            if [ $dataset == "qqp" ]; then
            textattack attack \
                --recipe ${attacker} \
                --model-from-file ./presets/transformer_substitute_models/${dataset}/load_model.py
                --dataset-from-file data/${dataset}/load_test.py \
                --num-examples ${NUM_EXAMPLES} \
                --log-to-csv ${RESULTS_DIR}/roberta_${dataset}_${attacker}.csv \
                --disable-stdout
            
            else
            textattack attack \
                --recipe ${attacker} \
                --model-from-file ./presets/textattack_models/${dataset}/load_lstm.py \
                --dataset-from-file data/${dataset}/load_test.py \
                --num-examples ${NUM_EXAMPLES} \
                --log-to-csv ${RESULTS_DIR}/${model}_${dataset}_${attacker}.csv \
                --disable-stdout
            fi
            
        done
    done
done


# 2. convert output files from textattack to our format (save to ./results folder) [!]
# 3. evaluate attacks using `dilma/commands/evaluate.py` script
## (in white-box/black-box scenario. SOTA models to fool)
# 4. save table with metrics to ./results folder

for dataset in "rotten_tomatoes" "ag_news" "sst2" "dstc" "qqp"; do
    
    if [ $dataset == "dstc" ]; then
    num_labels=46
    elif [ $dataset == "ag_news" ]; then
    num_labels=4
    else
    num_labels=2
    fi
    
    if [ $dataset == "qqp" ]; then
    model="roberta"
    else
    model="lstm"
    fi
    
    for attacker in "deepwordbug" "hotflip" "textbugger" "pwws"; do
        
        PYTHONPATH=. python dilma/commands/evaluate.py \
            ${RESULTS_DIR}/${model}_${dataset}_${attacker}.csv \
            --save-to ${RESULTS_DIR}/${model}_${dataset}_${attacker}.json \
            --target-clf-path ./presets/transformer_models/${dataset} \
            --output-from-textattack \
            --num-labels ${num_labels}
    done
done

PYTHONPATH=. python scripts/parse_attack_metrics.py 

# 5*. attack M examples of the train (!) set (will be needed for adversarial training and detection)
ADV_TRAIN_DIR='adv_training_data'
NUM_EXAMPLES_TRAIN=10000

if [ ! -d ${ADV_TRAIN_DIR} ]; then
  mkdir -p ${ADV_TRAIN_DIR};
fi

for dataset in "rotten_tomatoes" "ag_news" "sst2" "dstc"; do
    for model in "lstm"; do
        for attacker in "deepwordbug" "hotflip" "textbugger" "pwws"; do
            if [ $dataset == "qqp" ]; then
            textattack attack \
                --recipe ${attacker} \
                --model-from-file ./presets/transformer_substitute_models/${dataset}/load_model.py
                --dataset-from-file data/${dataset}/load_substitute_train.py \
                --num-examples ${NUM_EXAMPLES} \
                --log-to-csv ${RESULTS_DIR}/roberta_${dataset}_${attacker}.csv \
                --disable-stdoutT
            
            else
            
            textattack attack \
                --recipe ${attacker} \
                --model-from-file ./presets/textattack_models/${dataset}/load_lstm.py \
                --dataset-from-file data/${dataset}/load_substitute_train.py \
                --num-examples ${NUM_EXAMPLES_TRAIN} \
                --log-to-csv ${ADV_TRAIN_DIR}/${model}_${dataset}_${attacker}.csv \
                --disable-stdout
        done
    done
done