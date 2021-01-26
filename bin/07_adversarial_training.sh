#!/usr/bin/env bash

# 0. for each text classification dataset and for each attack
# 1. create dataset: take adversarial examples
## from 04, 05 steps and add them to the train split
# 2. re-train target classifiers [BERT]
# 3. create a table with metrics on the valid/test splits (save to ./results)
## [we check that the performance hasn't changed]
# 4. calculate attack metrics on the re-trained classifiers (save table to ./results)

RESULTS_DIR='results'
ADV_TRAIN_DIR='adv_training_data'
batch_size=64
logging_steps=100

logger_name=${1:-"logging_adv_train"}

echo "================================" >> ${logger_name}.txt
echo "dataset ${dataset} ADVERSARIAL TRAINING" >> ${logger_name}.txt

#train on adversarial data
for dataset in "rotten_tomatoes" "ag_news" "sst2" "dstc"; do
    for attacker in "deepwordbug" "textbugger" "pwws" "hotflip"; do

        if [ $dataset == "dstc" ]; then
        num_labels=46
        elif [ $dataset == "news" ]; then
        num_labels=4
        else
        num_labels=2
        fi

        echo "================================" >> ${logger_name}.txt

        for num_examples in 200 500 1000 2000 3000 4000 5000; do
            echo "Train ${attacker} start; num_examples ${num_examples}" >> ${logger_name}.txt
            python scripts/train_model.py \
                --model_name_or_path datasets/${dataset}/model/ \
                --config_name datasets/${dataset}/model/ \
                --tokenizer_name datasets/${dataset}/model/ \
                --task_name ${dataset} \
                --do_train \
                --do_eval \
                --logging_steps ${logging_steps} \
                --per_device_train_batch_size ${batch_size} \
                --per_device_eval_batch_size ${batch_size} \
                --learning_rate 2e-5 \
                --num_train_epochs 25 \
                --save_steps -1 \
                --evaluation_strategy "epoch" \
                --save_total_limit 0 \
                --evaluate_during_training \
                --output_dir models/${dataset}/adv_trained_model/ \
                --adversarial_data_path ${ADV_TRAIN_DIR}/lstm_${dataset}_${attacker}.csv
                --adversarial_training_original_data_amount ${num_examples} \
                --adversarial_training_perturbed_data_amount ${num_examples} \
                --use_custom_trainer \
                --save_last


            echo "Train ${attacker} end num_examples ${num_examples}" >> ${logger_name}.txt
            echo "Test ${attacker} start " >> ${logger_name}.txt

            PYTHONPATH=. python dilma/commands/evaluate.py \
                ${RESULTS_DIR}/${model}_${dataset}_${attacker}.csv \
                --save-to ${ADV_TRAIN_DIR}/${dataset}_${num_examples}.json \
                --target-clf-path models/${dataset}/adv_trained_model/ \
                --output-from-textattack \
                --num-labels ${num_labels}

            echo "Test ${attacker} end" >> ${logger_name}.txt
        done
    done
done