#!/usr/bin/env bash

# 0. for each text classification dataset and for each attack
# 1. create dataset: take adversarial examples
## from 04, 05 steps and add them to the train split
# 2. re-train target classifiers [BERT]

ADV_TRAIN_DIR='adv_training_data'
ADV_TRAIN_DIR_RESULT='after_training_data'
batch_size=64
logging_steps=100

dataset=$1
logger_name=$2
num_labels=$3

echo "================================" >> ${logger_name}.txt
echo "dataset ${dataset} ADVERSARIAL TRAINING" >> ${logger_name}.txt

#train on adversarial data
for attacker in "deepwordbug" "textbugger" "pwws" "hotflip"; do
    echo "================================" >> ${logger_name}.txt
    
    python scripts/eval_attack.py \
            --model_name_or_path models/${dataset}/bert/ \
            --config_name datasets/${dataset}/adv_trained_model/ \
            --tokenizer_name datasets/${dataset}/adv_trained_model/ \
            --attack_file ${ADV_TRAIN_DIR}/lstm_${dataset}_${attacker}.csv \
            --path_to_save ${ADV_TRAIN_DIR_RESULT}/lstm_${dataset}_${attacker}_0.csv  \
            --batch_size ${batch_size} \
            --num_labels ${num_labels}
    
    for num_examples in 100 200 300 500 750 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000; do_eval
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
            --num_train_epochs 50 \
            --save_steps -1 \
            --evaluation_strategy "epoch" \
            --save_total_limit 0 \
            --evaluate_during_training \
            --output_dir models/${dataset}/adv_trained_model/ \
            --adversarial_data_path ${ADV_TRAIN_DIR}/lstm_${dataset}_${attacker}.csv
            --adversarial_training_original_data_amount ${num_examples} \
            --adversarial_training_perturbed_data_amount ${num_examples} \
            --use_custom_trainer


        echo "Train ${attacker} end num_examples ${num_examples}" >> ${logger_name}.txt
        echo "Test ${attacker} start " >> ${logger_name}.txt
        python scripts/eval_attack.py \
            --model_name_or_path models/${dataset}/adv_trained_model/ \
            --config_name models/${dataset}/adv_trained_model/ \
            --tokenizer_name models/${dataset}/adv_trained_model/ \
            --attack_file ${ADV_TRAIN_DIR}/lstm_${dataset}_${attacker}.csv \
            --path_to_save ${ADV_TRAIN_DIR_RESULT}/lstm_${dataset}_${attacker}_${num_examples}.csv
            --batch_size ${batch_size} \
            --num_labels ${num_labels}
        
        echo "Test ${attacker} end" >> ${logger_name}.txt
    done
done

# 3. create a table with metrics on the valid/test splits (save to ./results)
## [we check that the performance hasn't changed]
# 4. calculate attack metrics on the re-trained classifiers (save table to ./results)
