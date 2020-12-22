#!/usr/bin/env bash

num_examples=1000
batch_size=64
logging_steps=75
logger_name='logger_number_num_examples'

for dataset in "news"; do
    echo "================================" >> ${logger_name}.txt
    echo "dataset ${dataset} ADVERSARIAL RETRAINING" >> ${logger_name}.txt
    
    if [ $dataset == "news" ]; then
    split_name='test'
    else
    split_name='validation'
    fi
    
    #train on adversarial data
    for attacker in "deepwordbug" "textbugger" "pwws"; do
        for num_examples in 100 200 500 1000 1500 2000 3000; do
            echo "Train ${attacker} start; num_examples ${num_examples}" >> ${logger_name}.txt
            python src/train_model.py \
                --model_name_or_path datasets/${dataset}/model/ \
                --config_name datasets/${dataset}/model/ \
                --tokenizer_name datasets/${dataset}/model/ \
                --task_name ${dataset} \
                --do_train \
                --do_eval \
                --logging_steps ${logging_steps} \
                --per_device_train_batch_size ${batch_size} \
                --per_device_eval_batch_size ${batch_size} \
                --learning_rate 1e-5 \
                --num_train_epochs 10.0 \
                --save_steps -1 \
                --evaluation_strategy "epoch" \
                --evaluate_during_training \
                --adversarial_data_path datasets/${dataset}/attacks/${attacker}_3.csv \
                --adversarial_training_original_data_amount ${num_examples} \
                --adversarial_training_perturbed_data_amount ${num_examples} \
                --output_dir datasets/${dataset}/adv_trained_model/

            echo "Train ${attacker} end" >> ${logger_name}.txt
            #attack tuned model
            echo "2 attack ${attacker} start" >> ${logger_name}.txt
            textattack attack \
                --dataset-from-file datasets/${dataset}/data/load_${dataset}_${split_name}.py \
                --log-to-csv datasets/${dataset}/attacks/${attacker}_${num_examples}.csv \
                --recipe ${attacker} \
                --num-examples 1000 \
                --model-from-file datasets/${dataset}/adv_trained_model/load_${dataset}_model.py
            echo "2 attack ${attacker} end" >> ${logger_name}.txt
        done
    done
done