#!/usr/bin/env bash

num_examples=1000
batch_size=64
logging_steps=75
logger_name='transfer_attack'
dataset=$1

echo "================================" >> ${logger_name}.txt
echo "dataset ${dataset}" >> ${logger_name}.txt

if [ $dataset == "tomato" ]; then
split_name_validation='validation'
split_name_test='test'
else
split_name_validation='validation'
split_name_test='test'
fi

echo "split_name_validation ${split_name_validation}" >> ${logger_name}.txt
echo "split_name_test ${split_name_test}" >> ${logger_name}.txt

for attacker in "deepwordbug" "textbugger" "pwws"; do
    echo "Initial Attack ${attacker} start" >> ${logger_name}.txt
    textattack attack \
        --dataset-from-file datasets/${dataset}/data/load_${dataset}_${split_name_validation}.py \
        --log-to-csv datasets/${dataset}/attacks/${attacker}_from_${attacker}.csv \
        --recipe ${attacker} \
        --num-examples ${num_examples} \
        --model-from-file datasets/${dataset}/model/load_${dataset}_model.py \
        --disable-stdout
    echo "Initial Attack ${attacker} end" >> ${logger_name}.txt
    done


for attacker_from in "deepwordbug" "textbugger" "pwws"; do
    echo "================================" >> ${logger_name}.txt
    echo "Additional Train ${attacker_from} start" >> ${logger_name}.txt
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
        --adversarial_data_path datasets/${dataset}/attacks/${attacker_from}_from_${attacker_from}.csv \
        --validation_split_name ${split_name_validation} \
        --output_dir datasets/${dataset}/adv_trained_model/

    echo "Additional Train ${attacker_from} end" >> ${logger_name}.txt

    for attacker_under in "deepwordbug" "textbugger" "pwws"; do
        echo "Begin attack from ${attacker_from} to ${attacker_under} start" >> ${logger_name}.txt

        textattack attack \
            --dataset-from-file datasets/${dataset}/data/load_${dataset}_${split_name_test}.py \
            --log-to-csv datasets/${dataset}/attacks/${attacker}_from_${attacker_from}_to${attacker_under}.csv \
            --recipe ${attacker} \
            --num-examples ${num_examples} \
            --model-from-file datasets/${dataset}/adv_trained_model/load_${dataset}_model.py \
            --disable-stdout

        echo "End attack from ${attacker_from} to ${attacker_under} start" >> ${logger_name}.txt

    done
done