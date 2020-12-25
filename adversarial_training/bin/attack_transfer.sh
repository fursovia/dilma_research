#!/usr/bin/env bash

num_examples=1000
attack_num_examples=1000
batch_size=64
logging_steps=75

dataset=$1
logger_name=$2

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


for attacker_under in "deepwordbug" "textbugger" "pwws"; do
    echo "Start Attack from ${attacker_under} on initial model" >> ${logger_name}.txt
    textattack attack \
        --dataset-from-file datasets/${dataset}/data/load_${dataset}_${split_name_test}.py \
        --log-to-csv datasets/${dataset}/attacks/${attacker_from}_0.csv \
        --recipe ${attacker} \
        --num-examples ${attack_num_examples} \
        --model-from-file datasets/${dataset}/model/load_${dataset}_model.py \
        --disable-stdout
    echo "END Attack from ${attacker_under} on initial model" >> ${logger_name}.txt
done

for num_examples in 100 200 500 1000 1500 2000 3000 5000 10000; do
    echo "================================" >> ${logger_name}.txt
    for attacker_from in "deepwordbug" "textbugger" "pwws"; do
        echo "Additional Train ${attacker_from} ${num_examples} examples start" >> ${logger_name}.txt
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
            --adversarial_data_path datasets/${dataset}/attacks/${attacker_from}_${validation_split_name}_max_number.csv \
            --validation_split_name ${split_name_validation} \
            --adversarial_training_original_data_amount ${num_examples} \
            --adversarial_training_perturbed_data_amount ${num_examples} \
            --output_dir datasets/${dataset}/adv_trained_model/

        echo "Additional Train ${attacker_from} end" >> ${logger_name}.txt

        for attacker_under in "deepwordbug" "textbugger" "pwws"; do
            echo "Begin attack from ${attacker_from} to ${attacker_under} start" >> ${logger_name}.txt

            textattack attack \
                --dataset-from-file datasets/${dataset}/data/load_${dataset}_${split_name_test}.py \
                --log-to-csv datasets/${dataset}/attacks/from_${attacker_from}_to_${attacker_under}_${num_examples}.csv \
                --recipe ${attacker} \
                --num-examples ${attack_num_examples} \
                --model-from-file datasets/${dataset}/adv_trained_model/load_${dataset}_model.py \
                --disable-stdout

            echo "End attack from ${attacker_from} to ${attacker_under} start" >> ${logger_name}.txt

        done
    done
done