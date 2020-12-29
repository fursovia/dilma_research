#!/usr/bin/env bash

num_examples=1000
attack_num_examples=500
batch_size=64
logging_steps=75

dataset=$1
logger_name=$2

echo "================================" >> ${logger_name}.txt
echo "dataset ${dataset}" >> ${logger_name}.txt

if [ $dataset == "news" ]; then
validation_split_name='test'
else
validation_split_name='validation'
fi

if [ $dataset == "sst2" ]; then
test_split_name='validation'
else
test_split_name='test'
fi

echo "split_name_validation ${split_name_validation}" >> ${logger_name}.txt
echo "test_split_name ${test_split_name}" >> ${logger_name}.txt


for attacker in "deepwordbug" "textbugger" "pwws"; do
    echo "Start Attack from ${attacker_under} on initial model" >> ${logger_name}.txt
    textattack attack \
        --dataset-from-file datasets/${dataset}/data/load_${dataset}_${test_split_name}.py \
        --log-to-csv datasets/${dataset}/attacks/${attacker}_0.csv \
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
            --num_train_epochs 10 \
            --save_steps -1 \
            --evaluation_strategy "epoch" \
            --evaluate_during_training \
            --output_dir datasets/${dataset}/adv_trained_model/ \
            --adversarial_data_path datasets/${dataset}/attacks/${attacker_from}_${validation_split_name}_max_number.csv \
            --adversarial_training_original_data_amount ${num_examples} \
            --adversarial_training_perturbed_data_amount ${num_examples} 
            --use_custom_trainer \
            --stat_file_for_saving stat_${attacker}_${num_examples}.json

        echo "Additional Train ${attacker_from} ${num_examples} examples end" >> ${logger_name}.txt

        for attacker_under in "deepwordbug" "textbugger" "pwws"; do
            echo "Begin attack from ${attacker_from} to ${attacker_under} start" >> ${logger_name}.txt

            textattack attack \
                --dataset-from-file datasets/${dataset}/data/load_${dataset}_${test_split_name}.py \
                --log-to-csv datasets/${dataset}/attacks/from_${attacker_from}_to_${attacker_under}_${num_examples}.csv \
                --recipe ${attacker_under} \
                --num-examples ${attack_num_examples} \
                --model-from-file datasets/${dataset}/adv_trained_model/load_${dataset}_model.py \
                --disable-stdout

            echo "End attack from ${attacker_from} to ${attacker_under} start" >> ${logger_name}.txt

        done
    done
done