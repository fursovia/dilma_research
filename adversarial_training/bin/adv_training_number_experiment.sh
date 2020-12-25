#!/usr/bin/env bash

batch_size=64
logging_steps=75

dataset=$1
num_examples_maximum=$2
logger_name=$3

echo "================================" >> ${logger_name}.txt
echo "dataset ${dataset} ADVERSARIAL RETRAINING" >> ${logger_name}.txt

if [ $dataset == "news" ]; then
validation_split_name='test'
else
validation_split_name='validation'
fi

test_split_name='test'

for attacker in "deepwordbug" "textbugger" "pwws"; do
    echo "Attack ${attacker} start, ${num_examples_maximum} examples" >> ${logger_name}.txt
    textattack attack \
        --dataset-from-file datasets/${dataset}/data/load_${dataset}_${validation_split_name}.py \
        --log-to-csv datasets/${dataset}/attacks/${attacker}_${validation_split_name}_max_number.csv \
        --recipe ${attacker} \
        --num-examples ${num_examples_maximum} \
        --model-from-file datasets/${dataset}/model/load_${dataset}_model.py \
        --disable-stdout
    echo "Attack ${attacker} end" >> ${logger_name}.txt
done


#train on adversarial data
for attacker in "deepwordbug" "textbugger" "pwws"; do
    for num_examples in 100 200 500 1000 1500 2000 3000 5000 10000; do
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
            --num_train_epochs 10 \
            --save_steps -1 \
            --evaluation_strategy "epoch" \
            --evaluate_during_training \
            --adversarial_data_path datasets/${dataset}/attacks/${attacker}_${validation_split_name}_max_number.csv \
            --adversarial_training_original_data_amount ${num_examples} \
            --adversarial_training_perturbed_data_amount ${num_examples} \
            --output_dir datasets/${dataset}/adv_trained_model/

        echo "Train ${attacker} end num_examples ${num_examples}" >> ${logger_name}.txt
        echo "Attack ${attacker} start num_examples ${num_examples}" >> ${logger_name}.txt
        textattack attack \
            --dataset-from-file datasets/${dataset}/data/load_${dataset}_${test_split_name}.py \
            --log-to-csv datasets/${dataset}/attacks/${attacker}_${num_examples}.csv \
            --recipe ${attacker} \
            --num-examples 1000 \
            --model-from-file datasets/${dataset}/adv_trained_model/load_${dataset}_model.py \
            --disable-stdout
        echo "Attack ${attacker} end" >> ${logger_name}.txt
    done
done