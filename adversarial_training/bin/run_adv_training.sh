#!/usr/bin/env bash

num_examples=1000
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

test_split_name='test'

echo "validation_split_name ${validation_split_name}" >> ${logger_name}.txt
echo "test_split_name ${test_split_name}" >> ${logger_name}.txt

if [ $dataset == "news" ]; then
train_epochs=2
elif [ $dataset == "tomato" ]; then
train_epochs=5
else
train_epochs=5
fi

echo "Training start" >> ${logger_name}.txt
python src/train_model.py \
    --model_name_or_path 'roberta-base' \
    --config_name 'roberta-base' \
    --tokenizer_name 'roberta-base' \
    --task_name ${dataset} \
    --do_train \
    --do_eval \
    --logging_steps ${logging_steps} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --learning_rate 2e-5 \
    --num_train_epochs ${train_epochs} \
    --save_steps -1 \
    --evaluation_strategy "epoch" \
    --evaluate_during_training \
    --use_scheduler "use" \
    --output_dir datasets/${dataset}/model/ \
    --validation_split_name ${validation_split_name}

echo "trained" >> ${logger_name}.txt

python src/train_model.py \
    --model_name_or_path datasets/${dataset}/model/ \
    --config_name datasets/${dataset}/model/ \
    --tokenizer_name datasets/${dataset}/model/ \
    --task_name ${dataset} \
    --do_eval \
    --per_device_eval_batch_size 64 \
    --output_dir datasets/${dataset}/model/ \
    --validation_split_name ${validation_split_name} 

echo "validated" >> ${logger_name}.txt

#attack model
for attacker in "deepwordbug" "textbugger" "pwws"; do
    echo "1 Attack ${attacker} start" >> ${logger_name}.txt
    textattack attack \
        --dataset-from-file datasets/${dataset}/data/load_${dataset}_${validation_split_name}.py \
        --log-to-csv datasets/${dataset}/attacks/${attacker}_1.csv \
        --recipe ${attacker} \
        --num-examples ${num_examples} \
        --model-from-file datasets/${dataset}/model/load_${dataset}_model.py \
        --disable-stdout
    echo "1 Attack ${attacker} end" >> ${logger_name}.txt
    done

#train on adversarial data
for attacker in "deepwordbug" "textbugger" "pwws"; do
    echo "Train ${attacker} start" >> ${logger_name}.txt
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
        --adversarial_data_path datasets/${dataset}/attacks/${attacker}_1.csv \
        --validation_split_name ${validation_split_name} \
        --output_dir datasets/${dataset}/adv_trained_model/

    echo "Train ${attacker} end" >> ${logger_name}.txt
    #attack tuned model
    echo "2 attack ${attacker} start" >> ${logger_name}.txt
    textattack attack \
        --dataset-from-file datasets/${dataset}/data/load_${dataset}_${test_split_name}.py \
        --log-to-csv datasets/${dataset}/attacks/${attacker}_2.csv \
        --recipe ${attacker} \
        --num-examples ${num_examples} \
        --model-from-file datasets/${dataset}/adv_trained_model/load_${dataset}_model.py \
        --disable-stdout
    echo "2 attack ${attacker} end" >> ${logger_name}.txt
    done

echo "metrics" >> ${logger_name}.txt
python src/compute_metrics.py \
    --folder_path datasets/${dataset}/attacks/ \
    --save_path datasets/${dataset}/metrics.csv