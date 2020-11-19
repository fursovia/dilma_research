# DILMA [research repository]


## Competitors

* **hotflip**: Beam search and gradient-based word swap (["HotFlip: White-Box Adversarial Examples for Text Classification" (Ebrahimi et al., 2017)](https://arxiv.org/abs/1712.06751)).
* **deepwordbug**: Greedy replace-1 scoring and multi-transformation character-swap attack (["Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers" (Gao et al., 2018)](https://arxiv.org/abs/1801.04354)).
* **pwws**: Greedy attack with word importance ranking based on word saliency and synonym swap scores (["Generating Natural Language Adversarial Examples through Probability Weighted Word Saliency" (Ren et al., 2019)](https://www.aclweb.org/anthology/P19-1103/)).
* **textbugger**: Greedy attack with word importance ranking and a combination of synonym and character-based swaps ([(["TextBugger: Generating Adversarial Text Against Real-world Applications" (Li et al., 2018)](https://arxiv.org/abs/1812.05271)).
* etc.

## Dataset/tasks

* [glue^sst2](https://huggingface.co/nlp/viewer/?dataset=glue&config=sst2)
* [glue^qqp](https://huggingface.co/nlp/viewer/?dataset=glue&config=qqp)
* [glue^mnli](https://huggingface.co/nlp/viewer/?dataset=glue&config=mnli)
* [ag_news](https://huggingface.co/nlp/viewer/?dataset=ag_news)
* [rotten_tomatoes](https://huggingface.co/nlp/viewer/?dataset=rotten_tomatoes)


## Models

* lstm
* roberta-base


export TASK_NAME="sst2"

CUDA_VISIBLE_DEVICES="2" python train_bert.py \
  --model_name_or_path bert-base-uncased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir ./logs/$TASK_NAME/
  
  
  
DATA_DIR=$2
LOG_DIR=${3:-"null"}

export TF_FORCE_GPU_ALLOW_GROWTH=true

TRAIN_PATH=${DATA_DIR}/train.json
VALID_PATH=${DATA_DIR}/valid.json
TRAIN_DATA_PATH=${TRAIN_PATH} \
    VALID_DATA_PATH=${VALID_PATH} \
    allennlp train ${CONFIG_PATH} \
    --serialization-dir ${LOG_DIR} 
    
    
pip install torch==1.7.0+cu101-f https://download.pytorch.org/whl/torch_stable.html


215002-1811-sst2-clf_gru
215129-1811-ag_news-clf_gru
220128-1811-rotten_tomatoes-clf_gru