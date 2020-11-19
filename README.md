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


# Train Models


bash bin/train.sh ./configs/models/clf_gru.jsonnet ./data/sst2
bash bin/train.sh ./configs/models/clf_gru.jsonnet ./data/rotten_tomatoes
bash bin/train.sh ./configs/models/clf_gru.jsonnet ./data/ag_news


# Attack models

CUDA_VISIBLE_DEVICES="3" \
    CLF_PATH="./presets/models/sst2.tar.gz" \
    DATA_PATH="./data/sst2/valid.json" \
    python dilma/commands/attack.py ./configs/attacks/dilma.jsonnet --samples 500