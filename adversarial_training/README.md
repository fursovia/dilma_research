## Adversarial Training experiments

## Prepare folder for experiment
'''console
kek
'''

## Model

* roberta-base


# Script for Adversarial training

```bash
bash bin/train.sh ./configs/models/clf_gru.jsonnet ./data/sst2
bash bin/train.sh ./configs/models/clf_gru.jsonnet ./data/rotten_tomatoes
bash bin/train.sh ./configs/models/clf_gru.jsonnet ./data/ag_news
```
