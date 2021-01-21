#!/usr/bin/env bash

# 1. download datasets using huggingface
# 2. covert datasets to the format we use
# 3. save datasets to ./data

wget my_link.com/data.zip
unzip ...

PYTHONPATH=. python scripts/create_classification_datasets.py $PATH


# добавить все датасеты в скрипт (+ substitute frac)
# переделать скрипт обучения textattack (с поддержкой файла)
# обучение lstm с помощью allennlp
# fgsm + textattack
# dilma attacks