# DILMA

To reproduce results from the paper

1. Install packages

```bash
poetry install
poetry shell
```

2. Run bash scripts

```bash
export CUDA_VISIBLE_DEVICES="1"

bash bin/00_prepare_clf_datasets.sh
bash bin/01_train_classifiers.sh 
bash bin/02_prepare_deep_lev_dataset.sh 
bash bin/03_train_deep_levenshtein.sh 
bash bin/04_baseline_attacks.sh 
bash bin/05_dilma_attacks.sh
bash bin/06_adversarial_detection.sh
bash bin/07_adversarial_training.sh
```