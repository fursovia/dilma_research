## Adversarial Training experiments



### Prepare folder for experiment

Script for preparing folder structure for adv.training experiments. Files can be loaded or from Huggingface datasets or from json file with splitted dataset.

```console
!python src/save_dataset.py \
--dataset_name tomato \
--huggingface_name rotten_tomatoes \
--from_huggingface \
--path_to_save datasets 
```

Resulting folder:

```bat
.
├── datasets
    ├── dataset1
    ├── ...
    └── tomato
        ├── model
        │   ├──load_tomato_model.py
        │   └──model files
        │
        ├── data
        │   ├──load_tomato_{split_name}.py
        │   └──{split_name}.txt
        │
        ├── model
        │   ├──load_tomato_model.py
        │   └──model files
        │
        ├── attacks
        │   └──attack output metrics
        │   
        └──adv_trained_model
            ├──load_tomato_model.py
            └──model files
```

## Model

* roberta-base (all models from Huggingface works)


# Script for Adversarial training
Script iterates through datasets, train model on Train part of dataset, attacks it. 
Then use perturbed data for adversarial training of model and measure attack again.

```bash
bash bin/run_adv_training.sh
```
