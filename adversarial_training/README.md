## Adversarial Training experiments



### Prepare folder for experiment

Script for preparing folder structure for adv.training experiments. Files can be loaded or from Huggingface datasets or from json file with splitted dataset.

```console
!python src/save_dataset.py \
--task_name tomato \
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
        ├── data(if dataset not in huggingface)
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

1. Model is trained on Train part of dataset
2. Model is attacked 
3. Model is finetuned on successfull examples from the attack
4. Attack is repeated

```bash
bash bin/run_adv_training.sh dataset_name logging_file_name
```

1. Script run adversarial attack, use n perturbed examples
2. Model is finetuned on successfull examples
3. Attack is repeated

```bash
bash bin/adv_training_number_experiment.sh number_of_examples_to_attack logging_file_name
```

Iteration over all attacks
1. Script run adversarial attack_1, use n perturbed examples
2. Model is finetuned on successfull examples
3. We use attack_2 on model, obtained from adversarial training on perturbed data from attack_1

```bash
bash bin/attack_transfer.sh dataset_name logging_file_name
```


## Requirements

1. transformers 3.3.0
2. datasets 1.1.3
3. textattack