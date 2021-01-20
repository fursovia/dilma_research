#!/usr/bin/env bash

# 1. for each dataset
## attack first K examples of the test (!) split (not validation) using `textattack` package
## and two attacks (fgsm and textfool) from our repository
# 2. covert output files from textattack to our format (save to ./results folder) [!]
# 3. evaluate attacks using `dilma/commands/evaluate.py` script
## (in white-box/black-box scenario. SOTA models to fool)
# 4. save table with metrics to ./results folder
# 5*. attack M examples of the train (!) set (will be needed for adversarial training and detection)