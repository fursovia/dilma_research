#!/usr/bin/env bash

# 0. for each text classification dataset and each attack
# 1. create dataset: take adversarial examples
## from 04, 05 steps and corresponding original examples
# 2. train a LSTM classifier to detect adversarial-vs-non-adversarial
# 3. calculate ROC AUC, Accuracy metrics. Save table to ./results
