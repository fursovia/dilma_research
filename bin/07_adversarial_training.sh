#!/usr/bin/env bash

# 0. for each text classification dataset and for each attack
# 1. create dataset: take adversarial examples
## from 04, 05 steps and add them to the train split
# 2. re-train target classifiers [BERT]
# 3. create a table with metrics on the valid/test splits (save to ./results)
## [we check that the performance hasn't changed]
# 4. calculate attack metrics on the re-trained classifiers (save table to ./results)