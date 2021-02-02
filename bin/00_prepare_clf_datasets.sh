#!/usr/bin/env bash

# 1. download datasets using huggingface
# 2. convert datasets to the format we use
# 3. save datasets to ./data

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1v3leUGIR7ObKqa7-2hqYQt2f1wVR-G5S' -O dstc.zip
unzip dstc.zip
rm dstc.zip

PYTHONPATH=. python scripts/create_classification_datasets.py --substitute-fraction 0.5 --dstc-path dstc.json
