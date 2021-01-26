from pathlib import Path
import json
import argparse
import os
import torch
import pyarrow as pa
import numpy as np
import tqdm
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from sklearn.metrics import accuracy_score

from transformers import (AutoTokenizer,
                          AutoConfig,
                          AutoModelForSequenceClassification,
                          HfArgumentParser,
                          default_data_collator
                          )
from torch.utils.data import DataLoader
from datasets import Dataset

from dilma.utils.data import clear_texts


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.com/models"},
        default='roberta-base'
    )
    config_name: Optional[str] = field(
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"},
        default='roberta-base'
    )
    tokenizer_name: Optional[str] = field(
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"},
        default='roberta-base'
    )


@dataclass
class OtherArguments:
    """
    Some other Arguments
    """
    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    batch_size: Optional[int] = field(
        default=16,
        metadata={
            "help": "Batch size for DataLoader sampler"
        },
    )
    pad_to_max_length: bool = field(
        default=True,
    )
    device: str = field(
        metadata={"help": "Train using Cuda or CPU"},
        default='cuda',
    )
    attack_file: str = field(
        metadata={"help": "path to output of textattack"},
        default=None,
    )
    path_to_save: str = field(
        metadata={"help": "name for statistic of training saving"},
        default=None,
    )
    num_labels: int = field(
        metadata={
            "help": "number of labels in dataset"},
        default=2,
    )
    verbose: bool = field(
        default=True,
    )


def create_dataloader(texts, labels, tokenizer,
                      batch_size: int = 16) -> DataLoader:
    encodings = tokenizer(texts,
                          truncation=True,
                          padding='max_length',
                          max_length=128)
    encodings['label'] = labels
    dataset = Dataset(pa.Table.from_pydict(encodings))
    dataset.set_format(
        type='torch',
        columns=[
            'input_ids',
            'attention_mask',
            'label'])
    return DataLoader(dataset,
                      batch_size=batch_size,
                      collate_fn=default_data_collator
                      )


def run_validation(model, batcher, device):
    soft = torch.nn.Softmax(1)
    predictions = list()
    scores = list()
    for batch in tqdm.tqdm(batcher):
        output = model(
            batch['input_ids'].to(device),
            batch['attention_mask'].to(device))
        probs = soft(output[0])
        predictions.extend(probs.max(1).indices.tolist())
        scores.extend([i.tolist()[batch['labels'][i_id]]
                       for i_id, i in enumerate(probs)])
    return predictions, scores


def main():

    parser = HfArgumentParser(
        (ModelArguments,
         OtherArguments))
    model_args, other_args = parser.parse_args_into_dataclasses()

    device = torch.device(other_args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)
    attack_file = pd.read_csv(other_args.attack_file)
    original_text = clear_texts(attack_file['original_text'].tolist())
    perturbed_text = clear_texts(attack_file['perturbed_text'].tolist())
    labels = [int(i) for i in attack_file['ground_truth_output']]
    original_dataloader = create_dataloader(
        original_text, labels, tokenizer, other_args.batch_size)
    perturbed_dataloader = create_dataloader(
        perturbed_text, labels, tokenizer, other_args.batch_size)

    config = AutoConfig.from_pretrained(
        model_args.config_name, num_labels=other_args.num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, config=config)
    model.eval()
    model.to(device)

    original_pred, original_score = run_validation(
        model, original_dataloader, device)
    perturbed_pred, perturbed_score = run_validation(
        model, perturbed_dataloader, device)
    attack_file['original_output'] = original_pred
    attack_file['original_score'] = original_score
    attack_file['perturbed_output'] = perturbed_pred
    attack_file['perturbed_score'] = perturbed_score

    if other_args.path_to_save is not None:
        attack_file.to_csv(other_args.path_to_save)

    if other_args.verbose:
        print(f"Original accuracy: {accuracy_score(labels, original_pred)}")
        print(f"Perturbed accuracy: {accuracy_score(labels, perturbed_pred)}")


if __name__ == '__main__':
    main()
