import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import EvalPrediction
from datasets import load_dataset
import random
import re
import jsonlines
from typing import Sequence, Dict, Any, List


def clear_texts(list_of_texts: List[str]) -> List[str]:
    return [re.sub(r"(\[\[)|(\]\])", "", i) for i in list_of_texts]


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w0-9 ]+", "", text)
    text = re.sub(r"\s\s+", " ", text).strip()
    return text


def random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def get_data_from_file(path: str) -> List[Dict[str, Any]]:
    data = []
    with jsonlines.open(path, "r") as reader:
        for items in reader:
            data.append(items)
    if 'qqp' not in path:
        dataset = [(i['text'], int(i['label'])) for i in data]
    else:
        dataset = [( i['question1'] + '</s></s>' + i['question2'], int(i['label'])) for i in data]
    return dataset


def get_adv_data(path: str) -> List[Any]:
    df = pd.read_csv(path)
    df = df[df['result_type'] == 'Successful']
    labels = [int(i) for i in df['ground_truth_output'].tolist()]
    texts = [re.sub(r"(\[\[)|(\]\])", "", i)
             for i in df['perturbed_text'].tolist()]
    texts = [text.replace("Question1: "," ").replace(">>>>Question2: ","</s></s>") for text in texts]
    return [(t, l) for t, l in list(zip(texts, labels))]

def get_data_huggingface(data_args: Any, split: str = 'train') -> List[Any]:
    if data_args.huggingface_subset_name is None:
        data = load_dataset(data_args.huggingface_name)
    else:
        data = load_dataset(
            data_args.huggingface_name,
            data_args.huggingface_subset_name)
    return data


class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(p: EvalPrediction):
    is_regression = False
    preds = p.predictions[0] if isinstance(
        p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
    accuracy = (preds == p.label_ids).astype(np.float32).mean().item()
    return {"accuracy": accuracy}


task_to_keys = {
    "glue^cola": ("sentence", None),
    "glue^mnli": ("premise", "hypothesis"),
    "glue^mrpc": ("sentence1", "sentence2"),
    "glue^qnli": ("question", "sentence"),
    "glue^qqp": ("question1", "question2"),
    "glue^rte": ("sentence1", "sentence2"),
    "glue^sst2": ("sentence", None),
    "glue^stsb": ("sentence1", "sentence2"),
    "glue^wnli": ("sentence1", "sentence2"),
    "rotten_tomatoes": ("text", None),
    "ag_news": ("text", None),
    "dstc": ("text", None)
}
