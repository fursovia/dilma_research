import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import EvalPrediction
import random
import re
from typing import Sequence, Dict, Any, List


def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def get_data(path):
    text_file = open(path, "r")
    raw_lines = text_file.readlines()
    dataset = [(' '.join(i.split()[1:]), int(i.split()[0])) for i in raw_lines]
    return dataset

def get_adv_data(path):
    df = pd.read_csv(path)
    df = df[df['result_type'] == 'Successful']
    labels = [int(i) for i in df['ground_truth_output'].tolist()]
    texts = [re.sub("(\[\[)|(\]\])", "", i) for i in df['perturbed_text'].tolist()]
    return [(t, l) for t, l in list(zip(texts, labels))]

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
        
def compute_metrics(p: EvalPrediction):
    is_regression = False
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
    accuracy = (preds == p.label_ids).astype(np.float32).mean().item()
    return {"accuracy": accuracy}
