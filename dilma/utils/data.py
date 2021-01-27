from typing import List, Dict, Any, Sequence
import re

import jsonlines
import torch
from allennlp.data.vocabulary import Vocabulary


def clean_text(text: str) -> str:
    text = text.lower()
    # text = re.sub(r"[^\w0-9 ]+", "", text)
    text = re.sub(r"\s\s+", " ", text).strip()
    return text


def pad_punctuation(string: str) -> str:
    string = re.sub('([.,!?()])', r' \1 ', string)
    string = re.sub(r'\s{2,}', ' ', string)
    return string


def load_jsonlines(path: str) -> List[Dict[str, Any]]:
    data = []
    with jsonlines.open(path, "r") as reader:
        for items in reader:
            data.append(items)
    return data


def write_jsonlines(data: Sequence[Dict[str, Any]], path: str) -> None:
    with jsonlines.open(path, "w") as writer:
        for ex in data:
            writer.write(ex)


def clear_texts(list_of_texts: List[str]) -> List[str]:
    return [re.sub(r"(\[\[)|(\]\])", "", i).replace("Question1: ","").replace(">>>>Question2: "," ") for i in list_of_texts]




def decode_indexes(
    indexes: torch.Tensor, vocab: Vocabulary, namespace="transactions", drop_start_end: bool = True,
) -> List[str]:
    out = [vocab.get_token_from_index(idx.item(), namespace=namespace) for idx in indexes]

    if drop_start_end:
        return out[1:-1]

    return out
