from typing import List, Dict, Any, Sequence
import re

import jsonlines


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w0-9 ]+", "", text)
    text = re.sub(r"\s\s+", " ", text).strip()
    return text


def pad_punctuation(string: str) -> str:
    string = re.sub('([.,!?()])', r' \1 ', string)
    string = re.sub('\s{2,}', ' ', string)
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
