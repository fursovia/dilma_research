import datasets
import typer
import json
from pathlib import Path

from dilma.utils.data import clean_text, write_jsonlines
from sklearn.model_selection import train_test_split

task_to_keys = {
    "cola": ["sentence"],
    "mnli": ["premise", "hypothesis"],
    "mrpc": ["sentence1", "sentence2"],
    "qnli": ["question", "sentence"],
    "qqp": ["question1", "question2"],
    "rte": ["sentence1", "sentence2"],
    "sst2": ["sentence"],
    "stsb": ["sentence1", "sentence2"],
    "wnli": ["sentence1", "sentence2"],
    "rotten_tomatoes": ["text"],
    "ag_news": ["text"]
}


def get_train_valid(dataset, valid_split: str, namespace: list):
    train = []
    for i in dataset['train']:
        train.append({key: i[key] for key in namespace})
    valid = []
    for i in dataset[valid_split]:
        valid.append({key: i[key] for key in namespace})

    return train, valid


def dataset_from_huggingface(dataset_name: str = None,
                             dataset_configuration: str = None,
                             valid_split: str = None,
                             substitute_fraction: float = 0.5):
    if dataset_configuration is None:
        data = datasets.load_dataset(dataset_name)
        name = dataset_name
    else:
        data = datasets.load_dataset(dataset_name, dataset_configuration)
        name = dataset_configuration

    Path('data', name).mkdir(parents=True, exist_ok=True)
    namespace = task_to_keys[name] + ['label']

    train, valid = get_train_valid(data, valid_split, namespace)

    train_labels = [element['label'] for element in train]
    substitute_train, _ = train_test_split(
        train, train_size=substitute_fraction, stratify=train_labels, random_state=31
    )

    write_jsonlines(train, str(Path('data', name, 'train.json')))
    write_jsonlines(valid, str(Path('data', name, 'valid.json')))
    write_jsonlines(substitute_train, str(
        Path('data', name, 'substitute_train.json')))


def dstc_dataset(dstc_path: str,
                 substitute_fraction: float = 0.5):
    Path('data', 'dstc').mkdir(parents=True, exist_ok=True)
    dstc = json.load(Path(dstc_path).open('r'))
    X = [i['text'] for i in dstc]
    y = [i['intent'] for i in dstc]

    intent2idx = {
        intent_name: intent_id for intent_id,
        intent_name in enumerate(
            set(y))}
    y = [intent2idx[i] for i in y]
    X_train, X_dev, y_train, y_dev = train_test_split(
        X, y, test_size=0.25, random_state=31, stratify=y)
    dataset = dict()
    for name, (x, y) in zip(['train', 'validation'],
                            [(X_train, y_train), (X_dev, y_dev)]):
        d = list()
        for text, label in list(zip(x, y)):
            d.append({'text': text, 'label': label})
        dataset[name] = d

    train_labels = [element['label'] for element in dataset['train']]
    substitute_train, _ = train_test_split(
        dataset['train'], train_size=substitute_fraction, stratify=train_labels, random_state=31
    )

    write_jsonlines(dataset['train'], str(Path('data', 'dstc', 'train.json')))
    write_jsonlines(dataset['validation'], str(
        Path('data', 'dstc', 'valid.json')))
    write_jsonlines(substitute_train, str(
        Path('data', 'dstc', 'substitute_train.json')))


app = typer.Typer()


@app.command()
def main(substitute_fraction: float = 0.5, dstc_path: str = None):
    Path('data').mkdir(parents=True, exist_ok=True)

    dataset_from_huggingface('glue', 'sst2', valid_split='validation')
    dataset_from_huggingface('glue', 'qqp', valid_split='validation')
    dataset_from_huggingface('rotten_tomatoes', valid_split='validation')
    dataset_from_huggingface('ag_news', valid_split='test')
    dstc_dataset(dstc_path)


if __name__ == "__main__":
    app()
