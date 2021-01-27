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


def get_load_text_file(task_name: str,
                       split_name: str
                       ) -> str:
    if 'qqp' not in split_name:
        text = ["import jsonlines",
                "dataset = []",
                f"with jsonlines.open('data/{task_name}/{split_name}.json', 'r') as reader:",
                "    for items in reader:",
                "        dataset.append(items)",
                "dataset = [(element['text'], element['label']) for element in dataset]"
                ]
    else:
        text = ["import jsonlines",
                "dataset = []",
                f"with jsonlines.open('data/{task_name}/{split_name}.json', 'r') as reader:",
                "    for items in reader:",
                "        dataset.append(items)",
                "dataset = [(element['question1'] + '</s></s>' + element['question2'], element['label']) for element in dataset]"
                ]
    return text


def save_file_for_dataset_loading(task_name: str,
                                  split_name: str
                                  ) -> str:
    with open(f"data/{task_name}/load_{split_name}.py", "w") as file_handler:
        for item in get_load_text_file(task_name, split_name):
            file_handler.write("{}\n".format(item))


def get_train_valid(dataset, valid_split: str, namespace: list):
    train = []
    for i in dataset['train']:
        train.append({('text' if key in ['text', 'sentence'] else key): (clean_text(
            i[key]) if isinstance(i[key], str) else i[key]) for key in namespace})
    valid = []
    for i in dataset[valid_split]:
        valid.append({('text' if key in ['text', 'sentence'] else key): (clean_text(
            i[key]) if isinstance(i[key], str) else i[key]) for key in namespace})

    return train, valid


def get_dataset(dataset_name: str = None,
                dataset_configuration: str = None,
                valid_split: str = None,
                substitute_fraction: float = 0.5,
                make_test_from_train: float = 0.0,
                random_state: int = 42
                ):

    from_huggingface = dataset_name in datasets.list_datasets()

    if from_huggingface:
        data = datasets.load_dataset(dataset_name, dataset_configuration)
        name = dataset_name if dataset_configuration is None else dataset_configuration
        namespace = task_to_keys[name] + ['label']
        train, valid = get_train_valid(data, valid_split, namespace)

        if name == 'rotten_tomatoes':
            test = []
            for i in data['test']:
                test.append({('text' if key in ['text', 'sentence'] else key): (clean_text(
                    i[key]) if isinstance(i[key], str) else i[key]) for key in namespace})
            write_jsonlines(test, str(Path('data', name, f"test.json")))
            save_file_for_dataset_loading(name, 'test')

    else:
        data = json.load(Path(dataset_name).open('r'))
        X = [clean_text(i['text']) for i in data]
        y = [i['intent'] for i in data]

        intent2idx = {
            intent_name: intent_id for intent_id,
            intent_name in enumerate(
                set(y))}
        y = [intent2idx[i] for i in y]
        X_train, X_dev, y_train, y_dev = train_test_split(
            X, y, test_size=0.25, random_state=random_state, stratify=y)
        dataset = dict()
        for name, (x, y) in zip(['train', 'validation'],
                                [(X_train, y_train), (X_dev, y_dev)]):
            d = list()
            for text, label in list(zip(x, y)):
                d.append({'text': text, 'label': label})
            dataset[name] = d
        train = dataset['train']
        valid = dataset['validation']
        name = 'dstc'

    Path('data', name).mkdir(parents=True, exist_ok=True)

    if make_test_from_train > 0:
        train_labels = [element['label'] for element in train]
        train, test = train_test_split(
            train, test_size=make_test_from_train, stratify=train_labels, random_state=random_state
        )
        write_jsonlines(test, str(Path('data', name, f"test.json")))
        save_file_for_dataset_loading(name, 'test')

    train_labels = [element['label'] for element in train]
    substitute_train, _ = train_test_split(
        train, train_size=substitute_fraction, stratify=train_labels, random_state=random_state
    )

    for data, split_name in list(zip([train, valid, substitute_train], [
                                 'train', 'valid', 'substitute_train'])):
        write_jsonlines(data, str(Path('data', name, f"{split_name}.json")))
        save_file_for_dataset_loading(name, split_name)


app = typer.Typer()


# TODO
# add test split (from what?)
# args.dataset_from_huggingface instead? (low priority)
# substitute_fraction from main

@app.command()
def main(substitute_fraction: float = 0.5,
         dstc_path: str = None, random_state=42):
    Path('data').mkdir(parents=True, exist_ok=True)

    get_dataset(
        dataset_name='glue',
        dataset_configuration='sst2',
        valid_split='validation',
        make_test_from_train=5000,
        random_state=random_state)
    get_dataset(
        dataset_name='rotten_tomatoes',
        valid_split='validation',
        make_test_from_train=0,
        random_state=random_state)
    get_dataset(
        dataset_name='ag_news',
        valid_split='test',
        make_test_from_train=5000,
        random_state=random_state)
    get_dataset(
        dataset_name='glue',
        dataset_configuration='qqp',
        valid_split='validation',
        make_test_from_train=5000,
        random_state=random_state)
    get_dataset(
        dataset_name=dstc_path,
        make_test_from_train=5000,
        random_state=random_state)


if __name__ == "__main__":
    app()
