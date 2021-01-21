import datasets
import typer

from dilma.utils.data import clean_text, write_jsonlines
from sklearn.model_selection import train_test_split


def get_train_valid(dataset, valid_split, namespace):
    train = []
    for text, lab in zip(dataset['train'][namespace], dataset['train']['label']):
        train.append({'text': clean_text(text), "label": lab})

    valid = []
    for text, lab in zip(dataset[valid_split][namespace], dataset[valid_split]['label']):
        valid.append({'text': clean_text(text), "label": lab})

    return train, valid


app = typer.Typer()


@app.command()
def main(substitute_fraction: float = 0.5, dstc_path: str = None):
    data = datasets.load_dataset('glue', 'sst2')
    train, valid = get_train_valid(data, 'validation', 'sentence')

    train_labels = [element['label'] for element in train]
    substitute_train, _ = train_test_split(
        train, train_size=substitute_fraction, stratify=train_labels, random_state=31
    )

    write_jsonlines(train, 'data/sst2/train.json')
    write_jsonlines(valid, 'data/sst2/valid.json')

    data = datasets.load_dataset('rotten_tomatoes')
    train, valid = get_train_valid(data, 'validation', 'text')
    write_jsonlines(train, 'data/rotten_tomatoes/train.json')
    write_jsonlines(valid, 'data/rotten_tomatoes/valid.json')

    data = datasets.load_dataset('ag_news')
    train, valid = get_train_valid(data, 'test', 'text')
    write_jsonlines(train, 'data/ag_news/train.json')
    write_jsonlines(valid, 'data/ag_news/valid.json')


if __name__ == "__main__":
    app()
