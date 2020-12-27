from typing import List, Dict, Any, Sequence
from pathlib import Path
import json
import argparse
from datasets import load_dataset


def load_text_file(dataset_name: str,
                   split_name: str
                   ) -> str:
    text = [f"text_file = open('datasets/{dataset_name}/data/{split_name}.txt', 'r')",
            "raw_lines = text_file.readlines()",
            "dataset = [(' '.join(i.split()[1:]), int(i.split()[0])) for i in raw_lines]"
            ]
    return text


def load_model_file(dataset_name: str,
                    model_type: str,
                    number_of_labels: int
                    ) -> str:
    text = ["import torch",
            "import textattack",
            "from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification",
            f"tokenizer = AutoTokenizer.from_pretrained('datasets/{dataset_name}/{model_type}/')",
            f"config = AutoConfig.from_pretrained('datasets/{dataset_name}/{model_type}/', num_labels={number_of_labels})",
            f"model = AutoModelForSequenceClassification.from_pretrained('datasets/{dataset_name}/{model_type}/', config = config)",
            f"model = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer, 64)"
            ]
    return text


def save_dataset(dataset: Dict[str, List[Dict[str, Any]]],
                 path: str,
                 dataset_name: str,
                 data_name: str = 'text',
                 label_name: str = 'label'
                 ) -> None:
    '''
    required input:
        dataset: {split_name: [{'label': int, 'text': str}]}
        path: path to place, where all files will be located
        dataset_name: str
        data_name: str, name of data field in input dataset
        label_name: str, name of label field in input dataset
    '''
    Path(path, dataset_name).mkdir(parents=True, exist_ok=True)
    Path(path, dataset_name, 'attacks').mkdir(parents=True, exist_ok=True)
    Path(path, dataset_name, 'model').mkdir(parents=True, exist_ok=True)
    Path(
        path,
        dataset_name,
        'adv_trained_model').mkdir(
        parents=True,
        exist_ok=True)
    p = Path(path, dataset_name, 'data')
    p.mkdir(parents=True, exist_ok=True)

    for split_name in list(dataset.keys()):
        with open(f"{path}/{dataset_name}/data/load_{dataset_name}_{split_name}.py", "w") as file_handler:
            for item in load_text_file(dataset_name, split_name):
                file_handler.write("{}\n".format(item))

    for model_type in ['model', 'adv_trained_model']:
        with open(f"{path}/{dataset_name}/{model_type}/load_{dataset_name}_model.py", "w") as file_handler:
            for item in load_model_file(dataset_name, model_type, len(
                    set([i[label_name] for i in dataset[list(dataset.keys())[0]]]))):
                file_handler.write("{}\n".format(item))

    for key, utterances in dataset.items():
        f = open(str(Path(p, f"{key}.txt")), "w")
        for d_id, d in enumerate(utterances):
            f.write("%s %s" % (d[label_name], d[data_name]))
            if d_id < len(utterances) - 1:
                f.write('\n')
        f.close()


def get_dataset(args) -> Dict[str, List[Dict[str, Any]]]:
    if args.from_huggingface:
        if args.huggingface_subset_name is None:
            return load_dataset(args.huggingface_name)
        else:
            return load_dataset(args.huggingface_name,
                                args.huggingface_subset_name)
    else:
        return json.load(Path(args.file_path).open('r'))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument(
        '--from_huggingface',
        action='store_true',
        default=None)
    parser.add_argument('--huggingface_name', type=str, default=None)
    parser.add_argument('--huggingface_subset_name', type=str, default=None)
    parser.add_argument('--file_path', type=str, default=None)
    parser.add_argument('--path_to_save', type=str)
    parser.add_argument('--name_of_text_field', type=str, default='text')
    parser.add_argument('--name_of_label_field', type=str, default='label')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    dataset = get_dataset(args)
    save_dataset(dataset,
                 args.path_to_save,
                 args.dataset_name,
                 args.name_of_text_field,
                 args.name_of_label_field
                 )
