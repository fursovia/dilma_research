from pathlib import Path
import json
import argparse
import random
import torch
from torch.utils.data import DataLoader
from utils_discriminator import (random_seed,
                                 get_train_test_data,
                                 CustomDataset,
                                 CustomTransform,
                                 train,
                                 evaluate
                                 )

from transformers import (AutoTokenizer,
                          AutoConfig,
                          AutoModelForSequenceClassification
                          )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='roberta-base')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--file_path', type=str, default='attack.json')
    parser.add_argument(
        '--result_path',
        type=str,
        default='dataset_attack_discriminator.json')
    parser.add_argument('--checkpoint_path', type=str, default='trained.pt')
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--n_epoch', type=int, default=10)
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def main(args):
    device = torch.device(args.device)
    if args.seed is not None:
        random_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    config = AutoConfig.from_pretrained(
        args.model_dir, num_labels=2)

    result = {'acc': list(), 'roc_auc': list()}

    for j in range(args.n_splits):
        print(f"{j + 1} SPLIT OUT OF {args.n_splits}")
        seed = random.randint(0, 10e6)
        train_data, test_data = get_train_test_data(
            data_path=args.file_path, test_size=args.test_size, random_state=seed)
        transform = CustomTransform(tokenizer, max_len=80)
        train_dataset = CustomDataset(train_data, transform=transform)
        test_dataset = CustomDataset(test_data, transform=transform)
        batcher = {'train': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True),
                   'dev': DataLoader(test_dataset, batch_size=args.batch_size)}

        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_dir, config=config).to(device)

        train(model, batcher, args)
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        current_res = evaluate(model, batcher, args)
        result['acc'].append(current_res['acc'])
        result['roc_auc'].append(current_res['roc_auc'])

        json.dump(result, Path(args.result_path).open('w'))
        del model, checkpoint, train_data, test_data, train_dataset, test_dataset, batcher


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
