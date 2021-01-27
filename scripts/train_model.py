from pathlib import Path
import shutil
import json
import argparse
import os
import torch
import random
import pyarrow as pa
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from transformers import (AutoTokenizer,
                          AutoConfig,
                          AutoModelForSequenceClassification,
                          Trainer,
                          HfArgumentParser,
                          TrainingArguments,
                          get_linear_schedule_with_warmup,
                          default_data_collator
                          )
from torch.utils.data import DataLoader
from datasets import Dataset, concatenate_datasets, load_from_disk
from custom_trainer import CustomTrainer

from utils import (random_seed,
                   get_data_from_file,
                   get_adv_data,
                   get_data_huggingface,
                   task_to_keys,
                   compute_metrics
                   )

ARGS_SPLIT_TOKEN = "^"


texts = ["asd", 'qwe', '12412', 'qweqew']
adv_texts = ["aaa", 'qwwe', '12312', 'qweqwe']  # some of them are unsuccessful


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        metadata={"help": "The name of the task to train on"},
    )
    from_huggingface: bool = field(
        metadata={"help": "use custom Trainer or not"},
        default=False,
    )
    huggingface_name: Optional[str] = field(
        metadata={"help": "The name of dataset to load from Huggingface"},
        default=None,
    )
    huggingface_subset_name: Optional[str] = field(
        metadata={
            "help": "The name of subset if dataset to load from Huggingface(if exists)"},
        default=None,
    )
    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
    )


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
    device: str = field(
        metadata={"help": "Train using Cuda or CPU"},
        default='cuda',
    )
    use_scheduler: bool = field(
        metadata={"help": "Use linear scheduler or not"},
        default=False,
    )
    adversarial_data_path: str = field(
        metadata={"help": "path to adv data"},
        default=None,
    )
    adversarial_training_original_data_amount: int = field(
        metadata={
            "help": "number of non-adversarial sentences to use during adv. training"},
        default=1000,
    )
    adversarial_training_perturbed_data_amount: int = field(
        metadata={
            "help": "number of adversarial sentences to use during adv. training"},
        default=1000,
    )
    validation_split_name: str = field(
        metadata={"help": "use validation or test split for testing model"},
        default=None,
    )
    use_custom_trainer: bool = field(
        metadata={"help": "use custom Trainer or not"},
        default=False,
    )
    stat_file_for_saving: str = field(
        metadata={"help": "name for statistic of training saving"},
        default=None,
    )
    optimizer_path: str = field(
        metadata={
            "help": "Path to load optimizer, if it already was used"},
        default=None,
    )
    use_early_stopping: bool = field(
        metadata={"help": "stop if val. loss increases during last n epochs"},
        default=False,
    )
    save_last: bool = field(
        metadata={"help": "stop if val. loss increases during last n epochs"},
        default=False,
    )
    substitute_train: bool = field(
        metadata={"help": "use substitute set of data or train"},
        default=False,
    )


def main():

    parser = HfArgumentParser(
        (ModelArguments,
         DataTrainingArguments,
         TrainingArguments,
         OtherArguments))
    model_args, data_args, training_args, other_args = parser.parse_args_into_dataclasses()
    training_args.disable_tqdm = True
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    device = torch.device(other_args.device)
    if training_args.seed is not None:
        random_seed(training_args.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)

    if other_args.validation_split_name is None:
        other_args.validation_split_name = 'validation' if data_args.huggingface_name != 'ag_news' else 'test'

    if data_args.from_huggingface:  # load dataset from huggingface
        data_args.task_name = f"{data_args.huggingface_name}{ARGS_SPLIT_TOKEN}{data_args.huggingface_subset_name}" if data_args.huggingface_subset_name is not None else data_args.huggingface_name

        def preprocess_function(examples):
            args = (
                (examples[sentence1_key],) if sentence2_key is None else (
                    examples[sentence1_key], examples[sentence2_key])
            )
            result = tokenizer(*args,
                               padding="max_length",
                               max_length=data_args.max_seq_length,
                               truncation=True)
            if "label" in examples:
                result["label"] = examples["label"]
            return result

        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
        data = get_data_huggingface(data_args)
        num_labels = len(set(data['train']['label']))

        data = data.map(preprocess_function, batched=True)
        data.set_format(
            type='torch',
            columns=[
                'input_ids',
                'attention_mask',
                'label'])
        train_dataset = data['train']
        validation_dataset = data[other_args.validation_split_name]

    else:  # load dataset from file
        if other_args.substitute_train:
            train_data = get_data_from_file(
                f"data/{data_args.task_name}/substitute_train.json")
            def get_load_model_file():
                text = ["import torch",
                        "import textattack",
                        "from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification",
                        "tokenizer = AutoTokenizer.from_pretrained('./presets/transformer_substitute_models/qqp/')",
                        "config = AutoConfig.from_pretrained('./presets/transformer_substitute_models/qqp/', num_labels=2)",
                        "model = AutoModelForSequenceClassification.from_pretrained('./presets/transformer_substitute_models/qqp/', config = config)",
                        "model = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer, 64)"
                       ]
                return text
            
            with open(f"{training_args.output_dir}/load_model.py", "w") as file_handler:
                for item in get_load_model_file():
                    file_handler.write("{}\n".format(item))
            
        else:
            train_data = get_data_from_file(
                f"data/{data_args.task_name}/train.json")
        num_labels = len(set([i[1] for i in train_data]))
        train_encodings = tokenizer([i[0] for i in train_data],
                                    truncation=True,
                                    padding='max_length',
                                    max_length=data_args.max_seq_length)
        train_encodings['label'] = [i[1] for i in train_data]
        train_dataset = Dataset(pa.Table.from_pydict(train_encodings))
        train_dataset.set_format(
            type='torch',
            columns=[
                'attention_mask',
                'input_ids',
                'label'])

        val_data = get_data_from_file(
            f"data/{data_args.task_name}/valid.json")
        val_encodings = tokenizer([i[0] for i in val_data],
                                  truncation=True,
                                  padding='max_length',
                                  max_length=data_args.max_seq_length)
        val_encodings['label'] = [i[1] for i in val_data]
        validation_dataset = Dataset(pa.Table.from_pydict(val_encodings))
        validation_dataset.set_format(
            type='torch', columns=[
                'attention_mask', 'input_ids', 'label'])

    validation_dataset = validation_dataset.select(
        random.sample(
            list(np.arange(len(validation_dataset))),
            min(5000, len(validation_dataset))
        )
    )

    if other_args.adversarial_data_path is not None:
        """
        Load perturbed data and concat it to subset of train dataset
        """

        columns_to_remove = [i for i in train_dataset.features if i not in ['input_ids',
                                                                            'attention_mask',
                                                                            'label']]
        train_dataset.remove_columns_(columns_to_remove)
        train_dataset = Dataset.from_pandas(
            pd.DataFrame(
                {i: train_dataset.data[i]
                    for i in train_dataset.data.column_names}
            )
        )
        random.seed(123)
        if other_args.adversarial_training_original_data_amount != -1:
            train_dataset = train_dataset.select(
                random.sample(
                    list(range(len(train_dataset))),
                    other_args.adversarial_training_original_data_amount
                )
            )
        train_dataset.save_to_disk('path_to_save_dataset1')
        train_dataset = load_from_disk('path_to_save_dataset1')

        adv_data = get_adv_data(other_args.adversarial_data_path)
        adv_encodings = tokenizer([i[0] for i in adv_data],
                                  truncation=True,
                                  padding='max_length',
                                  max_length=data_args.max_seq_length)
        adv_encodings['label'] = [i[1] for i in adv_data]
        adv_dataset = Dataset(pa.Table.from_pydict(adv_encodings))
        adv_dataset.set_format(
            type='torch',
            columns=[
                'input_ids',
                'attention_mask',
                'label'])
        adv_dataset = Dataset.from_pandas(
            pd.DataFrame(
                {i: adv_dataset.data[i]
                    for i in train_dataset.data.column_names}
            )
        )
        adv_dataset = adv_dataset.select(
            random.sample(
                list(np.arange(len(adv_dataset))),
                min(other_args.adversarial_training_perturbed_data_amount,
                    len(adv_dataset))
            )
        )

        adv_dataset.save_to_disk('path_to_save_dataset2')
        adv_dataset = load_from_disk('path_to_save_dataset2')

        train_dataset = concatenate_datasets([train_dataset,
                                              adv_dataset])

    train_dataset = train_dataset.shuffle()
    if other_args.adversarial_data_path is not None:
        shutil.rmtree('path_to_save_dataset1')
        shutil.rmtree('path_to_save_dataset2')
    print(len(train_dataset), len(validation_dataset))

    config = AutoConfig.from_pretrained(
        model_args.config_name, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, config=config)

    batches_in_epoch = len(train_dataset) // training_args.per_device_train_batch_size + \
        int(len(train_dataset) % training_args.per_device_train_batch_size > 0)
    training_args.max_steps = training_args.num_train_epochs * batches_in_epoch
    training_args.warmup_steps = batches_in_epoch
    training_args.eval_steps = batches_in_epoch
    training_args.save_steps = batches_in_epoch

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_args.learning_rate)

    if other_args.optimizer_path is not None:  # load optimizer
        if Path(other_args.optimizer_path).is_file():
            def optimizer_to(optim, device):
                for param in optim.state.values():
                    if isinstance(param, torch.Tensor):
                        param.data = param.data.to(device)
                        if param._grad is not None:
                            param._grad.data = param._grad.data.to(device)
                    elif isinstance(param, dict):
                        for subparam in param.values():
                            if isinstance(subparam, torch.Tensor):
                                subparam.data = subparam.data.to(device)
                                if subparam._grad is not None:
                                    subparam._grad.data = subparam._grad.data.to(
                                        device)
            optimizer.load_state_dict(torch.load(other_args.optimizer_path))
            optimizer_to(optimizer, device)

    if other_args.use_scheduler:
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=training_args.warmup_steps,
                                                    num_training_steps=training_args.max_steps)
    else:
        scheduler = None

    trainer_class = CustomTrainer if other_args.use_custom_trainer else Trainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        optimizers=(optimizer, scheduler),
        data_collator=default_data_collator,
    )
    if training_args.do_train:
        if other_args.use_custom_trainer:
            trainer.train(
                model_path=model_args.model_name_or_path if os.path.isdir(
                    model_args.model_name_or_path) else None,
                stat_file_for_saving=other_args.stat_file_for_saving,
                use_early_stopping=other_args.use_early_stopping
            )
        else:
            trainer.train(
                model_path=model_args.model_name_or_path if os.path.isdir(
                    model_args.model_name_or_path) else None,
            )
        if other_args.save_last:
            trainer.save_model()
#             torch.save(
#                 trainer.optimizer.state_dict(),
#                 os.path.join(training_args.output_dir,f"optimizer.pt")
#             )

    if training_args.do_eval:
        eval_result = trainer.evaluate(eval_dataset=validation_dataset)
        output_eval_file = os.path.join(
            training_args.output_dir,
            f"eval_results_{data_args.task_name}.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                for key, value in eval_result.items():
                    writer.write(f"{key} = {value}\n")


if __name__ == '__main__':
    main()


