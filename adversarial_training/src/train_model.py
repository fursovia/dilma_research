from pathlib import Path
import json
import argparse
import os
import torch
import random
from torch.utils.data import DataLoader
from utils import random_seed, get_data, CustomDataset, compute_metrics, get_adv_data
from dataclasses import dataclass, field
from typing import Optional
from transformers import (AutoTokenizer,
                          AutoConfig,
                          AutoModelForSequenceClassification,
                          Trainer,
                          HfArgumentParser,
                          TrainingArguments,
                          get_linear_schedule_with_warmup
                          )
from transformers import default_data_collator
from custom_trainer import CustomTrainer


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
            "help": "Path to pretrained model or model identifier from huggingface.co/models"},
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
    use_scheduler: str = field(
        metadata={"help": "Use linear scheduler or not"},
        default=None,
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


def main():

    parser = HfArgumentParser(
        (ModelArguments,
         DataTrainingArguments,
         TrainingArguments,
         OtherArguments))
    model_args, data_args, training_args, other_args = parser.parse_args_into_dataclasses()
    training_args.disable_tqdm = True

    assert data_args.task_name in [
        'tomato', 'sst2', 'news', 'dstc'], 'Unknown dataset'

    device = torch.device(other_args.device)
    if training_args.seed is not None:
        random_seed(training_args.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)

    if other_args.adversarial_data_path is None:
        train_data = get_data(f"datasets/{data_args.task_name}/data/train.txt")
        num_labels = len(sorted(set([i[1] for i in train_data])))
    else:
        train_data = get_data(f"datasets/{data_args.task_name}/data/train.txt")
        num_labels = len(sorted(set([i[1] for i in train_data])))
        train_data = random.sample(
            train_data, min(
                len(train_data), other_args.adversarial_training_original_data_amount))
        adv_data = get_adv_data(other_args.adversarial_data_path)
        adv_data = random.sample(
            adv_data, min(
                len(adv_data), other_args.adversarial_training_perturbed_data_amount))
        train_data = train_data + adv_data
        random.sample(train_data, len(train_data))

    random.sample(train_data, len(train_data))
    train_encodings = tokenizer([i[0] for i in train_data],
                                truncation=True,
                                padding='max_length',
                                max_length=data_args.max_seq_length)
    train_dataset = CustomDataset(train_encodings, [i[1] for i in train_data])

    if other_args.validation_split_name is None:
        other_args.validation_split_name = 'validation' if data_args.task_name != 'news' else 'test'
    test_data = get_data(
        f"datasets/{data_args.task_name}/data/{other_args.validation_split_name}.txt")
    random.sample(test_data, min(1000, len(test_data)))
    test_encodings = tokenizer([i[0] for i in test_data],
                               truncation=True,
                               padding='max_length',
                               max_length=data_args.max_seq_length)
    test_dataset = CustomDataset(test_encodings, [i[1] for i in test_data])

    config = AutoConfig.from_pretrained(
        model_args.config_name, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, config=config)

    batches_in_epoch = len(train_dataset) // training_args.per_device_train_batch_size + \
        int(len(train_dataset) % training_args.per_device_train_batch_size > 0)
    training_args.max_steps = training_args.num_train_epochs * batches_in_epoch
    training_args.warmup_steps = batches_in_epoch
    training_args.eval_steps = batches_in_epoch

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_args.learning_rate)

    if other_args.use_scheduler is not None:
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
        eval_dataset=test_dataset,
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
                stat_file_for_saving=other_args.stat_file_for_saving
            )
        else:
            trainer.train(
                model_path=model_args.model_name_or_path if os.path.isdir(
                    model_args.model_name_or_path) else None,
            )

        trainer.save_model()

    if training_args.do_eval:
        eval_result = trainer.evaluate(eval_dataset=test_dataset)
        output_eval_file = os.path.join(
            training_args.output_dir,
            f"eval_results_{data_args.task_name}.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                for key, value in eval_result.items():
                    writer.write(f"{key} = {value}\n")


if __name__ == '__main__':
    main()
