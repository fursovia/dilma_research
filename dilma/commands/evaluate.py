import json
import typer
import pandas as pd
import numpy as np
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
from sklearn.metrics import accuracy_score
from typing import List
import torch

from allennlp.predictors import TextClassifierPredictor
from transformers import (AutoTokenizer,
                          AutoConfig,
                          AutoModelForSequenceClassification,
                          )

from dilma.utils.data import load_jsonlines, clear_texts
from dilma.utils.metrics import (
    normalized_accuracy_drop,
    misclassification_error,
    probability_drop,
    bert_score,
    calculate_wer,
    ngam_statistics,
    stanza_metrics,
    sentence_bert
)

from scripts.eval_attack import create_dataloader, run_validation
import dilma

DATASET_NAME_TO_MODEL_NAME = {
    "sst2": "textattack/roberta-base-SST-2",
    "ag_news": "textattack/roberta-base-ag-news",
    "rotten_tomatoes": "textattack/roberta-base-rotten-tomatoes"
}
NDIGITS = 3


def get_predictor(archive_path: str) -> Predictor:
    archive = load_archive(archive_path, cuda_device=-1)
    predictor = Predictor.from_archive(
        archive=archive, predictor_name="text_classification")
    return predictor


def get_metrics(original_text: List[str] = None,
                perturbed_text: List[str] = None,
                y_pred=None,
                y_adv=None,
                y_true=None,
                probability=None,
                adversarial_probability=None,
                ):

    metrics = dict()
    if y_pred is not None:
        acc_original = accuracy_score(y_true, y_pred)
        metrics['acc_original'] = round(acc_original, NDIGITS)
        typer.echo(f"acc_original = {acc_original:.3f}")

    acc_perturbed = accuracy_score(y_true, y_adv)
    metrics['acc_perturbed'] = round(acc_perturbed, NDIGITS)
    typer.echo(f"acc_perturbed = {acc_perturbed:.3f}")

    wers = [calculate_wer(i, j)
            for i, j in list(zip(original_text, perturbed_text))]
    nad = normalized_accuracy_drop(wers=wers, y_true=y_true, y_adv=y_adv)
    metrics['NAD'] = round(nad, NDIGITS)
    metrics['Mean_WER'] = round(float(np.mean(wers)), NDIGITS)
    typer.echo(f"NAD = {nad:.3f}")
    typer.echo(f"Mean WER = {metrics['Mean_WER']:.3f}")

    if probability is not None:
        prob_drop = probability_drop(
            true_prob=probability,
            adv_prob=adversarial_probability)
        typer.echo(f"Probability drop = {prob_drop:.3f}")
        metrics['PD'] = round(prob_drop, NDIGITS)

    bert_score_result = bert_score(original_text, perturbed_text)
    typer.echo(f"Mean Bert Score = {bert_score_result:.3f}")
    metrics['BertScore'] = bert_score_result
    
    sentence_bert_cosine = sentence_bert(original_text, perturbed_text)
    typer.echo(f"Mean sbert cosine distance = {sentence_bert_cosine:.3f}")
    metrics['Sentence_Bert_cos'] = sentence_bert_cosine

    k_dist, k_ent = ngam_statistics(original_text, perturbed_text)
    typer.echo(f"k_dist : {k_dist}")
    typer.echo(f"k_ent : {k_ent}")
    metrics['k_dist'] = k_dist
    metrics['k_ent'] = k_ent

    stanza_metrics_results = stanza_metrics(original_text, perturbed_text)
    typer.echo(f"Counted_stanza_metrics")
    for metric, value in stanza_metrics_results.items():
        metrics[metric] = value

    return metrics


def main(
    output_path: str,
    save_to: str = typer.Option(None),
    target_clf_path: str = typer.Option(None),
    output_from_textattack: bool = typer.Option(
        False, "--output-from-textattack"),
    device: str = typer.Option('cuda'),
    batch_size: int = typer.Option(32),
    num_labels: int = typer.Option(2),
    allennlp_model: str = None,
):

    if output_from_textattack:
        output = pd.read_csv(output_path)
        output = output.sample(min(len(output), 1000))
        y_pred = [int(i) for i in output['original_output'].tolist()]
        y_adv = [int(i) for i in output['perturbed_output'].tolist()]
        y_true = [int(i) for i in output['ground_truth_output'].tolist()]
        original_text = clear_texts(output['original_text'].tolist())
        perturbed_text = clear_texts(output['perturbed_text'].tolist())
    else:
        output = load_jsonlines(output_path)
        y_true = [int(output[i]["data"]["label"]) for i in range(len(output))]
        y_adv = [int(output[i]["adversarial_data"]["label"])
                 for i in range(len(output))]
        original_text = [output[i]["data"]["text"] for i in range(len(output))]
        perturbed_text = [output[i]["adversarial_data"]["text"]
                          for i in range(len(output))]
        probability = [i['probability'] for i in output]
        adversarial_probability = [
            i['adversarial_probability'] for i in output]

    if target_clf_path is not None:
        device = torch.device(device)
        if target_clf_path in DATASET_NAME_TO_MODEL_NAME:
            target_clf_path = DATASET_NAME_TO_MODEL_NAME[target_clf_path]
        tokenizer = AutoTokenizer.from_pretrained(target_clf_path)
        original_dataloader = create_dataloader(
            original_text, y_true, tokenizer, batch_size)
        perturbed_dataloader = create_dataloader(
            perturbed_text, y_true, tokenizer, batch_size)

        config = AutoConfig.from_pretrained(
            target_clf_path, num_labels=num_labels)
        model = AutoModelForSequenceClassification.from_pretrained(
            target_clf_path, config=config)
        model.eval()
        model.to(device)

        y_pred, probability = run_validation(
            model, original_dataloader, device)
        y_adv, adversarial_probability = run_validation(
            model, perturbed_dataloader, device)
    else:
        if allennlp_model is not None:
            predictor = TextClassifierPredictor.from_path(allennlp_model)

            y_pred = []
            adversarial_probability = []
            for text in perturbed_text:
                p = predictor.predict(text)
                y_pred.append(int(p['label']))
                adversarial_probability.append(float(max(p['probs'])))

            probability = []
            for text in original_text:
                p = predictor.predict(text)
                probability.append(float(max(p['probs'])))
        else:
            y_pred = None
            probability = None  # np.zeros() if textattack
            adversarial_probability = None

    metrics = get_metrics(
        original_text,
        perturbed_text,
        y_pred,
        y_adv,
        y_true,
        probability,
        adversarial_probability
    )

    if save_to is not None:
        with open(save_to, "w") as f:
            json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    typer.run(main)

