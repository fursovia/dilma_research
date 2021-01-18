import json

import typer
import pandas as pd
import numpy as np
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive

from dilma.utils.data import load_jsonlines
from dilma.utils.metrics import (
    normalized_accuracy_drop,
    misclassification_error,
    probability_drop,
)


DATASET_NAME_TO_MODEL_NAME = {
    "sst2": "bert-uncased-sst2",
}
NDIGITS = 3


def get_predictor(archive_path: str) -> Predictor:
    archive = load_archive(archive_path, cuda_device=-1)
    predictor = Predictor.from_archive(archive=archive, predictor_name="text_classification")
    return predictor


def get_metrics(output, y_true, y_adv):
    nad = normalized_accuracy_drop(wers=output["wer"], y_true=y_true, y_adv=y_adv)
    typer.echo(f"NAD = {nad:.2f}")

    misclf_error = misclassification_error(y_true=y_true, y_adv=y_adv)
    typer.echo(f"Misclassification Error = {misclf_error:.2f}")

    prob_drop = probability_drop(true_prob=output["probability"], adv_prob=output["adversarial_probability"])
    typer.echo(f"Probability drop = {prob_drop:.2f}")

    mean_wer = float(np.mean(output["wer"]))
    typer.echo(f"Mean WER = {mean_wer:.2f}")

    metrics = {
        "NAD": round(nad, NDIGITS),
        "ME": round(misclf_error, NDIGITS),
        "PD": round(prob_drop, NDIGITS),
        "Mean_WER": round(mean_wer, NDIGITS),
    }
    return metrics


def main(
    output_path: str, save_to: str = typer.Option(None), target_clf_path: str = typer.Option(None),
):
    output = load_jsonlines(output_path)
    output = pd.DataFrame(output).drop(columns="history")

    y_true = [output["data"][i]["label"] for i in range(len(output))]
    y_adv = [output["adversarial_data"][i]["label"] for i in range(len(output))]

    if target_clf_path is not None:
        # change probability
        # change predicted label
        pass

    metrics = get_metrics(output, y_true, y_adv)

    if save_to is not None:
        with open(save_to, "w") as f:
            json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    typer.run(main)
