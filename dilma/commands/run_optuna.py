from functools import partial
from pathlib import Path
import json

import typer
import optuna

from dilma.commands.attack import attack
from dilma.commands.evaluate import main as evaluate


BASIC_CONFIG = {
  "attacker": {
    "type": "dilma",
    "archive_path": None,
    "bert_name_or_path": "bert-base-uncased",
    "deeplev_archive_path": None,
    "beta": 0.0,
    "num_steps": 30,
    "lr": 0.001,
    "num_gumbel_samples": 1,
    "tau": 1.0,
    "num_samples": 10,
    "temperature": 1.0,
    "add_mask": True,
    "device": 0
  }
}


def set_trial(trial: optuna.Trial):
    deeplev_archive_path = trial.suggest_categorical(
        "deeplev_archive_path", ["null", "./presets/models/deeplev.tar.gz"]
    )
    if deeplev_archive_path != "null":
        beta = trial.suggest_float("beta", 0.01, 10.0)
    else:
        beta = 0.0
        deeplev_archive_path = None

    num_steps = trial.suggest_int("num_steps", 10, 50)
    lr = trial.suggest_float("lr", 0.001, 1.0)
    tau = trial.suggest_float("tau", 0.1, 5.0)
    num_samples = trial.suggest_int("num_samples", 10, 50)
    temperature = trial.suggest_float("temperature", 0.1, 5.0)
    add_mask = trial.suggest_categorical("add_mask", [True, False])

    config = {
        "deeplev_archive_path": deeplev_archive_path,
        "beta": beta,
        "num_steps": num_steps,
        "lr": lr,
        "tau": tau,
        "num_samples": num_samples,
        "temperature": temperature,
        "add_mask": add_mask
    }
    return config


def get_objective(
        trial: optuna.Trial,
        serialization_dir: str,
        data_path: str,
        num_samples: int,
        dataset_name: str
) -> float:
    curr_config = set_trial(trial)
    BASIC_CONFIG['attacker'].update(curr_config)
    config = BASIC_CONFIG
    config['attacker']['archive_path'] = f"./presets/models/{dataset_name}.tar.gz"

    results_dir = Path(serialization_dir) / f"{trial.number}"
    config_path = str(results_dir / "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    attack(config_path, data_path=data_path, out_dir=str(results_dir), samples=num_samples)
    evaluate(
        output_path=str(results_dir / "data.json"),
        save_to=str(results_dir / "metrics.json"),
        target_clf_path=f"./presets/transformer_models/{dataset_name}"
    )

    with open(str(results_dir / "metrics.json")) as f:
        metrics = json.load(f)
    nad = metrics['NAD']
    return nad


def main(
        serialization_dir: str,
        data_path: str,
        num_samples: str,
        dataset_name: str,
        num_trials: int = 100,
        n_jobs: int = 1,
        timeout: int = 60 * 60 * 24,
        study_name: str = "optuna_dilma"
):
    study = optuna.create_study(
        # storage="sqlite:///result/dilma_attacker.db",
        sampler=optuna.samplers.TPESampler(seed=245),
        study_name=study_name,
        direction="maximize",
        load_if_exists=True,
    )

    objective = partial(
        get_objective,
        serialization_dir=serialization_dir,
        data_path=data_path,
        num_samples=num_samples,
        dataset_name=dataset_name
    )
    study.optimize(
        objective,
        n_jobs=n_jobs,
        n_trials=num_trials,
        timeout=timeout,
    )


if __name__ == "__main__":
    typer.run(main)