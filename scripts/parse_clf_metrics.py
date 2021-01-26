from pathlib import Path
import json

import typer

app = typer.Typer()


@app.command()
def main(logdir: Path, output_path: Path):

    # textattack
    metrics = {}
    for dirname in logdir.iterdir():
        with open(dirname / "lstm/log.txt") as f:
            data = f.readlines()

        the_accuracy_line = list(filter(lambda x: x.startswith("Saved model accuracy"), data))[-1]
        accuracy = float(the_accuracy_line.split()[-1].strip().replace("%", ""))
        metrics[dirname.name] = accuracy

    with open(output_path / "textattack_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)

    # allennlp
    ## do something

    # transformers
    ## do something


if __name__ == "__main__":
    app()
