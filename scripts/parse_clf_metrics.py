from pathlib import Path
import json

import typer

app = typer.Typer()


@app.command()
def main(logdir: Path =  Path('presets'), output_path: Path = Path('presets')):

    # textattack
    metrics = {}
    
    for dirname in Path(logdir,'textattack_models').iterdir():
        try:
            with open(Path(dirname, 'log.txt')) as f:
                    data = f.readlines()

            the_accuracy_line = list(filter(lambda x: x.startswith("Saved model accuracy"), data))[-1]
            accuracy = float(the_accuracy_line.split()[-1].strip().replace("%", ""))
            metrics['substitute_lstm_' + dirname.name] = accuracy
        except:
            continue
    
    # allennlp
    for dirname in Path(logdir,'allennlp_models').iterdir():
        try:
            stat = json.load(Path(dirname, 'metrics.json').open('r'))
            metrics['substitute_lstm_allennlp_' + dirname.name] = stat['best_validation_accuracy']
        except:
            continue
    
    # transformers
    for dirname in Path(logdir,'transformer_models').iterdir():
        try:
            with open(Path(dirname, 'eval_results_' + dirname.name + '.txt')) as f:
                    data = f.readlines()
            the_accuracy_line = list(filter(lambda x: x.startswith("eval_accuracy ="), data))[-1]
            accuracy = float(the_accuracy_line.split()[-1].strip().replace("\n", ""))
            metrics['target_roberta_' + dirname.name] = accuracy
        except:
            continue

    for dirname in Path(logdir,'transformer_substitute_models').iterdir():
        try:
            with open(Path(dirname, 'eval_results_' + dirname.name + '.txt')) as f:
                    data = f.readlines()
            the_accuracy_line = list(filter(lambda x: x.startswith("eval_accuracy ="), data))[-1]
            accuracy = float(the_accuracy_line.split()[-1].strip().replace("\n", ""))
            metrics['substitute_roberta_' + dirname.name] = accuracy
        except:
            continue

    with open(output_path / "validation_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    app()

