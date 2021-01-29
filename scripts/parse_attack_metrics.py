from pathlib import Path
import json
import pandas as pd
import typer

app = typer.Typer()


@app.command()
def main(logdir: Path =  Path('results'), output_path: str = 'results/attacks_metrics.csv'):

    result = list()
    for dirname in Path(logdir).iterdir():
        if str(dirname).endswith('.json'):
            name = dirname.name[:-5]
            substitute_model = name.split('_')[0]
            attack = name.split('_')[-1]
            dataset = '_'.join(name.split('_')[1:-1])
            stat = json.load(Path(dirname).open('r'))
            current_stat = {'dataset': dataset, 'attack': attack, 
                            'substitute_model': substitute_model, 'attacking_model': 'roberta'}
            current_stat.update(stat)
            result.append(current_stat)
    pd.DataFrame(result).sort_values(by=['dataset', 'attack']).to_csv(output_path, index=False)


if __name__ == "__main__":
    app()

