from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import json
import os
import typer

app = typer.Typer()

def _make_directories(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def get_acc(stats, attack):
    s = [i for i in stats if i['attack'] == attack]
    xs = [i['number_of_adv_examples'] for i in s]
    ys = [i['acc_perturbed'] for i in s]
    ys = [y for _, y in sorted(zip(xs, ys))]
    xs = sorted(xs)
    return xs, ys

def round05(number):
    return (round(number * 20) / 20)

def find_nad(dataset, attack):
    try:
        metrics = json.load(Path(f'results/roberta_{dataset}_{attack}.json').open('r'))
    except:
        metrics = json.load(Path(f'results/lstm_{dataset}_{attack}.json').open('r'))
    return metrics['NAD']

@app.command()
def main(logdir: Path =  Path('adv_training_data'), output_path: Path = Path('adv_training_results')):
    
    _make_directories(output_path)
    #accuracy
    result = dict()
    for name in logdir.iterdir():
        if str(name).endswith('.json'):
            stat = json.load(Path(name).open('r'))
            name = str(name.name).split('.json')[0]
            if 'finetune' in name:
                name = name.split('_finetune')[0]
                method = 'finetune'
            else:
                name = name.split('_fromscratch')[0]
                method = 'from_scratch'
            attack = name.split('_')[0]
            number_of_adv_examples = int(name.split('_')[-1])
            dataset = '_'.join(name.split('_')[1:-1])
            stat['attack'] = attack
            stat['number_of_adv_examples'] = number_of_adv_examples

            if dataset not in result.keys():
                result[dataset] = {'finetune': list(), 'from_scratch': list()}
            result[dataset][method].append(stat)
    
    for dataset, stats in result.items():
        
        number_of_files = len(stats['finetune'] + stats['from_scratch'])
        if number_of_files > 0:
            
            minimum_perturbed_acc = min([i['acc_perturbed'] for i in stats['finetune']] 
                                        + [i['acc_perturbed'] for i in stats['from_scratch']])
            original_acc = np.mean(np.array([i['acc_original'] for i in stats['finetune']] 
                                            +[i['acc_original'] for i in stats['from_scratch']]))
            ticks = list(set([round05(i) for i in np.arange(minimum_perturbed_acc, original_acc + 0.03, 0.03)]))
            attacks = set([i['attack'] for i in stats['finetune']] + [i['attack'] for i in stats['from_scratch']])
            fig, axs = plt.subplots(1,len(attacks), figsize = (12, 4), squeeze=False)
            for i, attack in list(zip(np.arange(len(attacks)), list(attacks))):
                scratch_xs, scratch_ys = get_acc(stats['from_scratch'], attack)
                finetune_xs, finetune_ys = get_acc(stats['finetune'], attack)

                axs[0, i].plot(scratch_xs, scratch_ys, '-o', label='scratch_under_attack')
                axs[0, i].plot(finetune_xs, finetune_ys, '-o', label='finetune_under_attack')
                plt.yscale('log')
                plt.xscale('log')
                axs[0, i].set_yticks(ticks, minor=False)
                axs[0, i].set_yticklabels(ticks, fontdict=None, minor=False)
                axs[0, i].set_title(f"{attack}")
                axs[0, i].grid()
                axs[0, i].axhline(original_acc, color='r', label='without attack')

            lines_labels = [axs[0, i].get_legend_handles_labels()]
            lines, labels = [sum(a, []) for a in zip(*lines_labels)]
            fig.legend(lines, labels, 'right')

            fig.text(0.5, 1, dataset, va='center', ha='center', fontsize=18)
            fig.text(0.5, -0.04, 'Number of adversarial examples', va='center', ha='center', fontsize=18)
            fig.text(-0.03, 0.5, 'Accuracy under attack', va='center', ha='center', rotation='vertical', fontsize=18)
            plt.tight_layout()
            plt.savefig(f"{str(output_path)}/{dataset}.png")
    #nad
    nad = {'finetune': dict(), 'from_scratch': dict()}
    for name in logdir.iterdir():
        if str(name).endswith('.json'):
            stat = json.load(Path(name).open('r'))
            name = str(name.name).split('.json')[0]
            if 'finetune' in name:
                name = name.split('_finetune')[0]
                method = 'finetune'
            else:
                name = name.split('_fromscratch')[0]
                method = 'from_scratch'
            attack = name.split('_')[0]
            number_of_adv_examples = int(name.split('_')[-1])
            dataset = '_'.join(name.split('_')[1:-1])
            stat['attack'] = attack
            stat['number_of_adv_examples'] = number_of_adv_examples
            if dataset not in nad[method].keys():
                nad[method][dataset]  = dict()
            if attack not in nad[method][dataset].keys():
                nad[method][dataset][attack] = (find_nad(dataset, attack), stat['NAD'])
            else:
                smallest_nad = min(nad[method][dataset][attack][1], stat['NAD'])
                nad[method][dataset][attack] = (nad[method][dataset][attack][0], smallest_nad)
    pd.DataFrame(nad['finetune']).to_csv(f"{str(output_path)}/nad_finetune.csv")
    pd.DataFrame(nad['from_scratch']).to_csv(f"{str(output_path)}/nad_from_scratch.csv")
    
    

if __name__ == "__main__":
    app()

