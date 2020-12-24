from pathlib import Path
import argparse
import pandas as pd
import tqdm
import re
import numpy as np
from sklearn.metrics import accuracy_score
from nltk.tokenize.treebank import TreebankWordTokenizer
from src.metrics import (stanza_metrics,
                         dist_k, ent_k,
                         bert_score_count,
                         calculate_wer,
                         normalized_accuracy_drop)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--folder_path',
        type=str,
        default='datasets/tomato/attacks')
    parser.add_argument('--save_path', type=str, default='datasets/tomato')
    args = parser.parse_args()
    return args


def metrics_file(file_path, before_or_after = None):
    '''
    read output in text-attack-format
    '''
    name = str(file_path).split('/')[-1][:-4]
    attack_name = name.split('_')[0]
    if before_or_after is None:
        before_or_after = 'after' if str(file_path).endswith('2.csv') else 'before'
    data = pd.read_csv(file_path)
    y_true = [int(i) for i in data['ground_truth_output'].tolist()]
    y_pred = [int(i) for i in data['original_output'].tolist()]
    y_adv = [int(i) for i in data['perturbed_output'].tolist()]
    original_text =  [re.sub("(\[\[)|(\]\])", "", i) for i in data['original_text'].tolist()]
    perturbed_text =  [re.sub("(\[\[)|(\]\])", "", i) for i in data['perturbed_text'].tolist()]
    
    wers = [calculate_wer(i, j) for i,j in list(zip(original_text, perturbed_text))]
#     mean_prob_diff=float(np.mean(prob_diffs)),
    mean_wer=float(np.mean(wers)),
    nad = normalized_accuracy_drop(y_true, y_adv, wers)
    bert_score = bert_score_count(original_text, perturbed_text)
    accuracy = accuracy_score(y_true, y_pred)
    accuracy_under_attack = accuracy_score(y_true, y_adv)
    
    k_range = [1, 2, 4]
    k_dist = {'original': dict(), 'perturbed': dict()}
    k_ent = {'original': dict(), 'perturbed': dict()}
    tokenizer = TreebankWordTokenizer()
    for data_name, data in list(zip(['original', 'perturbed'], [
        original_text, perturbed_text])):
        for k in k_range:
            k_dist[data_name][k] = dist_k(data, tokenizer, k)
            k_ent[data_name][k] = ent_k(data, tokenizer, k)
            
#     stanza_metric = stanza_metrics(original_text, perturbed_text)
    
    
    return {'attack': attack_name, 
            'before_or_after': before_or_after,
            'NAD': nad,
            'Mean_wer': np.mean(wers),
            'bertscore': bert_score['mean'],
#             'grammar': stanza_metric,
            'acc': {"original": accuracy, "perturbed": accuracy_under_attack},
            'k_dist': k_dist,
            'k_ent': k_ent}

def prepare_output(dict_of_metrics):
    res = dict()
    for i, j in dict_of_metrics.items():
        if not isinstance(j, dict):
            res[i] = j
        elif i == 'bertscore':
            res['bertscore'] = np.round(j['F1'], 3)
        elif i == 'acc':
            for name, metric in j.items():
                res[f"acc_{name}"] = np.round(metric, 3)
        elif i in ['k_dist', 'k_ent']:
            for name, metric in j.items():
                res[f"{i}_{name}"] = ' '.join(
                    [f"{ngram}: {np.round(count, 3)}" for ngram, count in metric.items()])
    return res


def metrics_folder(folder_path, save_path = 'metrics.csv'):
    result = list()
    paths = Path(folder_path).glob('*.csv')
    for path in tqdm.tqdm(paths):
        if  str(path).endswith('1.csv') or  str(path).endswith('2.csv'):
            result.append(prepare_output(metrics_file(path)))
    df = pd.DataFrame(result)
    df.reset_index()
    df.to_csv(save_path, index=False)


if __name__ == '__main__':
    args = parse_arguments()
    metrics_folder(args.folder_path, args.save_path)
