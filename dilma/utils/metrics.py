# !pip install bert_score
# !pip install stanza
import functools
from typing import Sequence, List
from multiprocessing import Pool
from tqdm import tqdm

import numpy as np
import Levenshtein
from bert_score import score
from nltk import ngrams
from nltk.tokenize.treebank import TreebankWordTokenizer


@functools.lru_cache(maxsize=5000)
def calculate_wer(text_a: str, text_b: str) -> int:
    # taken from
    # https://github.com/SeanNaren/deepspeech.pytorch/blob/master/decoder.py
    b = set(text_a.split() + text_b.split())
    word2char = dict(zip(b, range(len(b))))

    w1 = [chr(word2char[w]) for w in text_a.split()]
    w2 = [chr(word2char[w]) for w in text_b.split()]

    return Levenshtein.distance(''.join(w1), ''.join(w2))


def _edit_distance_one_vs_all(x):
    a, bs = x
    return [calculate_wer(a, b) for b in bs]


def pairwise_edit_distances(
    sequences_a: Sequence[str], sequences_b: Sequence[str], n_jobs: int = 5, verbose: bool = False
) -> np.ndarray:
    bar = tqdm if verbose else lambda iterable, total, desc: iterable

    with Pool(n_jobs) as pool:
        distances = list(
            bar(
                pool.imap(
                    _edit_distance_one_vs_all, zip(
                        sequences_a, [
                            sequences_b for _ in sequences_a])),
                total=len(sequences_a),
                desc="# edit distance {}x{}".format(
                    len(sequences_a), len(sequences_b)),
            )
        )
    return np.array(distances)


def normalized_accuracy_drop(
        wers: List[int], y_true: List[int], y_adv: List[int], gamma: float = 1.0,) -> float:
    assert len(y_true) == len(y_adv)
    nads = []
    for wer, lab, alab in zip(wers, y_true, y_adv):
        if wer > 0 and lab != alab:
            nads.append(1 / wer ** gamma)
        else:
            nads.append(0.0)

    return sum(nads) / len(nads)


def misclassification_error(y_true: List[int], y_adv: List[int],) -> float:
    misses = []
    for lab, alab in zip(y_true, y_adv):
        misses.append(float(lab != alab))

    return sum(misses) / len(misses)


def probability_drop(true_prob: List[float], adv_prob: List[float],) -> float:
    prob_diffs = []
    for tp, ap in zip(true_prob, adv_prob):
        prob_diffs.append(tp - ap)

    return sum(prob_diffs) / len(prob_diffs)


def bert_score(sequences: List[str], adversarial_sequences: List[str]):
    '''
    return:
            P, R, F1 from BertScore(https://arxiv.org/pdf/1904.09675.pdf)
    '''
    P, R, F1 = score(sequences, adversarial_sequences, lang="en",
                     verbose=False, rescale_with_baseline=True)
    bertscore = {
        'P': P.mean().item(),
        'R': R.mean().item(),
        'F1': F1.mean().item()}
    return bertscore['F1']


def dist_k(utts: List[str], tokenizer, k: int) -> float:
    '''
    return:
            number of unique n-grams, divided by number of n-grams
    '''
    total_n_tokens = 0
    dist_kgrams = []
    for utt in utts:
        tokens = tokenizer.tokenize(utt)
        total_n_tokens += len(tokens)
        kgrams = list(ngrams(tokens, k))
        for gram in kgrams:
            if gram not in dist_kgrams:
                dist_kgrams.append(gram)

    return len(dist_kgrams) / total_n_tokens


def ent_k(utts: List[str], tokenizer, k: int) -> float:
    '''
    return:
            "entropy" of n-grams
    '''
    total_n_tokens = 0
    dist_kgrams_freq = {}
    for utt in utts:
        tokens = tokenizer.tokenize(utt)
        total_n_tokens += len(tokens)
        kgrams = list(ngrams(tokens, k))
        for gram in kgrams:
            if gram not in dist_kgrams_freq.keys():
                dist_kgrams_freq[gram] = 1
            else:
                dist_kgrams_freq[gram] += 1
    delim = sum(dist_kgrams_freq.values())
    freqs = dist_kgrams_freq.values()
    freqs = list(map(lambda x: -x * np.log(x / delim), freqs))

    return sum(freqs) / delim


def ngam_statistics(original_text,
                    perturbed_text,
                    k_range=[1, 2, 4]
                    ):
    k_dist = {'original': dict(), 'perturbed': dict()}
    k_ent = {'original': dict(), 'perturbed': dict()}
    tokenizer = TreebankWordTokenizer()
    for data_name, data in list(zip(['original', 'perturbed'], [
            original_text, perturbed_text])):
        for k in k_range:
            k_dist[data_name][k] = dist_k(data, tokenizer, k)
            k_ent[data_name][k] = ent_k(data, tokenizer, k)

    k_d = dict()
    for name, metric in k_dist.items():
        k_d[f"k_dist_{name}"] = ' '.join(
            [f"{ngram}: {np.round(count, 3)}" for ngram, count in metric.items()])
    k_e = dict()
    for name, metric in k_ent.items():
        k_e[f"k_ent_{name}"] = ' '.join(
            [f"{ngram}: {np.round(count, 3)}" for ngram, count in metric.items()])
    return k_d, k_e


def check_upos(sent1: str, sent2: str) -> (int, int, int):
    '''
    return:
            number of changed words(tag also changed)
            number of changed words(tag remained)
            number of remained words
    '''
    mismatched_upos = 0
    matched_upos = 0
    matched_words = 0
    for word1, word2 in zip(sent1.words, sent2.words):
        if word1.text != word2.text and word1.upos != word2.upos:
            mismatched_upos += 1
        elif word1.text != word2.text and word1.upos == word2.upos:
            matched_upos += 1
        elif word1.text == word2.text:
            matched_words += 1
    return mismatched_upos, matched_upos, matched_words


def check_entities(sent1: str, sent2: str) -> (int, int, int):
    '''
    return:
            number of remained entities
            number of missed entities
            number of added entities
    '''
    ents1 = [ent.text for ent in sent1.ents]
    ents2 = [ent.text for ent in sent2.ents]
    intersection = set.intersection(set(ents1), set(ents2))
    missed_ne = list(set(ents1) - intersection)
    added_ne = list(set(ents2) - intersection)
    return len(intersection), len(missed_ne), len(added_ne)


def check_unlabeled_dep_parse(sent1: str, sent2: str) -> (int, int, int):
    '''
    return:
            number of changed words
            number of words which changed, but syntax head remained
            number of unchanged words
    '''
    mismatched_head = 0
    matched_head = 0
    matched_words = 0
    for word1, word2 in zip(sent1.words, sent2.words):
        if word1.text != word2.text and word1.head != word2.head:
            mismatched_head += 1
        elif word1.text != word2.text and word1.head == word2.head:
            matched_head += 1
        elif word1.text == word2.text:
            matched_words += 1
    return mismatched_head, matched_head, matched_words


def check_labeled_dep_parse(sent1: str, sent2: str) -> (int, int, int):
    '''
    return:
            number of changed words
            number of words which changed(dependency relation also changed), but syntax head remained
            number of unchanged words
    '''
    mismatched_head = 0
    matched_head = 0
    matched_words = 0
    for word1, word2 in zip(sent1.words, sent2.words):
        if word1.text != word2.text and word1.head != word2.head:
            mismatched_head += 1
        elif word1.text != word2.text and word1.head == word2.head and word1.deprel != word2.deprel:
            matched_head += 1
        elif word1.text == word2.text:
            matched_words += 1
    return mismatched_head, matched_head, matched_words


def stanza_metrics(sequences: List[str], adversarial_sequences: List[str]):
    '''
    return:
            number of unchanged words
            number of changed words
            number of deleted + added named entities
            number of words with changed syntax head
            number of words with changed deprel
    '''
    stat = {
        'matched_words': 0,
        'mismatched_words': 0,
        'changed_entities': 0,
        'changed_head': 0,
        'changed_deprel': 0}

    import stanza
    stanza.download('en')
    nlp = stanza.Pipeline('en')

    for (
            sequence,
            adversarial_sequence) in zip(
            sequences,
            adversarial_sequences):
        sent1 = nlp(sequence).sentences[0]
        sent2 = nlp(adversarial_sequence).sentences[0]
        mismatched_upos, matched_upos, matched_words = check_upos(sent1, sent2)
        kept_ne, missed_ne, added_ne = check_entities(sent1, sent2)
        mismatched_head, matched_head, matched_words = check_unlabeled_dep_parse(
            sent1, sent2)
        mismatched_head, mismatched_deprel, matched_words = check_labeled_dep_parse(
            sent1, sent2)
        stat['matched_words'] += matched_words
        stat['mismatched_words'] += mismatched_upos
        stat['changed_entities'] += missed_ne + added_ne
        stat['changed_head'] += mismatched_head
        stat['changed_deprel'] += mismatched_deprel

    return stat
