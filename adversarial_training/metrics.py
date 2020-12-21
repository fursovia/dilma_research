# !pip install bert_score
# !pip install stanza

from nltk import ngrams
from nltk.tokenize.treebank import TreebankWordTokenizer
import numpy as np
from bert_score import score
import Levenshtein
import stanza
from typing import List
stanza.download('en')
nlp = stanza.Pipeline('en')


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


def bert_score_count(sequences: List[str], adversarial_sequences: List[str]):
    '''
    return:
            P, R, F1 from BertScore(https://arxiv.org/pdf/1904.09675.pdf)
    '''
    P, R, F1 = score(sequences, adversarial_sequences, lang="en",
                     verbose=False, rescale_with_baseline=True)
    bertscore = {
        'all': {'P': P.tolist(), 'R': R.tolist(), 'F1': F1.tolist()},
        'mean': {'P': P.mean().item(), 'R': R.mean().item(), 'F1': F1.mean().item()}}
    return bertscore


def nlp_metrics(
        sequences: List[str],
        adversarial_sequences: List[str],
        k_range: List[int] = [1, 2, 4],
        count_ent_dist=True,
        count_bert_score=True,
        count_stanza_metrics=True):
    '''
    return:
            dict of metrics
    '''
    assert len(sequences) == len(adversarial_sequences)

    metrics = dict()

    if count_ent_dist:
        tokenizer = TreebankWordTokenizer()
        k_dist = {'sequences': dict(), 'adversarial_sequences': dict()}
        k_ent = {'sequences': dict(), 'adversarial_sequences': dict()}
        for data_name, data in list(zip(['sequences', 'adversarial_sequences'], [
                                    sequences, adversarial_sequences])):
            for k in k_range:
                k_dist[data_name][k] = dist_k(data, tokenizer, k)
                k_ent[data_name][k] = ent_k(data, tokenizer, k)
        metrics['k_dist'] = k_dist
        metrics['k_ent'] = k_ent

    if count_bert_score:
        bertscore = bert_score_count(sequences, adversarial_sequences)
        metrics['BertScore'] = bertscore

    if count_stanza_metrics:
        stanza_metric = stanza_metrics(sequences, adversarial_sequences)
        for metric, value in stanza_metric.items():
            metrics[metric] = value

    return metrics


def calculate_wer(text_a: str, text_b: str) -> int:
    # taken from https://github.com/SeanNaren/deepspeech.pytorch/blob/master/decoder.py
    b = set(text_a.split() + text_b.split())
    word2char = dict(zip(b, range(len(b))))

    w1 = [chr(word2char[w]) for w in text_a.split()]
    w2 = [chr(word2char[w]) for w in text_b.split()]
    return Levenshtein.distance(''.join(w1), ''.join(w2))

def normalized_accuracy_drop(
        wers: List[int],
        y_true: List[int],
        y_adv: List[int],
        gamma: float = 1.0
) -> float:
    assert len(y_true) == len(y_adv)
    nads = []
    for wer, lab, alab in zip(wers, y_true, y_adv):
        if wer > 0 and lab != alab:
            nads.append(1 / wer ** gamma)
        else:
            nads.append(0.0)

    return sum(nads) / len(nads)


