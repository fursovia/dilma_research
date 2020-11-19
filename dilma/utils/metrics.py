import functools
from typing import Sequence
from multiprocessing import Pool
from tqdm import tqdm

import numpy as np
import Levenshtein


@functools.lru_cache(maxsize=5000)
def calculate_wer(text_a: str, text_b: str) -> int:
    # taken from https://github.com/SeanNaren/deepspeech.pytorch/blob/master/decoder.py
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
                pool.imap(_edit_distance_one_vs_all, zip(sequences_a, [sequences_b for _ in sequences_a])),
                total=len(sequences_a),
                desc="# edit distance {}x{}".format(len(sequences_a), len(sequences_b)),
            )
        )
    return np.array(distances)
