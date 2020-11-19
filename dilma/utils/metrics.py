import functools
import re

import Levenshtein


def pad_punctuation(string: str) -> str:
    string = re.sub('([.,!?()])', r' \1 ', string)
    string = re.sub('\s{2,}', ' ', string)
    return string


@functools.lru_cache(maxsize=5000)
def calculate_wer(text_a: str, text_b: str) -> int:
    text_a = pad_punctuation(text_a)
    text_b = pad_punctuation(text_b)

    # taken from https://github.com/SeanNaren/deepspeech.pytorch/blob/master/decoder.py
    b = set(text_a.split() + text_b.split())
    word2char = dict(zip(b, range(len(b))))

    w1 = [chr(word2char[w]) for w in text_a.split()]
    w2 = [chr(word2char[w]) for w in text_b.split()]

    return Levenshtein.distance(''.join(w1), ''.join(w2))
