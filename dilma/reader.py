from typing import Dict, Optional, Union
import json

import numpy as np
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, Field, ArrayField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer


@DatasetReader.register(name="pairwise")
class PairwiseReader(DatasetReader):
    def __init__(
            self,
            tokenizer: Tokenizer,
            token_indexers: Dict[str, TokenIndexer] = None,
            skip_label_indexing: bool = False,
            lazy: bool = False
    ) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._tokenizer = tokenizer
        self._skip_label_indexing = skip_label_indexing

    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            for line in data_file.readlines():
                if not line:
                    continue
                items = json.loads(line)
                seq_a = items["text1"]
                seq_b = items["text2"]
                label = items.get("label")
                instance = self.text_to_instance(sequence_a=seq_a, sequence_b=seq_b, label=label)
                yield instance

    def text_to_instance(
        self,
        sequence_a: str,
        sequence_b: str,
        label: Optional[str] = None
    ) -> Instance:
        fields: Dict[str, Field] = dict()
        fields["sequence_a"] = TextField(self._tokenizer.tokenize(sequence_a), self._token_indexers)
        fields["sequence_b"] = TextField(self._tokenizer.tokenize(sequence_b), self._token_indexers)

        if label is not None:
            fields["label"] = LabelField(str(label), skip_indexing=self._skip_label_indexing)

        return Instance(fields)


@DatasetReader.register(name="deep_levenshtein")
class DeepLevenshteinReader(DatasetReader):
    def __init__(
            self,
            tokenizer: Tokenizer,
            token_indexers: Dict[str, TokenIndexer] = None,
            lazy: bool = False
    ) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._tokenizer = tokenizer

    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            for line in data_file.readlines():
                if not line:
                    continue
                items = json.loads(line)
                seq_a = items["text1"]
                seq_b = items["text2"]
                dist = items.get("distance")
                instance = self.text_to_instance(sequence_a=seq_a, sequence_b=seq_b, distance=dist)
                yield instance

    def text_to_instance(
        self,
        sequence_a: str,
        sequence_b: str,
        distance: Optional[float] = None
    ) -> Instance:
        fields: Dict[str, Field] = dict()
        fields["sequence_a"] = TextField(self._tokenizer.tokenize(sequence_a), self._token_indexers)
        fields["sequence_b"] = TextField(self._tokenizer.tokenize(sequence_b), self._token_indexers)

        if distance is not None:
            fields["distance"] = ArrayField(
                array=np.array([distance])
            )

        return Instance(fields)
