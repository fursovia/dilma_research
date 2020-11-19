from typing import Dict, Optional, Union
import json

import numpy as np
from overrides import overrides
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import DatasetReader, TextClassificationJsonReader
from allennlp.data.fields import MetadataField, TextField, Field, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer


@DatasetReader.register('pairwise')
class PairwiseReader(TextClassificationJsonReader):
    def __init__(
        self,
        token_indexers: Optional[Dict[str, TokenIndexer]] = None,
        tokenizer: Optional[Tokenizer] = None,
        segment_sentences: bool = False,
        max_sequence_length: Optional[int] = None,
        skip_label_indexing: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            token_indexers=token_indexers,
            tokenizer=tokenizer,
            segment_sentences=segment_sentences,
            max_sequence_length=max_sequence_length,
            skip_label_indexing=skip_label_indexing,
            **kwargs,
        )

    @overrides
    def text_to_instance(self, text: str, label: Union[str, int] = None) -> Instance:
        if self._preprocessor is not None:
            text = self._preprocessor(text)
        instance = super().text_to_instance(text=text, label=label)
        instance.fields['texts'] = MetadataField(text)
        return instance


@DatasetReader.register(name="deep_levenshtein")
class DeepLevenshteinReader(DatasetReader):
    def __init__(
            self,
            token_indexers: Dict[str, TokenIndexer],
            tokenizer: Tokenizer,
            lazy: bool = False
    ) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers
        self._tokenizer = tokenizer

    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            for line in data_file.readlines():
                if not line:
                    continue
                items = json.loads(line)
                seq_a = items["text_a"]
                seq_b = items["text_b"]
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
