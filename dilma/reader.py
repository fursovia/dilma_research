from typing import Dict, Optional, Union

from allennlp.data.dataset_readers import DatasetReader, TextClassificationJsonReader
from allennlp.data.fields import MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from overrides import overrides


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
