from typing import Dict, Optional, Tuple, List, Union

from overrides import overrides
import torch

from allennlp.common import Registrable, FromParams
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, util
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy


class Aggregator(torch.nn.Module, FromParams):
    def __init__(self, layers: List[int], pooling: str, aggregation: str):
        """
        Args:
            layers: indexes of layers
            pooling: (cls, last, mean, sum) how to extract vector from hidden states for each layer
            aggregation: (mean, sum, concat) how to combine vectors from each layer
        """
        super(Aggregator, self).__init__()
        self.layers = layers
        self.pooling = pooling
        self.aggregation = aggregation

    def forward(self, outputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:

        vectors = []
        for layer in self.layers:
            output = outputs[layer]

            if self.pooling == 'cls':
                out = output[:, 0, :]
            elif self.pooling == 'last':
                out = output[:, -1, :]
            elif self.pooling == 'mean':
                out = torch.mean(output, dim=1)
            elif self.pooling == 'sum':
                out = torch.sum(output, dim=1)
            else:
                raise NotImplementedError

            vectors.append(out)

        vectors = torch.stack(vectors)
        if self.aggregation == 'mean':
            final_out = torch.mean(vectors, dim=0)
        elif self.aggregation == 'sum':
            final_out = torch.sum(vectors, dim=0)
        elif self.aggregation == 'concat':
            final_out = vectors.transpose(0, 1).reshape(vectors.size(0), -1)
        else:
            raise NotImplementedError

        return final_out


@Model.register("classifier_plus")
class ClassifierPlus(Model):

    def __init__(
        self,
        vocab: Vocabulary,
        aggregator: Aggregator,
        bert_name_or_path: str = "bert-base-uncased",
        feedforward: FeedForward = None,
        dropout: float = None,
        num_labels: int = None,
        label_namespace: str = "labels",
        namespace: str = "tokens",
    ) -> None:

        super().__init__(vocab)
        self._aggregator = aggregator
        self._feedforward = feedforward
        if feedforward is not None:
            self._classifier_input_dim = feedforward.get_output_dim()
        else:
            self._classifier_input_dim = self._seq2vec_encoder.get_output_dim()

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None
        self._label_namespace = label_namespace
        self._namespace = namespace

        if num_labels:
            self._num_labels = num_labels
        else:
            self._num_labels = vocab.get_vocab_size(namespace=self._label_namespace)
        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

    def forward(  # type: ignore
        self, tokens: TextFieldTensors, texts: List[str], label: torch.IntTensor = None
    ) -> Dict[str, torch.Tensor]:

        inputs = bert_tokenizer(
            text=texts, return_tensors='pt', padding=True, truncation=True
        )
        bert_out = bert_model(**inputs, return_dict=True, output_hidden_states=True)
        hidden_states = bert_out.hidden_states
        embedded_text = self._aggregator(hidden_states)

        if self._dropout:
            embedded_text = self._dropout(embedded_text)

        if self._feedforward is not None:
            embedded_text = self._feedforward(embedded_text)

        logits = self._classification_layer(embedded_text)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "probs": probs}
        # output_dict["token_ids"] = util.get_token_ids_from_text_field_tensors(tokens)
        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(logits, label)

        return output_dict
