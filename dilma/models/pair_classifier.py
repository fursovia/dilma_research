from typing import Dict, Optional, Union

import torch
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder, Seq2SeqEncoder
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy


OneHot = torch.Tensor


@Model.register(name="pair_classifier")
class PairClassifier(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2vec_encoder: Seq2VecEncoder,
        seq2seq_encoder: Optional[Seq2SeqEncoder] = None,
    ) -> None:
        super().__init__(vocab)
        self.text_field_embedder = text_field_embedder
        self.seq2seq_encoder = seq2seq_encoder
        self.seq2vec_encoder = seq2vec_encoder
        self.linear = torch.nn.Linear(self.seq2vec_encoder.get_output_dim() * 4, 2)
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

    def encode_sequence(self, sequence: Union[OneHot, TextFieldTensors]) -> torch.Tensor:

        if isinstance(sequence, OneHot):
            # TODO: sparse tensors support
            embedded_sequence = torch.matmul(sequence, self.text_field_embedder._token_embedders["tokens"].weight)
            indexes = torch.argmax(sequence, dim=-1)
            mask = (~torch.eq(indexes, 0)).float()
        else:
            embedded_sequence = self.text_field_embedder(sequence)
            mask = util.get_text_field_mask(sequence).float()
        # It is needed if we pad the initial sequence (or truncate)
        mask = torch.nn.functional.pad(mask, pad=[0, embedded_sequence.size(1) - mask.size(1)])
        if self.seq2seq_encoder is not None:
            embedded_sequence = self.seq2seq_encoder(embedded_sequence, mask=mask)
        embedded_sequence_vector = self.seq2vec_encoder(embedded_sequence, mask=mask)
        return embedded_sequence_vector

    def forward_on_embeddings(
            self,
            embedded_sequence_a: torch.Tensor,
            embedded_sequence_b: torch.Tensor
    ) -> torch.Tensor:
        diff = torch.abs(embedded_sequence_a - embedded_sequence_b)
        representation = torch.cat(
            [
                embedded_sequence_a, embedded_sequence_b, diff, embedded_sequence_a * embedded_sequence_b
            ],
            dim=-1
        )
        approx_distance = self.linear(representation)
        return approx_distance

    def forward(
        self,
        sequence_a: Union[OneHot, TextFieldTensors],
        sequence_b: Union[OneHot, TextFieldTensors],
        label: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        embedded_sequence_a = self.encode_sequence(sequence_a)
        embedded_sequence_b = self.encode_sequence(sequence_b)
        logits = self.forward_on_embeddings(embedded_sequence_a, embedded_sequence_b)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        output_dict = {"logits": logits, "probs": probs}

        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(logits, label)
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"accuracy": self._accuracy.get_metric(reset)}
        return metrics
