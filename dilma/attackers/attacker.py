from typing import List, Optional, Dict, Any, Union
from abc import ABC, abstractmethod

from dataclasses import dataclass
from dataclasses_json import dataclass_json
from allennlp.common.registrable import Registrable
from allennlp.nn.util import move_to_device
from allennlp.data import Batch
from allennlp.models import load_archive
from allennlp.data import TextFieldTensors
import torch

from dilma.constants import ClassificationData, PairClassificationData
from dilma.utils.metrics import calculate_wer


ModelsInput = Dict[str, Union[TextFieldTensors, torch.Tensor]]


@dataclass_json
@dataclass
class AttackerOutput:
    data: Union[ClassificationData, PairClassificationData]
    adversarial_data: Union[ClassificationData, PairClassificationData]
    probability: float  # original probability
    adversarial_probability: float
    prob_diff: Optional[float] = None
    wer: Optional[int] = None
    history: Optional[List[Dict[str, Any]]] = None

    def __post_init__(self):
        if self.prob_diff is None:
            self.prob_diff = self.probability - self.adversarial_probability

        if self.wer is None:
            try:
                self.wer = calculate_wer(self.data.text, self.adversarial_data.text)
            except AttributeError:
                self.wer = calculate_wer(self.data.text1, self.adversarial_data.text1)


class Attacker(ABC, Registrable):

    SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]

    def __init__(self, archive_path: str, device: int = -1,) -> None:
        archive = load_archive(archive_path, cuda_device=device)
        self.classifier = archive.model
        # self.classifier.eval()
        self.reader = archive.dataset_reader
        self.vocab = self.classifier.vocab

        self.device = device
        unused_tokens = [
            token for token in self.vocab.get_token_to_index_vocabulary("tokens") if token.startswith("[unused")
        ]
        self.SPECIAL_TOKENS.extend(unused_tokens)
        self.special_indexes = [self.vocab.get_token_index(token, "tokens") for token in self.SPECIAL_TOKENS]

    @abstractmethod
    def attack(self, data_to_attack: Union[ClassificationData, PairClassificationData]) -> AttackerOutput:
        pass

    def move_to_device(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.device >= 0:
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
        return inputs

    @staticmethod
    def truncate_start_end_tokens(data: torch.Tensor) -> torch.Tensor:
        return data[:, 1:-1]

    def get_probs_from_indexes(self, indexes: torch.Tensor, indexes2: torch.Tensor = None) -> torch.Tensor:
        onehot = torch.nn.functional.one_hot(indexes.long(), num_classes=self.vocab.get_vocab_size())
        if indexes2 is not None:
            onehot2 = torch.nn.functional.one_hot(indexes2.long(), num_classes=self.vocab.get_vocab_size())
        else:
            onehot2 = None
        probs = self.get_probs_from_onehot(onehot, onehot2)
        return probs

    def get_probs_from_onehot(self, onehot: torch.Tensor, onehot2: torch.Tensor = None) -> torch.Tensor:
        if onehot2 is None:
            emb_out = self.classifier.get_embeddings(onehot.float())
            probs = self.classifier.forward_on_embeddings(
                embedded_text=emb_out['embedded_text'],
                mask=emb_out['mask']
            )['probs']
        else:
            emb1 = self.classifier.encode_sequence(onehot.float())
            emb2 = self.classifier.encode_sequence(onehot2.float())
            probs = self.classifier.forward_on_embeddings(emb1, emb2)['probs']
        return probs

    def get_probs_from_textfield_tensors(self, inputs: ModelsInput, inputs2: ModelsInput = None) -> torch.Tensor:
        if inputs2 is None:
            probs = self.classifier(inputs['tokens'])['probs']
        else:
            probs = self.classifier(inputs['tokens'], inputs2['tokens'])['probs']
        return probs

    def get_probs_from_string(self, text: str, text2: str = None) -> torch.Tensor:
        if text:
            if text2 is None:
                inputs = self.text_to_textfield_tensors(text)
                return self.get_probs_from_textfield_tensors(inputs)
            else:
                inputs = self.text_to_textfield_tensors(text, text2)
                return self.get_probs_from_textfield_tensors(inputs['sequence_a'], inputs['sequence_b'])
        else:
            num_labels = self.classifier._num_labels
            probs = torch.ones(1, num_labels) / num_labels
            return move_to_device(probs, self.device)

    def text_to_textfield_tensors(self, text: str, text2: str = None) -> Dict[str, torch.Tensor]:
        if text2 is None:
            instances = Batch([
                self.reader.text_to_instance(text)
            ])
        else:
            instances = Batch([
                self.reader.text_to_instance(text, text2)
            ])
        instances.index_instances(self.vocab)
        inputs = instances.as_tensor_dict()
        return move_to_device(inputs, cuda_device=self.device)

    def probs_to_label(self, probs: torch.Tensor) -> str:
        label_idx = probs.argmax().item()
        label = self.index_to_label(label_idx)
        return label

    def index_to_label(self, label_idx: int) -> str:
        label = self.vocab.get_index_to_token_vocabulary("labels").get(label_idx, str(label_idx))
        return str(label)

    def label_to_index(self, label: str) -> int:
        label_idx = self.vocab.get_token_to_index_vocabulary("labels").get(str(label), label)
        return label_idx

    @staticmethod
    def find_best_attack(outputs: List[AttackerOutput]) -> AttackerOutput:
        # drop zero-length examples
        outputs = list(filter(lambda x: len(x.adversarial_data.text) > 0, outputs))
        # outputs = list(filter(lambda x: len(x.data.text.split()) > x.wer, outputs))
        if len(outputs) == 1:
            return outputs[0]

        changed_label_outputs = []
        for output in outputs:
            if output.data.label != output.adversarial_data.label and output.wer > 0:
                changed_label_outputs.append(output)

        if changed_label_outputs:
            sorted_outputs = sorted(changed_label_outputs, key=lambda x: x.prob_diff, reverse=True)
            best_output = min(sorted_outputs, key=lambda x: x.wer)
        else:
            best_output = max(outputs, key=lambda x: x.prob_diff)

        return best_output
