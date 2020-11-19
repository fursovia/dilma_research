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


ModelsInput = Dict[str, Union[TextFieldTensors, torch.Tensor]]


@dataclass_json
@dataclass
class AttackerOutput:
    data: Union[ClassificationData, PairClassificationData]
    adversarial_data: Union[ClassificationData, PairClassificationData]
    probability: float  # original probability
    adversarial_probability: float
    prob_diff: float
    wer: int
    history: Optional[List[Dict[str, Any]]] = None


class Attacker(ABC, Registrable):

    SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", ".", ";", ",", "-"]

    def __init__(self, archive_path: str, device: int = -1,) -> None:
        archive = load_archive(archive_path, cuda_device=device)
        self.classifier = archive.model
        # self.classifier.eval()
        self.reader = archive.dataset_reader
        self.vocab = self.classifier.vocab

        self.device = device
        self.special_indexes = [self.vocab.get_token_index(token, "tokens") for token in self.SPECIAL_TOKENS]

    @abstractmethod
    def attack(self, data_to_attack: Union[ClassificationData, PairClassificationData]) -> AttackerOutput:
        pass

    def get_probs_from_indexes(self, indexes: torch.Tensor) -> torch.Tensor:
        onehot = torch.nn.functional.one_hot(indexes.long(), num_classes=self.vocab.get_vocab_size())
        probs = self.get_probs_from_onehot(onehot)
        return probs

    def get_probs_from_onehot(self, onehot: torch.Tensor) -> torch.Tensor:
        emb_out = self.classifier.get_embeddings(onehot.float())
        probs = self.classifier.forward_on_embeddings(
            embedded_text=emb_out['embedded_text'],
            mask=emb_out['mask']
        )['probs']
        return probs

    def get_probs_from_textfield_tensors(self, inputs: ModelsInput) -> torch.Tensor:
        probs = self.classifier(inputs)['probs']
        return probs

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

    def text_to_tensor(self, text: str) -> torch.Tensor:
        instances = Batch([
            self.reader.text_to_instance(text)
        ])

        instances.index_instances(self.vocab)
        inputs = instances.as_tensor_dict()["tokens"]
        return move_to_device(inputs, cuda_device=self.device)

    @staticmethod
    def find_best_attack(outputs: List[AttackerOutput]) -> AttackerOutput:
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
