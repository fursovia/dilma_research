from typing import Union, Optional, Dict
from copy import deepcopy

from transformers import AutoTokenizer, BertLMHeadModel
from allennlp.models import load_archive
import torch
from torch.nn.functional import gumbel_softmax

from dilma.constants import ClassificationData, PairClassificationData, MASK_TOKEN
from dilma.attackers.attacker import Attacker, AttackerOutput
from dilma.utils.data import clean_text


class DILMAPlus(Attacker):
    def __init__(
            self,
            archive_path: str,
            bert_name_or_path: str = "bert-base-uncased",
            deeplev_archive_path: Optional[str] = None,
            beta: float = 1.0,
            num_steps: int = 8,
            lr: float = 0.001,
            num_gumbel_samples: int = 1,
            tau: float = 1.0,
            num_samples: Optional[int] = None,
            temperature: float = 0.8,
            add_mask: bool = True,
            device: int = -1,
    ) -> None:
        pass
