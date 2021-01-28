from typing import Union, Optional, Dict, List
from copy import deepcopy

from transformers import AutoTokenizer, BertLMHeadModel
from allennlp.models import load_archive
import torch
from torch.distributions import Categorical
from torch.nn.functional import gumbel_softmax

from dilma.constants import ClassificationData, PairClassificationData, MASK_TOKEN
from dilma.attackers.attacker import Attacker, AttackerOutput
from dilma.utils.data import clean_text


@Attacker.register("dilma")
class DILMA(Attacker):
    """
    DIfferentiable Language Model Attack (DILMA)

    Args:
        archive_path: archive path of the classifier (see available models at ./presets/models)
        bert_name_or_path: Language Model from transformers
        deeplev_archive_path: archive path of the Deep Levenshtein model (e.g. ./presets/models/deeplev.tar.gz)
        beta: Deep Levenshtein loss coefficient
        num_steps: number of steps we update weights of LM
        lr: learning rate
        num_gumbel_samples: number of times we sample from Gumbel-Softmax to estimate probability and edit-distance
        tau: temperature of Gumbel-Softmax
        num_samples: number of times we sample from categorical distribution to get adversarial examples.
            If `None`, we do argmax.
        temperature: temperature for sampling from categorical distribution (if num_samples is not None)
        add_mask: whether to add a `[MASK]` token to the end of the text (this is a trick to make LM work better)
    """
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
        super().__init__(archive_path, device)
        self.bert_model = BertLMHeadModel.from_pretrained(bert_name_or_path)
        self.bert_model.eval()

        if self.device >= 0 and torch.cuda.is_available():
            self.bert_model = self.bert_model.to(self.device)

        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_name_or_path)
        if deeplev_archive_path is not None:
            archive = load_archive(deeplev_archive_path, cuda_device=device)
            self.deeplev = archive.model
        else:
            self.deeplev = None

        self.beta = beta
        self.lr = lr
        self.num_gumbel_samples = num_gumbel_samples
        self.tau = tau
        self.num_samples = num_samples
        self.temperature = temperature
        self.num_steps = num_steps
        self.add_mask = add_mask

    def maybe_add_mask_token(self, text: str) -> str:
        if self.add_mask:
            return text + " " + MASK_TOKEN
        return text

    def calculate_loss(self, prob: torch.Tensor, distance: Optional[torch.Tensor] = None) -> torch.Tensor:

        clf_term = -torch.log(torch.tensor(1.0, device=prob.device) - prob)

        if distance is not None:
            loss = clf_term + self.beta * ((torch.tensor(1.0, device=distance.device) - distance) ** 2)
        else:
            loss = clf_term

        return loss

    def update_weights(self, lm_model: BertLMHeadModel):
        for parameter in lm_model.bert.parameters():
            parameter.data -= self.lr * parameter.grad.data
            parameter.grad = None

    @torch.no_grad()
    def sample_from_language_model(self, lm_model: BertLMHeadModel, inputs: Dict[str, torch.Tensor]) -> List[str]:
        bert_out = lm_model(**inputs, return_dict=True)
        logits = bert_out.logits / self.temperature
        lm_probs = torch.softmax(logits, dim=-1)
        lm_probs[:, :, self.special_indexes] = 0.0

        adv_texts = []
        if self.num_samples is None:
            indexes = lm_probs.argmax(dim=2)
        else:
            indexes = Categorical(probs=lm_probs[0]).sample((self.num_samples,))

        for idx in indexes:
            # skip CLS and SEP tokens
            indexes_to_decode = idx.cpu().numpy().tolist()[1:-1]
            adv_text = clean_text(self.bert_tokenizer.decode(indexes_to_decode))
            adv_texts.append(adv_text)

        return adv_texts

    def attack(self, data_to_attack: Union[ClassificationData, PairClassificationData]) -> AttackerOutput:

        text = data_to_attack.text
        label_to_attack = data_to_attack.label
        label_to_attack_idx = self.label_to_index(label_to_attack)
        initial_prob = None

        if self.deeplev is not None:
            deeplev_input = self.text_to_textfield_tensors(text)
            with torch.no_grad():
                deeplev_emb = self.deeplev.encode_sequence(sequence=deeplev_input["tokens"])
        else:
            deeplev_emb = None

        # we do this so for each .attack() we have an initial bert (with initial weights)
        bert = deepcopy(self.bert_model)

        inputs = self.bert_tokenizer(
            text=self.maybe_add_mask_token(text), return_tensors='pt', padding=True, truncation=True
        )
        inputs = self.move_to_device(inputs)

        outputs = []
        for step in range(self.num_steps):
            bert_out = bert(**inputs, return_dict=True)

            if initial_prob is None:
                initial_prob = self.get_probs_from_indexes(
                    self.truncate_start_end_tokens(inputs['input_ids'])
                )[0, label_to_attack_idx]

            onehot_with_grads = gumbel_softmax(bert_out.logits, tau=self.tau, hard=True)
            onehot_with_grads = self.truncate_start_end_tokens(onehot_with_grads)

            if deeplev_emb is not None:
                adv_deeplev_emb = self.deeplev.encode_sequence(onehot_with_grads)
                distance = self.deeplev.forward_on_embeddings(deeplev_emb, adv_deeplev_emb)
            else:
                distance = None

            clf_prob = self.get_probs_from_onehot(onehot_with_grads)[0, label_to_attack_idx]
            loss = self.calculate_loss(clf_prob, distance=distance)
            loss.backward()
            self.update_weights(bert)

            adv_texts = self.sample_from_language_model(bert, inputs)

            for adv_text in adv_texts:
                with torch.no_grad():
                    clf_probs = self.get_probs_from_string(adv_text)
                    adv_label = self.probs_to_label(clf_probs)
                    adv_prob = clf_probs[0, label_to_attack_idx]

                output = AttackerOutput(
                    data=ClassificationData(text=text, label=str(label_to_attack)),
                    adversarial_data=ClassificationData(text=adv_text, label=str(adv_label)),
                    probability=initial_prob.item(),
                    adversarial_probability=adv_prob.item(),
                )
                outputs.append(output)

        best_output = self.find_best_attack(outputs)
        best_output.history = [deepcopy(out.to_dict()) for out in outputs]
        return best_output
