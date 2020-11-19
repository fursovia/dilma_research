from typing import Union
from copy import deepcopy

from transformers import AutoTokenizer, BertLMHeadModel
import torch

from dilma.constants import ClassificationData, PairClassificationData, MASK_TOKEN
from dilma.attackers.attacker import Attacker, AttackerOutput
from dilma.utils.metrics import calculate_wer
from dilma.utils.data import clean_text


@Attacker.register("dilma")
class DILMA(Attacker):
    def __init__(
        self,
        archive_path: str,
        bert_name_or_path: str = "bert-base-uncased",
        alpha: float = 2.0,
        beta: float = 1.0,
        lr: float = 0.001,
        num_gumbel_samples: int = 3,
        tau: float = 1.0,
        num_samples: int = 5,
        temperature: float = 0.8,
        num_steps: int = 8,
        add_mask: bool = True,
        device: int = -1,
    ) -> None:
        super().__init__(archive_path, device)
        self.bert_model = BertLMHeadModel.from_pretrained(bert_name_or_path)
        self.bert_model.eval()

        if self.device >= 0 and torch.cuda.is_available():
            self.bert_model = self.bert_model.to(self.device)

        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_name_or_path)

        self.alpha = alpha
        self.beta = beta
        self.lr = lr
        self.num_gumbel_samples = num_gumbel_samples
        self.tau = tau
        self.num_samples = num_samples
        self.temperature = temperature
        self.num_steps = num_steps
        self.add_mask = add_mask

    def tokenize_text(self, text: str):
        return

    def attack(self, data_to_attack: Union[ClassificationData, PairClassificationData]) -> AttackerOutput:

        text = data_to_attack.text
        label_to_attack = data_to_attack.label
        label_to_attack_idx = self.label_to_index(label_to_attack)
        initial_prob = None

        bert = deepcopy(self.bert_model)
        if self.add_mask:
            input_text = text + " " + MASK_TOKEN
        else:
            input_text = text

        inputs = self.bert_tokenizer(text=input_text, return_tensors='pt', padding=True, truncation=True)
        if self.device >= 0:
            inputs = {key: val.to(self.device) for key, val in inputs.items()}

        outputs = []
        for step in range(self.num_steps):
            bert_out = bert(**inputs, return_dict=True)

            if initial_prob is None:
                clf_probs = self.get_probs_from_indexes(inputs['input_ids'][:, 1:-1])
                initial_prob = clf_probs[0, label_to_attack_idx]

            onehot = torch.nn.functional.gumbel_softmax(bert_out.logits, tau=self.tau, hard=True)[:, 1:-1]

            clf_probs = self.get_probs_from_onehot(onehot)
            prob = clf_probs[0, label_to_attack_idx]
            loss = -torch.log(1 - prob)

            loss.backward()

            for parameter in bert.bert.parameters():
                parameter.data -= self.lr * parameter.grad.data
                parameter.grad = None

            with torch.no_grad():
                bert_out = bert(**inputs, return_dict=True)
                lm_probs = torch.softmax(bert_out.logits, dim=-1)
                lm_probs[:, :, self.special_indexes] = 0.0

                indexes = lm_probs.argmax(dim=2)[:, 1:-1]
                clf_probs = self.get_probs_from_indexes(indexes)
                adv_label = self.probs_to_label(clf_probs)
                adv_prob = clf_probs[0, label_to_attack_idx]
                adv_text = self.bert_tokenizer.decode(indexes.cpu().numpy()[0].tolist())
                adv_text = clean_text(adv_text)

            output = AttackerOutput(
                data=ClassificationData(text=text, label=str(label_to_attack)),
                adversarial_data=ClassificationData(text=adv_text, label=str(adv_label)),
                probability=initial_prob.item(),
                adversarial_probability=adv_prob.item(),
                prob_diff=(initial_prob - adv_prob).item(),
                wer=calculate_wer(text, adv_text),
            )
            outputs.append(output)

        best_output = self.find_best_attack(outputs)
        return best_output
