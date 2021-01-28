from typing import Union, Dict, List
from copy import deepcopy

from transformers import AutoTokenizer, BertLMHeadModel
import torch
from torch.distributions import Categorical

from dilma.constants import ClassificationData, PairClassificationData, MASK_TOKEN
from dilma.attackers.attacker import Attacker, AttackerOutput
from dilma.utils.data import clean_text


@Attacker.register("sampling_fool")
class SamplingFool(Attacker):

    def __init__(
        self,
        archive_path: str,
        bert_name_or_path: str = "bert-base-uncased",
        num_samples: int = 40,
        temperature: float = 1.5,
        add_mask: bool = True,
        device: int = -1,
    ) -> None:
        super().__init__(archive_path, device)
        self.bert_model = BertLMHeadModel.from_pretrained(bert_name_or_path)
        self.bert_model.eval()

        if self.device >= 0 and torch.cuda.is_available():
            self.bert_model = self.bert_model.to(self.device)

        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_name_or_path)
        self.num_samples = num_samples
        self.temperature = temperature
        self.add_mask = add_mask

    def maybe_add_mask_token(self, text: str) -> str:
        if self.add_mask:
            return text + " " + MASK_TOKEN
        return text

    @torch.no_grad()
    def sample_from_language_model(self, lm_model: BertLMHeadModel, inputs: Dict[str, torch.Tensor]) -> List[str]:
        bert_out = lm_model(**inputs, return_dict=True)
        logits = bert_out.logits / self.temperature
        lm_probs = torch.softmax(logits, dim=-1)
        lm_probs[:, :, self.special_indexes] = 0.0

        adv_texts = []
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

        inputs = self.bert_tokenizer(
            text=self.maybe_add_mask_token(text), return_tensors='pt', padding=True, truncation=True
        )
        inputs = self.move_to_device(inputs)

        initial_prob = self.get_probs_from_indexes(
            self.truncate_start_end_tokens(inputs['input_ids'])
        )[0, label_to_attack_idx]

        outputs = []
        adv_texts = self.sample_from_language_model(self.bert_model, inputs)
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
