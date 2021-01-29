"""Explaining and Harnessing Adversarial Examples.
Generating Natural Language Adversarial Examples on a Large Scale with Generative Models"""

from typing import Union
from copy import deepcopy
import random

import torch
from allennlp.nn import util

from dilma.attackers.attacker import Attacker, AttackerOutput
from dilma.constants import ClassificationData, PairClassificationData
from dilma.utils.data import decode_indexes


@Attacker.register('fgsm')
class FGSMAttacker(Attacker):

    def __init__(self, archive_path: str, num_steps: int = 10, epsilon: float = 0.01, device: int = -1):
        super().__init__(archive_path, device)

        self.num_steps = num_steps
        self.epsilon = epsilon

        self.emb_layer = util.find_embedding_layer(self.classifier).weight
        self.vocab_size = self.vocab.get_vocab_size()

    def attack(self, data_to_attack: Union[ClassificationData, PairClassificationData]) -> AttackerOutput:
        # get inputs to the model
        inputs = self.text_to_textfield_tensors(text=data_to_attack.text)

        adversarial_indexes = inputs["tokens"]["tokens"]["tokens"][0]

        # original probability of the true label
        orig_prob = self.get_probs_from_textfield_tensors(inputs)[self.label_to_index(data_to_attack.label)].item()

        # get mask and embeddings
        emb_out = self.classifier.get_embeddings(inputs["tokens"])

        # disable gradients using a trick
        embeddings = emb_out["embedded_text"].detach()
        embeddings_splitted = [e for e in embeddings[0]]

        outputs = []
        for step in range(self.num_steps):
            # choose random index of embeddings (except for start/end tokens)
            random_idx = random.randint(0, max(1, len(data_to_attack.text) - 1))
            # only one embedding can be modified
            embeddings_splitted[random_idx].requires_grad = True

            # calculate the loss for current embeddings
            loss = self.classifier.forward_on_embeddings(
                embeddings=torch.stack(embeddings_splitted, dim=0).unsqueeze(0),
                mask=emb_out["mask"],
                label=inputs["label"],
            )["loss"]
            loss.backward()

            # update the chosen embedding
            embeddings_splitted[random_idx] = (
                    embeddings_splitted[random_idx] + self.epsilon * embeddings_splitted[random_idx].grad.data.sign()
            )
            self.classifier.zero_grad()

            # find the closest embedding for the modified one
            distances = torch.nn.functional.pairwise_distance(embeddings_splitted[random_idx], self.emb_layer)
            # we dont choose special tokens
            distances[self.special_indexes] = 10 ** 16

            # swap embeddings
            closest_idx = distances.argmin().item()
            embeddings_splitted[random_idx] = self.emb_layer[closest_idx]
            embeddings_splitted = [e.detach() for e in embeddings_splitted]

            # get adversarial indexes
            adversarial_indexes[random_idx] = closest_idx

            adv_data = deepcopy(data_to_attack)
            # TODO: decode using tokenizer
            adv_data.text = " ".join(decode_indexes(adversarial_indexes, vocab=self.vocab))

            adv_inputs = self.text_to_textfield_tensors(adv_data.text)

            # get adversarial probability and adversarial label
            adv_probs = self.get_probs_from_textfield_tensors(adv_inputs)
            adv_data.label = self.probs_to_label(adv_probs)
            adv_prob = adv_probs[self.label_to_index(data_to_attack.label)].item()

            output = AttackerOutput(
                data=ClassificationData(text=data_to_attack.text, label=str(data_to_attack.label)),
                adversarial_data=ClassificationData(text=adv_data.text, label=str(adv_data.label)),
                probability=orig_prob,
                adversarial_probability=adv_prob,
            )

            outputs.append(output)

        best_output = self.find_best_attack(outputs)
        best_output.history = [output.to_dict() for output in outputs]

        return best_output
