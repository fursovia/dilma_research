from allennlp.common import Params
import pytest

import dilma
import dilma.constants
from dilma.attackers.attacker import Attacker, AttackerOutput
from dilma.tests import CONFIG_DIR, PRESETS_DIR
from dilma.utils.data import load_jsonlines


ATTACKERS_CONFIG_DIR = CONFIG_DIR / "attacks"
CONFIGS = list(ATTACKERS_CONFIG_DIR.glob("*.jsonnet"))


def test_num_configs():
    assert len(CONFIGS) >= 1


class TestAttackers:

    test_data = load_jsonlines(str(PRESETS_DIR / "test.json"))[:10]

    @pytest.mark.parametrize("config_path", CONFIGS)
    def test_from_params(self, config_path):

        try:
            params = Params.from_file(
                str(config_path),
                ext_vars={
                    "CLF_PATH": "./presets/models/rotten_tomatoes.tar.gz",
                },
            )
            params["attacker"]["device"] = -1
            attacker = Attacker.from_params(params["attacker"])
        except Exception as e:
            raise AssertionError(
                f"unable to load params from {config_path}, because {e}"
            )

        output = attacker.attack(dilma.constants.ClassificationData(**self.test_data[0]))
        assert isinstance(output, AttackerOutput)
        assert isinstance(output.wer, int)
        assert output.wer >= 0
        assert isinstance(output.prob_diff, float)
        assert abs(output.prob_diff) <= 1.0
        assert isinstance(output.probability, float)
        assert output.probability >= 0.0
        assert isinstance(output.adversarial_probability, float)
        assert output.adversarial_probability >= 0.0
