from dataclasses import dataclass
from dataclasses_json import dataclass_json


MASK_TOKEN = "[MASK]"


@dataclass_json
@dataclass
class ClassificationData:
    text: str
    label: str


@dataclass_json
@dataclass
class PairClassificationData:
    text_a: str
    text_b: str
    label: str