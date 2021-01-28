local clf_path = std.extVar("CLF_PATH");

{
  "attacker": {
    "type": "dilma",
    "archive_path": clf_path,
    "bert_name_or_path": "bert-base-uncased",
    "num_samples": 40,
    "temperature": 1.5,
    "add_mask": true,
    "device": 0
  }
}