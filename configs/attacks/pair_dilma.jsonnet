local clf_path = std.extVar("CLF_PATH");

{
  "attacker": {
    "type": "pair_dilma",
    "archive_path": clf_path,
    "bert_name_or_path": "bert-base-uncased",
    "deeplev_archive_path": null,
    "beta": 1.0,
    "num_steps": 30,
    "lr": 0.001,
    "num_gumbel_samples": 1,
    "tau": 1.0,
    "num_samples": 10,
    "temperature": 1.0,
    "add_mask": true,
    "device": 0
  }
}