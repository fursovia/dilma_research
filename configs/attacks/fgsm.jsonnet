local clf_path = std.extVar("CLF_PATH");

{
  "attacker": {
    "type": "fgsm",
    "archive_path": clf_path,
    "num_steps": 30,
    "epsilon": 1.0,
    "device": 0
  }
}