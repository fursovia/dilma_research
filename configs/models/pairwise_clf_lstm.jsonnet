local transformer_model = "bert-base-uncased";

{
  "dataset_reader": {
    "type": "pairwise",
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": transformer_model,
      "add_special_tokens": false
    },
    "lazy": false
  },
  "train_data_path": std.extVar("TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("VALID_DATA_PATH"),
  "vocabulary": {
    "type": "extend",
    "directory": "presets/vocab",
    "padding_token": "[PAD]",
    "oov_token": "[UNK]"
  },
  "model": {
    "type": "pair_classifier",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 128,
          "trainable": true
        }
      }
    },
    "seq2vec_encoder": {
        "type": "gru",
        "input_size": 128,
        "hidden_size": 256,
        "num_layers": 1,
        "dropout": 0.1,
        "bidirectional": true
    },
    "dropout": 0.2
  },
  "data_loader": {
    "shuffle": true,
    "batch_size": 256,
    "num_workers": 0,
    "pin_memory": true
  },
  "trainer": {
    "num_epochs": 50,
    "patience": 3,
    "cuda_device": 0
  }
}