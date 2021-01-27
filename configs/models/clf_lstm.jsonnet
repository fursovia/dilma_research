local transformer_model = "bert-base-uncased";

{
  "dataset_reader": {
    "type": "text_classification_json",
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": transformer_model,
      "add_special_tokens": false
    },
//    "token_indexers": {
//      "tokens": {
//        "type": "pretrained_transformer",
//        "model_name": transformer_model,
//        "max_length": 512
//      }
//    },
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
    "type": "basic_classifier_one_hot_support",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 200,
          "trainable": true,
          "pretrained_file": null,
        }
      }
    },
    "seq2vec_encoder": {
        "type": "lstm",
        "input_size": 200,
        "hidden_size": 75,
        "num_layers": 1,
        "dropout": 0.3,
        "bidirectional": true
    },
    "dropout": 0.0,
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