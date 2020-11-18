{
  "dataset_reader": {
    "type": "text_classification_json",
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": "bert-base-uncased",
      "add_special_tokens": false,
    },
    "lazy": false
  },
  "train_data_path": std.extVar("CLS_TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("CLS_VALID_DATA_PATH"),
  "model": {
    "type": "basic_classifier_one_hot_support",
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
    "dropout": 0.1,
  },
  "data_loader": {
    "shuffle": true,
    "batch_size": 256,
    "num_workers": 0,
    "pin_memory": true
  },
//  "distributed": {
//    "master_port": 29555,
//    "cuda_devices": [
//      0,
//      1
//    ]
//  },
  "trainer": {
    "num_epochs": 50,
    "patience": 3,
    "cuda_device": 0
  }
}