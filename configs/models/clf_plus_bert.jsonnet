local transformer_model = "bert-base-uncased";
local embeddings_dropout = 0.1;
local transformer_dim = 768;

{
  "dataset_reader": {
    "type": "text_classification_json",
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": transformer_model,
      "add_special_tokens": true
    },
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": transformer_model,
        "max_length": 512
      }
    }
  },
  "train_data_path": std.extVar("TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("VALID_DATA_PATH"),
  "model": {
    "type": "basic_classifier",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "transformer_embedder",
          "model_name": transformer_model,
          "max_length": 512,
          "train_parameters": false,
          "last_layer_only": false
        }
      }
    },
    "feedforward": {
      "input_dim": transformer_dim,
      "num_layers": 1,
      "hidden_dims": transformer_dim,
      "activations": "tanh"
    },
    "dropout": embeddings_dropout,
  },
  "data_loader": {
    "shuffle": true,
    "batch_size": 128,
    "num_workers": 0,
    "pin_memory": true
  },
  "trainer": {
    "num_epochs": 100,
    "patience": 5,
    "cuda_device": 0,
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "cut_frac": 0.06
    },
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 2e-5,
      "weight_decay": 0.1,
    },
  }
}
