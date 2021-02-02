local transformer_model = "bert-base-uncased";

{
  "dataset_reader": {
    "type": "text_classification_json",
    "token_indexers": {"tokens": {
        "type": "single_id",
        "start_tokens": [
          "<START>"
        ],
        "end_tokens": [
          "<END>"
        ],
        "token_min_padding_length": 5
      }},
    "tokenizer": {
      "type": "just_spaces"
    },
    "skip_label_indexing": true,
    "lazy": false
  },
  "train_data_path": std.extVar("TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("VALID_DATA_PATH"),
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
      "type": "cnn",
      "embedding_dim": 200,
      "num_filters": 8,
      "ngram_filter_sizes": [
        3,
        5
      ]
    },
    "dropout": 0.2,
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