from pathlib import Path
from transformers import AutoTokenizer, BertLMHeadModel
from torch.distributions import Categorical
import torch

import typer

from dilma.utils.data import load_jsonlines, clean_text, write_jsonlines
from dilma.utils.metrics import calculate_wer

from sklearn.model_selection import train_test_split

from tqdm import tqdm


def sample_examples(text, bert_tokenizer, bert_model, device, temperature=2.0, num_samples=5):
    with torch.no_grad():
        inputs = bert_tokenizer(
            text=text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            add_special_tokens=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = bert_model(**inputs, return_dict=True)

    logits = outputs.logits
    indexes = Categorical(logits=logits[0] / temperature).sample((num_samples,))

    texts = []
    for idx in indexes:
        aug_text = bert_tokenizer.decode(idx.cpu().numpy().tolist()[1:-1])
        texts.append(clean_text(aug_text))

    return texts


app = typer.Typer()


@app.command()
def main(data_dir: Path = None):
    data_dir = data_dir or Path("./data")
    bert_model = BertLMHeadModel.from_pretrained('bert-base-uncased')
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert_model.eval()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    bert_model = bert_model.to(device)

    dataset = []
    for path in list(data_dir.glob('*/*.json')):
        if 'deeplev' not in str(path) and 'qqp' not in str(path):
            data = load_jsonlines(str(path))
            texts = [d['text'] for d in data]
            dataset.extend(texts)

    df = []
    for text in tqdm(dataset):
        examples = sample_examples(text, bert_tokenizer, bert_model, device, temperature=1.5)
        examples = list(set(examples))
        for ex in examples:
            df.append({"text1": text, "text2": ex, "distance": calculate_wer(text, ex)})

    train, valid = train_test_split(df, test_size=0.05)
    write_jsonlines(train, data_dir / "deeplev/train.json")
    write_jsonlines(valid, data_dir / "deeplev/valid.json")


if __name__ == "__main__":
    app()
