from pathlib import Path

from allennlp.common.params import Params
import typer
import jsonlines

from dilma.attackers import Attacker
from dilma.constants import ClassificationData, PairClassificationData
from dilma.utils.data import load_jsonlines


app = typer.Typer()


@app.command()
def attack(config_path: str, out_dir: str = None, samples: int = typer.Option(None, help="Number of samples")):
    out_dir = out_dir or "./results"
    out_dir = Path(out_dir)
    output_path = out_dir / "attacked_data.json"

    params = Params.from_file(config_path)
    # enable for testing params['attacker']['device'] = -1
    attacker = Attacker.from_params(params["attacker"])

    data = load_jsonlines(params["data_path"])[:samples]

    typer.secho(f"Saving results to {output_path} ...", fg="green")
    with jsonlines.open(output_path, "w") as writer:
        for i, sample in enumerate(data):

            try:
                inputs = ClassificationData(**sample)
            except TypeError:
                inputs = PairClassificationData(**sample)

            adversarial_output = attacker.attack(inputs)
            initial_text = adversarial_output.data.text
            adv_text = adversarial_output.adversarial_data.text

            if adversarial_output.data.label != adversarial_output.adversarial_data.label:
                adv_text = typer.style(adv_text, fg=typer.colors.GREEN, bold=True)
            else:
                adv_text = typer.style(adv_text, fg=typer.colors.RED, bold=True)

            prob_message = f"{adversarial_output.probability:.2f} -> {adversarial_output.adversarial_probability:.2f}"
            label_message = f"{adversarial_output.data.label} -> {adversarial_output.adversarial_data.label}"
            message = f"[{i} / {len(data)}] {prob_message} ||| {label_message}\n{initial_text}\n{adv_text}\n\n"
            typer.echo(message)

            adversarial_output.data = adversarial_output.data.to_dict()
            adversarial_output.adversarial_data = adversarial_output.adversarial_data.to_dict()
            writer.write(adversarial_output.to_dict())


if __name__ == "__main__":
    app()
