from pathlib import Path
from datetime import datetime

from allennlp.common.params import Params
import typer
import jsonlines

from dilma.attackers.attacker import Attacker, AttackerOutput
from dilma.constants import ClassificationData, PairClassificationData
from dilma.utils.data import load_jsonlines


app = typer.Typer()


@app.command()
def attack(
        config_path: str,
        data_path: str,
        out_dir: str = None,
        samples: int = typer.Option(None, help="Number of samples")
):
    date = datetime.utcnow().strftime('%H%M%S-%d%m')

    params = Params.from_file(config_path)
    # enable for testing params["attacker"]['device'] = -1
    attacker = Attacker.from_params(params["attacker"])
    data = load_jsonlines(data_path)[:samples]

    dataset_name = Path(data_path).parent.name
    attack_name = Path(config_path).stem

    if out_dir is None:
        out_dir = Path("./results") / f"{date}__{dataset_name}__{attack_name}"

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    params["data_path"] = str(data_path)
    params["out_dir"] = str(out_dir)
    config_path = out_dir / "config.json"
    params.to_file(str(config_path))
    output_path = out_dir / "data.json"

    typer.secho(f"Saving results to {output_path}...", fg="green")
    with jsonlines.open(output_path, "w") as writer:
        for i, sample in enumerate(data):

            try:
                inputs = ClassificationData(**sample)
            except TypeError:
                inputs = PairClassificationData(**sample)

            typer.echo(inputs.text)
            try:
                adversarial_output = attacker.attack(inputs)
            except Exception as e:
                error_message = typer.style(f"Failed to attack because {e}", fg=typer.colors.RED, bold=True)
                typer.echo(error_message)

                adversarial_output = AttackerOutput(
                    data=inputs, adversarial_data=inputs, probability=1.0, adversarial_probability=1.0
                )

            initial_text = adversarial_output.data.text
            adv_text = adversarial_output.adversarial_data.text

            if str(adversarial_output.data.label) != str(adversarial_output.adversarial_data.label):
                adv_text = typer.style(adv_text, fg=typer.colors.GREEN, bold=True)
            else:
                adv_text = typer.style(adv_text, fg=typer.colors.RED, bold=True)

            prob_mess = typer.style(
                f"p(x): {adversarial_output.probability:.2f} -> {adversarial_output.adversarial_probability:.2f}",
                fg=typer.colors.CYAN
            )
            label_mess = typer.style(
                f"y: {adversarial_output.data.label} -> {adversarial_output.adversarial_data.label}",
                fg=typer.colors.YELLOW
            )
            wer = typer.style(f"wer: {adversarial_output.wer}", fg=typer.colors.BLUE)
            message = f"[{i} / {len(data)}] {prob_mess}, {label_mess}, {wer}\n{initial_text}\n{adv_text}\n\n"
            typer.echo(message)

            adversarial_output.data = adversarial_output.data.to_dict()
            adversarial_output.adversarial_data = adversarial_output.adversarial_data.to_dict()
            writer.write(adversarial_output.to_dict())


if __name__ == "__main__":
    app()
