import argparse
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dir", help="Path to the directory containing model and config files."
    )
    parser.add_argument(
        "--episodes", type=int, help="How many episodes to render.", default=1
    )
    args = parser.parse_args()

    n_episodes = args.episodes
    if n_episodes < 1:
        parser.error("episodes has to be at least 1.")

    path = Path(args.dir)
    absolute = path.resolve()

    if not absolute.is_dir():
        parser.error("Path has to be a directory.")
    config_file = absolute / "config.json"
    if not config_file.is_file():
        parser.error("directory has to contain a config.json file.")
    models = list((absolute / "models").glob("actor_agent*.pt"))
    models = list(filter(lambda model: model.is_file(), models))
    if len(models) <= 0:
        parser.error(
            "directory has to contain a subdirectory with actor_agent*.pt files."
        )
    model_dir = absolute / "models"

    print(absolute)

    location = Path(__file__).resolve().parent
    train_file = location / "train.py"

    assert train_file.is_file()
    python = sys.executable

    command = [
        python,
        str(train_file),
        "--load_config",
        str(config_file),
        "--model_dir",
        str(model_dir),
        "--use_render",
        "True",
        "--render_episodes",
        str(n_episodes),
        "--torch_threads",
        "1",
    ]
    subprocess.run(command, cwd=location, check=True)
