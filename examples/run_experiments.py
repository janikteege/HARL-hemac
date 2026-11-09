import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments", nargs="+", help="Name of the experiment(s)")
    parser.add_argument(
        "--seeds", type=int, nargs="+", help="List of seeds", required=True
    )
    parser.add_argument("--steps", type=int, default=1200, help="Number of steps")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    args = parser.parse_args()

    step_count = args.steps
    if step_count < 1:
        parser.error("steps has to be at least 1.")
    workers = args.workers
    if workers < 1:
        parser.error("workers has to be at least 1.")

    this_file_location = Path(__file__).resolve().parent

    experiments = {}
    for experiment in args.experiments:
        experiment_file = (
            this_file_location.parent / "experiments" / f"{experiment}.json"
        )
        if not experiment_file.is_file():
            parser.error(f"experiment config was not found in {experiment_file}")
        experiments[experiment] = experiment_file

    seeds = args.seeds

    train_file = this_file_location / "train.py"
    if not train_file.is_file():
        parser.error(f"train file was not found in {train_file}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    log_dir = this_file_location.parent / "logs" / timestamp
    log_dir.mkdir(parents=True, exist_ok=False)
    print(f"Redirecting Logs to {log_dir}")

    python = sys.executable

    for experiment, experiment_file in experiments.items():
        for seed in seeds:
            command = [
                python,
                "-u",  # unbuffered - print everything immediately
                str(train_file),
                "--load_config",
                str(experiment_file),
                "--exp_name",
                experiment,
                "--seed",
                str(seed),
                "--seed_specify",
                "True",
                "--num_env_steps",
                str(step_count),
                "--n_rollout_threads",
                str(workers),
            ]
            print(f"====Starting {experiment} with Seed {seed}====", flush=True)
            with open(log_dir / f"{experiment}_seed_{seed}.log", "w") as log_file:
                subprocess.run(
                    command,
                    cwd=this_file_location,
                    check=True,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
