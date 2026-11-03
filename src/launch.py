"""Submit one experiment as a chain of Slurm jobs, and collect the results.

    python -m src.launch submit --approach mapping --backbone qwen-8b --encoder nllb-600m \\
        --stages 1,2,3 --benchmarks xgqa,cvqa,cvqa_mcq
    python -m src.launch collect

submit checks the flags, works out every path, and submits:
  1. one train.py job per stage, each starting after the one before it
     succeeds, and each loading the previous stage's checkpoint;
  2. one evaluate.py job per benchmark, starting after training succeeds.
direct and translate-then-test train nothing, so they submit only step 2.

Each run has its own folder, named from its settings:

    $SCRATCH/M2-ALIGN/runs/mapping_qwen-8b_nllb-600m_stages123_seed42/
        run.json                    the settings of this run
        stage{N}/mapping.pt         one checkpoint per trained stage
        results/{benchmark}/        evaluate.py's records and summary.json
        logs/                       the Slurm logs

collect reads every run's summary.json files into runs/results.tsv.

This file imports no torch, so it runs instantly on a login node.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(os.environ.get("SCRATCH", "."), "M2-ALIGN", "runs")
SEED = 42
ACCOUNT = "def-annielee"

APPROACHES = ["direct", "translate-then-test", "mapping"]
ENCODER_NAMES = ["nllb-600m", "nllb-3.3b", "mt5-large", "mt5-xl"]
BACKBONE_NAMES = ["qwen-2b", "qwen-8b", "qwen-32b"]
BENCHMARK_NAMES = ["xgqa", "cvqa", "cvqa_mcq", "worldcuisines_task1", "worldcuisines_task2", "mgsm", "msvamp"]

# Slurm resources per training stage. 
RESOURCES = {
    1: {"gpus": 4, "cpus": 48, "mem": "490G", "time": "2-00:00:00"},
    2: {"gpus": 1, "cpus": 8, "mem": "64G", "time": "1-00:00:00"},
    3: {"gpus": 1, "cpus": 8, "mem": "64G", "time": "3-00:00:00"},
}
EVAL_RESOURCES = {"gpus": 1, "cpus": 8, "mem": "64G", "time": "12:00:00"}
# xGQA has about 12,600 questions in each of its 7 languages.
EVAL_TIME = {"xgqa": "1-00:00:00"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit M2-ALIGN runs and collect their results.")
    commands = parser.add_subparsers(dest="command", required=True)

    submit_parser = commands.add_parser("submit", help="Submit the training and evaluation jobs for one run.")
    submit_parser.add_argument("--approach", required=True, choices=APPROACHES)
    submit_parser.add_argument("--backbone", required=True, choices=BACKBONE_NAMES)
    submit_parser.add_argument("--encoder", default="", help="mapping only.")
    submit_parser.add_argument("--stages", default="", help="mapping only, e.g. 1,2,3 or 2,3.")
    submit_parser.add_argument("--benchmarks", required=True, help="Comma-separated, e.g. xgqa,cvqa,cvqa_mcq.")
    submit_parser.add_argument("--seed", type=int, default=SEED)
    submit_parser.add_argument("--eval-only", action="store_true", help="mapping only: evaluate checkpoints already trained.")
    submit_parser.add_argument("--dry-run", action="store_true", help="Print the sbatch commands, submit nothing.")
    submit_parser.add_argument("--overwrite", action="store_true", help="Replace checkpoints or results that already exist.")

    commands.add_parser("collect", help="Gather every run's results into runs/results.tsv.")

    args = parser.parse_args()
    if args.command == "collect":
        collect()
        return
    args.stages = [int(s) for s in args.stages.split(",") if s.strip()]
    args.benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    problem = check_run(args)
    if problem:
        parser.error(problem)
    submit(args)


def check_run(args) -> str:
    """Return what is wrong with the flags, or an empty string if nothing is."""
    unknown = [b for b in args.benchmarks if b not in BENCHMARK_NAMES]
    if unknown:
        return f"unknown benchmarks {unknown}; choose from {BENCHMARK_NAMES}"
    if not args.benchmarks:
        return "--benchmarks is empty"
    if args.approach != "mapping":
        if args.encoder or args.stages or args.eval_only:
            return f"--approach {args.approach} trains nothing, so it takes no --encoder, --stages or --eval-only"
        return ""
    if args.encoder not in ENCODER_NAMES:
        return f"--approach mapping needs --encoder, one of {ENCODER_NAMES}"
    if not args.stages:
        return "--approach mapping needs --stages, e.g. 1,2,3"
    if any(s not in RESOURCES for s in args.stages) or args.stages != sorted(set(args.stages)):
        return f"--stages must be distinct stages from {sorted(RESOURCES)}, in order, e.g. 1,2,3"
    if args.eval_only and not os.path.exists(checkpoint_path(args, args.stages[-1])):
        return f"--eval-only needs a trained checkpoint at {checkpoint_path(args, args.stages[-1])}"
    return ""


# Paths
def run_dir(args) -> str:
    if args.approach == "mapping":
        stages = "".join(str(s) for s in args.stages)
        name = f"mapping_{args.backbone}_{args.encoder}_stages{stages}_seed{args.seed}"
    else:
        # Greedy decoding with a frozen backbone does not depend on the seed.
        name = f"{args.approach}_{args.backbone}"
    return os.path.join(RUNS_DIR, name)


def checkpoint_path(args, stage: int) -> str:
    return os.path.join(run_dir(args), f"stage{stage}", "mapping.pt")


def results_dir(args, benchmark: str) -> str:
    return os.path.join(run_dir(args), "results", benchmark)


# Submitting
def submit(args) -> None:
    folder = run_dir(args)
    train_stages = [] if args.eval_only else args.stages
    outputs = [checkpoint_path(args, s) for s in train_stages] + [results_dir(args, b) for b in args.benchmarks]
    existing = [p for p in outputs if os.path.exists(p)]
    if existing and not args.overwrite and not args.dry_run:
        sys.exit("These already exist. Pass --overwrite to replace them:\n  " + "\n  ".join(existing))
    if not args.dry_run:
        os.makedirs(os.path.join(folder, "logs"), exist_ok=True)
        settings = {k: getattr(args, k) for k in ("approach", "backbone", "encoder", "stages", "seed")}
        with open(os.path.join(folder, "run.json"), "w") as f:
            json.dump(settings, f, indent=2)

    after = ""
    for stage in train_stages:
        after = sbatch(args, f"stage{stage}", train_command(args, stage), RESOURCES[stage], after)
    for benchmark in args.benchmarks:
        resources = {**EVAL_RESOURCES, "time": EVAL_TIME.get(benchmark, EVAL_RESOURCES["time"])}
        sbatch(args, f"eval_{benchmark}", eval_command(args, benchmark), resources, after)
    print(f"run folder: {folder}")


def train_command(args, stage: int) -> list[str]:
    command = [
        "-m", "src.train", "--backbone", args.backbone, "--encoder", args.encoder, "--stage", str(stage),
        "--save-to", checkpoint_path(args, stage), "--seed", str(args.seed),
    ]
    # Every stage after the first starts from the stage before it.
    if stage != args.stages[0]:
        previous = args.stages[args.stages.index(stage) - 1]
        command += ["--load-from", checkpoint_path(args, previous)]
    # w/o Stage 1 also drops translation replay. Math replay is always on.
    if 1 in args.stages:
        command.append("--translation-replay")
    gpus = RESOURCES[stage]["gpus"]
    launcher = ["torchrun", "--standalone", f"--nproc-per-node={gpus}"] if gpus > 1 else ["python"]
    return launcher + command


def eval_command(args, benchmark: str) -> list[str]:
    command = [
        "python", "-m", "src.evaluate", "--approach", args.approach, "--backbone", args.backbone,
        "--benchmark", benchmark, "--results", results_dir(args, benchmark),
    ]
    if args.approach == "mapping":
        command += ["--encoder", args.encoder, "--checkpoint", checkpoint_path(args, args.stages[-1])]
    return command


def sbatch(args, job: str, command: list[str], resources: dict, after: str) -> str:
    """Submit command through job.sh. Starts only after job id `after` succeeds, if given. Returns the new job id."""
    name = f"{os.path.basename(run_dir(args))}_{job}"
    sbatch_args = [
        "sbatch", "--parsable", f"--account={ACCOUNT}", f"--job-name={name}", f"--chdir={REPO}",
        f"--output={os.path.join(run_dir(args), 'logs', job + '_%j.log')}",
        f"--gres=gpu:{resources['gpus']}", f"--cpus-per-task={resources['cpus']}",
        f"--mem={resources['mem']}", f"--time={resources['time']}",
    ]
    if after:
        sbatch_args += [f"--dependency=afterok:{after}", "--kill-on-invalid-dep=yes"]
    full = sbatch_args + [os.path.join(REPO, "job.sh")] + command
    if args.dry_run:
        print(" ".join(full) + "\n")
        return f"<{job}>"
    job_id = subprocess.run(full, check=True, capture_output=True, text=True).stdout.strip().split(";")[0]
    print(f"{job}: job {job_id}" + (f" (after {after})" if after else ""))
    return job_id


# Collecting
def collect() -> None:
    """Write runs/results.tsv: one line per run, benchmark and language."""
    columns = ["run", "approach", "backbone", "encoder", "stages", "seed", "benchmark", "lang", "accuracy", "n"]
    lines = ["\t".join(columns)]
    for run in sorted(os.listdir(RUNS_DIR)) if os.path.isdir(RUNS_DIR) else []:
        config_path = os.path.join(RUNS_DIR, run, "run.json")
        results = os.path.join(RUNS_DIR, run, "results")
        if not (os.path.exists(config_path) and os.path.isdir(results)):
            continue
        with open(config_path) as f:
            config = json.load(f)
        for benchmark in sorted(os.listdir(results)):
            summary_path = os.path.join(results, benchmark, "summary.json")
            if not os.path.exists(summary_path):
                continue
            with open(summary_path) as f:
                summary = json.load(f)
            for lang, score in summary.items():
                values = [
                    run, config["approach"], config["backbone"], config["encoder"] or "-",
                    ",".join(map(str, config["stages"])) or "-", str(config["seed"]),
                    benchmark, lang, str(score["accuracy"]), str(score["n"]),
                ]
                lines.append("\t".join(values))
    out = os.path.join(RUNS_DIR, "results.tsv")
    os.makedirs(RUNS_DIR, exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{len(lines) - 1} results -> {out}")


if __name__ == "__main__":
    main()
