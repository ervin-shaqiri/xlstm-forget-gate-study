import os
import sys
import argparse

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.trainer import run_experiment, normalize_task_name
from src.visualization import generate_task_visualizations


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified CLI runner for sLSTM Decoupled-Stabilizer Forget Gate Experiments (Parity, Even Pairs, Custom Cycle Navigation)."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["all", "parity", "parity_check", "even_pairs", "custom_cycle_nav", "custom_nav"],
        help="Task to run: 'parity', 'even_pairs', 'custom_cycle_nav', or 'all' (default: all).",
    )
    parser.add_argument(
        "--forget_gate",
        type=str,
        default="all",
        choices=["all", "enabled", "disabled", "fgate_enabled", "fgate_disabled", "baseline"],
        help="Forget gate mode with decoupled stabilizer: 'enabled', 'disabled', or 'all' (default: all).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train_and_eval",
        choices=["train_and_eval", "eval_only", "plot_only"],
        help="Execution mode: 'train_and_eval' (default), 'eval_only' (from saved checkpoints), or 'plot_only'.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override peak learning rate.")
    parser.add_argument("--train_samples", type=int, default=None, help="Override number of training samples.")
    parser.add_argument("--val_samples", type=int, default=None, help="Override number of validation samples.")
    parser.add_argument("--seq_len", type=int, default=None, help="Override maximum training sequence length.")
    parser.add_argument("--device", type=str, default="mps", choices=["mps", "cuda", "cpu"], help="Compute device.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.task == "all":
        tasks = ["parity_check", "even_pairs", "custom_cycle_nav"]
    else:
        tasks = [normalize_task_name(args.task)]

    if args.forget_gate == "all":
        variants = ["enabled", "disabled"]
    else:
        variants = [args.forget_gate]

    for t in tasks:
        if args.mode in ("train_and_eval", "eval_only"):
            for fg in variants:
                run_experiment(
                    task=t,
                    forget_gate=fg,
                    root_dir=ROOT_DIR,
                    mode=args.mode,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    train_samples=args.train_samples,
                    val_samples=args.val_samples,
                    seq_len=args.seq_len,
                    device=args.device,
                    seed=args.seed,
                )
        generate_task_visualizations(task=t, root_dir=ROOT_DIR)


if __name__ == "__main__":
    main()

