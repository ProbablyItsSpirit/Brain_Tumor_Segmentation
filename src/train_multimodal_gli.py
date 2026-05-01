"""CLI wrapper for the optimized multimodal GLI MVP (src)

Run this from the repository root as shown in `commands_to_run.txt`.
"""

from __future__ import annotations

import argparse
from train_multimodal_gli_mvp import train_multimodal_gli


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 multimodal GLI MVP (src)")
    parser.add_argument("--data-root", default="BraTS-2024-Complete")
    parser.add_argument("--gli-list-train", default="patient_lists/gli_train.txt")
    parser.add_argument("--gli-list-val", default="patient_lists/gli_val.txt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint-dir", default="checkpoints/gli_multimodal_mvp_seed42")
    parser.add_argument("--results-dir", default="results/gli_multimodal_mvp_seed42")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true", default=True)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--channels-last", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_multimodal_gli(
        data_root=args.data_root,
        gli_list_train=args.gli_list_train,
        gli_list_val=args.gli_list_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        amp=args.amp,
        accum_steps=args.accum_steps,
        channels_last=args.channels_last,
    )


if __name__ == "__main__":
    main()
