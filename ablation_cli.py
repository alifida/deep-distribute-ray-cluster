from __future__ import annotations

import argparse
import json

from ray_ps_async.ablation import run_ablation_suite
from ray_ps_async.config import ExperimentConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run novelty ablation suite.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--output-dir", default="new_updates")
    args = parser.parse_args()

    cfg = ExperimentConfig(
        dataset_root=args.dataset_root,
        num_workers=args.num_workers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        ray_address=args.ray_address,
    )
    out = run_ablation_suite(cfg, output_dir=args.output_dir)
    print(json.dumps(out.get("rows", []), indent=2))
    if "artifacts" in out:
        print(json.dumps(out["artifacts"], indent=2))


if __name__ == "__main__":
    main()
