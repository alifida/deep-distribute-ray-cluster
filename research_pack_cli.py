from __future__ import annotations

import argparse
import json

from ray_ps_async.config import ExperimentConfig
from ray_ps_async.research_pack import run_research_pack


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated multi-dataset research evaluation pack.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dataset-roots-csv", default="")
    parser.add_argument("--repeats", type=int, default=3)
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
    dataset_roots = [x.strip() for x in str(args.dataset_roots_csv).split(",") if x.strip()]
    out = run_research_pack(cfg, dataset_roots=dataset_roots or [args.dataset_root], repeats=args.repeats, output_dir=args.output_dir)
    print(json.dumps(out.get("best_by_dataset", {}), indent=2))
    if "artifacts" in out:
        print(json.dumps(out["artifacts"], indent=2))


if __name__ == "__main__":
    main()
