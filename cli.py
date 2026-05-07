import argparse
import json

from ray_ps_async.config import ExperimentConfig
from ray_ps_async.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Ray async parameter-server training")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sync-every-examples", type=int, default=8)
    parser.add_argument("--num-gpus-per-worker", type=float, default=0.0)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--model-name", default="tiny")
    args = parser.parse_args()

    cfg = ExperimentConfig(
        dataset_root=args.dataset_root,
        num_workers=args.num_workers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sync_every_examples=args.sync_every_examples,
        num_gpus_per_worker=args.num_gpus_per_worker,
        image_size=(args.image_height, args.image_width),
        learning_rate=args.learning_rate,
        model_name=args.model_name,
    )
    result = run_experiment(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

