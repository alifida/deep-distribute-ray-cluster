from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ExperimentConfig:
    dataset_root: str
    num_workers: int = 2
    epochs: int = 1
    batch_size: int = 8
    image_size: Tuple[int, int] = (224, 224)
    learning_rate: float = 1e-3
    sync_every_examples: int = 8
    num_gpus_per_worker: float = 0.0
    ray_address: str = "local"
    data_mode: str = "shared_path"
    allow_gpu_fallback: bool = True
    ssp_staleness: int = 2
    quorum_fraction: float = 0.6
    round_timeout_ms: int = 100
    quantization: str = "fp32"  # fp32 | fp16 | q8 | topk
    topk_pct: float = 1.0
    residual_feedback: bool = True
    ps_optimizer: str = "adam"  # adam | sgd
    ps_momentum: float = 0.9
    adaptive_sync: bool = True
    use_gpu_on_ps: bool = False
    random_seed: int = 42
    model_name: str = "tiny"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    max_samples_per_class: Optional[int] = None
    class_names: Optional[List[str]] = field(default=None)
    # API / multi-job: ties Ray actors to this id for force-termination (see runner.register_running_experiment).
    run_id: str = ""

