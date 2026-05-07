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
    use_gpu_on_ps: bool = False
    random_seed: int = 42
    model_name: str = "tiny"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    max_samples_per_class: Optional[int] = None
    class_names: Optional[List[str]] = field(default=None)

