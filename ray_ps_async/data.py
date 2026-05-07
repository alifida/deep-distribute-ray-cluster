import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import tensorflow as tf

from ray_ps_async.config import ExperimentConfig


@dataclass
class DatasetBundle:
    train_partitions: Dict[int, List[Tuple[str, int]]]
    val_data: List[Tuple[str, int]]
    test_data: List[Tuple[str, int]]
    class_names: List[str]


def _list_class_dirs(dataset_root: str, class_names: Sequence[str] | None) -> List[str]:
    if class_names:
        return list(class_names)
    class_dirs = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    class_dirs.sort()
    return class_dirs


def prepare_dataset(config: ExperimentConfig) -> DatasetBundle:
    rng = random.Random(config.random_seed)
    class_dirs = _list_class_dirs(config.dataset_root, config.class_names)
    class_to_idx = {name: idx for idx, name in enumerate(class_dirs)}

    train_partitions = {idx: [] for idx in range(config.num_workers)}
    val_data: List[Tuple[str, int]] = []
    test_data: List[Tuple[str, int]] = []

    for cls_name in class_dirs:
        cls_path = os.path.join(config.dataset_root, cls_name)
        files = [
            os.path.join(cls_path, f)
            for f in os.listdir(cls_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]
        files.sort()
        if config.max_samples_per_class:
            files = files[: config.max_samples_per_class]
        rng.shuffle(files)

        n = len(files)
        n_train = int(n * config.train_split)
        n_val = int(n * config.val_split)
        train_files = files[:n_train]
        val_files = files[n_train : n_train + n_val]
        test_files = files[n_train + n_val :]

        for i, path in enumerate(train_files):
            worker_idx = i % config.num_workers
            train_partitions[worker_idx].append((path, class_to_idx[cls_name]))
        val_data.extend((p, class_to_idx[cls_name]) for p in val_files)
        test_data.extend((p, class_to_idx[cls_name]) for p in test_files)

    for w in train_partitions:
        rng.shuffle(train_partitions[w])
    rng.shuffle(val_data)
    rng.shuffle(test_data)

    return DatasetBundle(
        train_partitions=train_partitions,
        val_data=val_data,
        test_data=test_data,
        class_names=class_dirs,
    )


def load_batch(samples: List[Tuple[str, int]], image_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image_size
    xs = []
    ys = []
    for path, label in samples:
        img = tf.keras.utils.load_img(path, target_size=(h, w))
        arr = tf.keras.utils.img_to_array(img).astype(np.float32) / 255.0
        xs.append(arr)
        ys.append(label)
    x = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)
    return x, y

