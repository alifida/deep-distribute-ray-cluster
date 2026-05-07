from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import ray
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from ray_ps_async.actors import DatasetServer, ParameterServer, Worker
from ray_ps_async.config import ExperimentConfig
from ray_ps_async.data import DatasetBundle, load_batch, prepare_dataset
from ray_ps_async.models import build_model
from ray_ps_async.serialization import lists_to_weights


def _evaluate(model: tf.keras.Model, samples: List[Tuple[str, int]], image_size: Tuple[int, int], batch_size: int) -> Dict[str, float]:
    if not samples:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "auc": 0.5}

    y_true: List[int] = []
    y_prob: List[float] = []
    for i in range(0, len(samples), batch_size):
        chunk = samples[i : i + batch_size]
        x, y = load_batch(chunk, image_size)
        pred = tf.reshape(model(x, training=False), [-1]).numpy()
        y_true.extend(y.astype(int).tolist())
        y_prob.extend(pred.tolist())

    y_hat = [1 if p >= 0.5 else 0 for p in y_prob]
    auc = 0.5
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        pass
    return {
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_hat, zero_division=0)),
        "auc": auc,
    }


def run_experiment(config: ExperimentConfig) -> Dict:
    if not ray.is_initialized():
        if str(config.ray_address).lower() == "auto":
            ray.init(address="auto", ignore_reinit_error=True)
        else:
            ray.init(ignore_reinit_error=True)

    data: DatasetBundle = prepare_dataset(config)
    available = ray.available_resources()
    available_gpus = float(available.get("GPU", 0.0))
    requested_gpu_per_worker = float(config.num_gpus_per_worker)
    needed_gpus = requested_gpu_per_worker * float(config.num_workers)
    gpu_fallback_applied = False
    effective_gpu_per_worker = requested_gpu_per_worker
    if requested_gpu_per_worker > 0 and available_gpus < needed_gpus and config.allow_gpu_fallback:
        effective_gpu_per_worker = 0.0
        gpu_fallback_applied = True

    runtime_env = {"working_dir": os.getcwd()}

    def _run_attempt(worker_gpu: float, data_mode: str):
        ps_opts = {"num_gpus": 1 if config.use_gpu_on_ps and worker_gpu > 0 else 0, "runtime_env": runtime_env}
        ps = ParameterServer.options(**ps_opts).remote(config)
        dataset_server = None
        if str(data_mode).lower() == "stream_from_head":
            dataset_server = DatasetServer.options(runtime_env=runtime_env).remote(data.train_partitions, config.image_size)

        workers = []
        for idx in range(config.num_workers):
            opts = {"num_gpus": float(worker_gpu), "runtime_env": runtime_env}
            actor = Worker.options(**opts).remote(f"worker_{idx}", idx, data.train_partitions[idx], config)
            workers.append(actor)

        worker_results_local = ray.get([w.train.remote(ps, dataset_server) for w in workers])
        global_state_local = ray.get(ps.get_global_weights.remote())
        comm_metrics_local = ray.get(ps.get_communication_metrics.remote())
        return worker_results_local, global_state_local, comm_metrics_local

    attempts: list[dict] = []
    worker_results = None
    global_state = None
    comm_metrics = None
    final_data_mode = str(config.data_mode)
    final_gpu_per_worker = float(effective_gpu_per_worker)
    fallback_reason = None

    try:
        worker_results, global_state, comm_metrics = _run_attempt(final_gpu_per_worker, final_data_mode)
        attempts.append({"worker_gpu": final_gpu_per_worker, "data_mode": final_data_mode, "status": "ok"})
    except Exception as first_exc:
        attempts.append({"worker_gpu": final_gpu_per_worker, "data_mode": final_data_mode, "status": "failed", "error": str(first_exc)})
        if not config.allow_gpu_fallback:
            raise
        fallback_reason = str(first_exc)

        # First rescue: force CPU, keep same data mode.
        final_gpu_per_worker = 0.0
        try:
            worker_results, global_state, comm_metrics = _run_attempt(final_gpu_per_worker, final_data_mode)
            attempts.append({"worker_gpu": final_gpu_per_worker, "data_mode": final_data_mode, "status": "ok"})
            gpu_fallback_applied = True
        except Exception as second_exc:
            attempts.append({"worker_gpu": final_gpu_per_worker, "data_mode": final_data_mode, "status": "failed", "error": str(second_exc)})
            # Second rescue: force CPU + shared_path for maximum compatibility.
            final_data_mode = "shared_path"
            worker_results, global_state, comm_metrics = _run_attempt(final_gpu_per_worker, final_data_mode)
            attempts.append({"worker_gpu": final_gpu_per_worker, "data_mode": final_data_mode, "status": "ok"})
            gpu_fallback_applied = True

    model = build_model(config.model_name, (config.image_size[0], config.image_size[1], 3))
    model.set_weights(lists_to_weights(global_state["weights"], global_state["shapes"]))

    val_metrics = _evaluate(model, data.val_data, config.image_size, config.batch_size)
    test_metrics = _evaluate(model, data.test_data, config.image_size, config.batch_size)

    return {
        "config": config.__dict__,
        "ray_resources": ray.available_resources(),
        "execution": {
            "effective_num_gpus_per_worker": final_gpu_per_worker,
            "gpu_fallback_applied": gpu_fallback_applied,
            "data_mode": final_data_mode,
            "fallback_reason": fallback_reason,
            "attempts": attempts,
        },
        "workers": worker_results,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "communication_cost": comm_metrics,
        "dataset_stats": {
            "num_workers": config.num_workers,
            "train_partition_sizes": {str(k): len(v) for k, v in data.train_partitions.items()},
            "val_samples": len(data.val_data),
            "test_samples": len(data.test_data),
            "classes": data.class_names,
        },
    }

