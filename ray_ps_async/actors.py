from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import ray
import tensorflow as tf

from ray_ps_async.config import ExperimentConfig
from ray_ps_async.data import load_batch
from ray_ps_async.models import build_model
from ray_ps_async.serialization import add_weights, subtract_weights, weights_to_lists, lists_to_weights


@ray.remote
class ParameterServer:
    def __init__(self, config: ExperimentConfig):
        input_shape = (config.image_size[0], config.image_size[1], 3)
        self.model = build_model(config.model_name, input_shape)
        self.global_weights = [w.astype(np.float32) for w in self.model.get_weights()]
        self.version = 0
        self.lr = float(config.learning_rate)
        self.worker_last_full_gradient: Dict[str, List[np.ndarray]] = {}
        self.comm = {
            "messages": 0,
            "bytes_worker_to_ps": 0,
            "bytes_ps_to_worker": 0,
            "full_gradient_messages": 0,
            "delta_gradient_messages": 0,
        }

    def register_worker(self, worker_id: str) -> dict:
        lists, shapes, b = weights_to_lists(self.global_weights)
        self.comm["messages"] += 1
        self.comm["bytes_ps_to_worker"] += b
        return {"weights": lists, "shapes": shapes, "version": self.version}

    def submit_gradients(
        self,
        worker_id: str,
        gradients_lists: list[list[float]],
        gradients_shapes: list[tuple[int, ...]],
        is_delta: bool,
    ) -> dict:
        payload, _, payload_bytes = weights_to_lists(lists_to_weights(gradients_lists, gradients_shapes))
        gradients = lists_to_weights(payload, gradients_shapes)

        if is_delta:
            previous = self.worker_last_full_gradient.get(worker_id)
            if previous is None:
                reconstructed = gradients
            else:
                reconstructed = add_weights(previous, gradients)
            self.worker_last_full_gradient[worker_id] = reconstructed
            self.comm["delta_gradient_messages"] += 1
        else:
            reconstructed = gradients
            self.worker_last_full_gradient[worker_id] = gradients
            self.comm["full_gradient_messages"] += 1

        self.global_weights = [w + (self.lr * g) for w, g in zip(self.global_weights, reconstructed)]
        self.version += 1

        w_lists, w_shapes, down_bytes = weights_to_lists(self.global_weights)
        self.comm["messages"] += 1
        self.comm["bytes_worker_to_ps"] += payload_bytes
        self.comm["bytes_ps_to_worker"] += down_bytes
        return {"weights": w_lists, "shapes": w_shapes, "version": self.version}

    def get_global_weights(self) -> dict:
        lists, shapes, b = weights_to_lists(self.global_weights)
        self.comm["messages"] += 1
        self.comm["bytes_ps_to_worker"] += b
        return {"weights": lists, "shapes": shapes, "version": self.version}

    def get_communication_metrics(self) -> dict:
        total = self.comm["bytes_worker_to_ps"] + self.comm["bytes_ps_to_worker"]
        return {**self.comm, "total_bytes": total, "total_mb": total / (1024 * 1024)}


@ray.remote
class DatasetServer:
    def __init__(self, partitions: Dict[int, List[Tuple[str, int]]], image_size: Tuple[int, int]):
        self.partitions = partitions
        self.image_size = image_size

    def partition_size(self, worker_idx: int) -> int:
        return len(self.partitions.get(worker_idx, []))

    def get_chunk(self, worker_idx: int, start: int, count: int) -> dict:
        part = self.partitions.get(worker_idx, [])
        if start >= len(part):
            return {"done": True, "x": None, "y": [], "count": 0}
        end = min(start + count, len(part))
        chunk = part[start:end]
        x, y = load_batch(chunk, self.image_size)
        return {"done": False, "x": x, "y": y, "count": int(end - start)}


@ray.remote
class Worker:
    def __init__(self, worker_id: str, worker_idx: int, partition: List[Tuple[str, int]], config: ExperimentConfig):
        self.worker_id = worker_id
        self.worker_idx = worker_idx
        self.partition = partition
        self.config = config
        input_shape = (config.image_size[0], config.image_size[1], 3)
        self.model = build_model(config.model_name, input_shape)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.last_sent_full_gradient: List[np.ndarray] | None = None
        self.local_steps = 0

    def _train_chunk(self, chunk: List[Tuple[str, int]]) -> List[np.ndarray]:
        before = [w.copy() for w in self.model.get_weights()]
        x, y = load_batch(chunk, self.config.image_size)

        with tf.GradientTape() as tape:
            preds = tf.reshape(self.model(x, training=True), [-1])
            loss = self.loss_fn(y, preds)
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        after = self.model.get_weights()
        return subtract_weights(after, before)

    def _train_arrays(self, x: np.ndarray, y: np.ndarray) -> List[np.ndarray]:
        before = [w.copy() for w in self.model.get_weights()]
        with tf.GradientTape() as tape:
            preds = tf.reshape(self.model(x, training=True), [-1])
            loss = self.loss_fn(y, preds)
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        after = self.model.get_weights()
        return subtract_weights(after, before)

    def train(self, ps: ray.actor.ActorHandle, dataset_server: ray.actor.ActorHandle | None = None) -> dict:
        reg = ray.get(ps.register_worker.remote(self.worker_id))
        self.model.set_weights(lists_to_weights(reg["weights"], reg["shapes"]))

        sync_n = max(1, int(self.config.sync_every_examples))
        if str(self.config.data_mode).lower() == "stream_from_head":
            part_size = int(ray.get(dataset_server.partition_size.remote(self.worker_idx))) if dataset_server else 0
        else:
            part_size = len(self.partition)

        for _ in range(int(self.config.epochs)):
            cursor = 0
            while cursor < part_size:
                if str(self.config.data_mode).lower() == "stream_from_head":
                    payload = ray.get(dataset_server.get_chunk.remote(self.worker_idx, cursor, sync_n)) if dataset_server else {"done": True}
                    if payload.get("done", False):
                        break
                    cursor += int(payload.get("count", 0))
                    grad = self._train_arrays(payload["x"], payload["y"])
                else:
                    end = min(cursor + sync_n, part_size)
                    chunk = self.partition[cursor:end]
                    cursor = end
                    if not chunk:
                        continue
                    grad = self._train_chunk(chunk)

                if self.last_sent_full_gradient is None:
                    payload = grad
                    is_delta = False
                    self.last_sent_full_gradient = grad
                else:
                    payload = subtract_weights(grad, self.last_sent_full_gradient)
                    is_delta = True
                    self.last_sent_full_gradient = grad

                lists, shapes, _ = weights_to_lists(payload)
                reply = ray.get(ps.submit_gradients.remote(self.worker_id, lists, shapes, is_delta))
                self.model.set_weights(lists_to_weights(reply["weights"], reply["shapes"]))
                self.local_steps += 1

        return {"worker_id": self.worker_id, "steps": self.local_steps, "samples_seen": part_size * int(self.config.epochs)}

