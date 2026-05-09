from __future__ import annotations

import math
import os
import socket
import time
from typing import Dict, List, Tuple

import numpy as np
import ray
import tensorflow as tf

from ray_ps_async.config import ExperimentConfig
from ray_ps_async.data import load_batch
from ray_ps_async.models import build_model
from ray_ps_async.serialization import (
    add_weights,
    decode_update,
    encode_update,
    subtract_weights,
    weights_to_lists,
    lists_to_weights,
    zeros_like_weights,
)


@ray.remote
class ParameterServer:
    def __init__(self, config: ExperimentConfig):
        input_shape = (config.image_size[0], config.image_size[1], 3)
        self.model = build_model(config.model_name, input_shape)
        self.global_weights = [w.astype(np.float32) for w in self.model.get_weights()]
        self.version = 0
        self.lr = float(config.learning_rate)
        self.ssp_staleness = max(0, int(getattr(config, "ssp_staleness", 2)))
        self.num_workers = max(1, int(getattr(config, "num_workers", 1)))
        qf = float(getattr(config, "quorum_fraction", 0.6))
        self.quorum_fraction = min(max(qf, 0.1), 1.0)
        self.round_timeout_ms = max(10, int(getattr(config, "round_timeout_ms", 100)))
        self.quorum_count = max(1, int(math.ceil(self.quorum_fraction * self.num_workers)))
        self.quantization = str(getattr(config, "quantization", "fp32")).lower()
        self.topk_pct = float(getattr(config, "topk_pct", 1.0))
        self.ps_optimizer = str(getattr(config, "ps_optimizer", "adam")).lower()
        self.ps_momentum = float(getattr(config, "ps_momentum", 0.9))
        self.opt_m = zeros_like_weights(self.global_weights)
        self.opt_v = zeros_like_weights(self.global_weights)
        self.opt_t = 0
        self.worker_last_full_gradient: Dict[str, List[np.ndarray]] = {}
        self.worker_handles: Dict[str, ray.actor.ActorHandle] = {}
        self.round_updates: List[List[np.ndarray]] = []
        self.round_started_at_ms: int | None = None
        self.comm = {
            "messages": 0,
            "bytes_worker_to_ps": 0,
            "bytes_ps_to_worker": 0,
            "full_gradient_messages": 0,
            "delta_gradient_messages": 0,
            "stale_rejections": 0,
            "quorum_commits": 0,
            "timeout_commits": 0,
            "pending_round_updates": 0,
            "push_broadcasts": 0,
        }

    def set_worker_handles(self, worker_ids: List[str], worker_handles: List[ray.actor.ActorHandle]) -> dict:
        self.worker_handles = {wid: wh for wid, wh in zip(worker_ids, worker_handles)}
        return {"status": "ok", "registered_workers": len(self.worker_handles)}

    def _broadcast_global_weights(self) -> None:
        if not self.worker_handles:
            return
        lists, shapes, _ = weights_to_lists(self.global_weights)
        for _, handle in self.worker_handles.items():
            try:
                # Fire-and-forget push update to reduce worker staleness between submits.
                handle.push_weights.remote(lists, shapes, self.version)
            except Exception:
                continue
        self.comm["push_broadcasts"] += 1

    def _commit_round(self, reason: str) -> None:
        if not self.round_updates:
            return
        agg = []
        num = float(len(self.round_updates))
        for layer_idx in range(len(self.round_updates[0])):
            layer_sum = np.zeros_like(self.round_updates[0][layer_idx], dtype=np.float32)
            for upd in self.round_updates:
                layer_sum += upd[layer_idx]
            agg.append(layer_sum / num)
        if self.ps_optimizer == "sgd":
            self.opt_m = [self.ps_momentum * m + g for m, g in zip(self.opt_m, agg)]
            self.global_weights = [w + (self.lr * m) for w, m in zip(self.global_weights, self.opt_m)]
        else:  # adam default
            self.opt_t += 1
            b1, b2, eps = 0.9, 0.999, 1e-8
            new_w = []
            new_m = []
            new_v = []
            for w, m, v, g in zip(self.global_weights, self.opt_m, self.opt_v, agg):
                m_t = (b1 * m) + ((1.0 - b1) * g)
                v_t = (b2 * v) + ((1.0 - b2) * (g * g))
                m_hat = m_t / (1.0 - (b1 ** self.opt_t))
                v_hat = v_t / (1.0 - (b2 ** self.opt_t))
                step = self.lr * m_hat / (np.sqrt(v_hat) + eps)
                new_w.append(w + step)
                new_m.append(m_t)
                new_v.append(v_t)
            self.global_weights = new_w
            self.opt_m = new_m
            self.opt_v = new_v
        self.version += 1
        if reason == "quorum":
            self.comm["quorum_commits"] += 1
        elif reason == "timeout":
            self.comm["timeout_commits"] += 1
        self._broadcast_global_weights()
        self.round_updates = []
        self.round_started_at_ms = None
        self.comm["pending_round_updates"] = 0

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
        gradients_meta: dict,
        is_delta: bool,
        base_version: int,
    ) -> dict:
        if int(base_version) < int(self.version) - int(self.ssp_staleness):
            # Reject stale update and force worker resync to latest global weights.
            w_lists, w_shapes, down_bytes = weights_to_lists(self.global_weights)
            self.comm["messages"] += 1
            self.comm["bytes_ps_to_worker"] += down_bytes
            self.comm["stale_rejections"] += 1
            return {
                "status": "stale",
                "resync_required": True,
                "weights": w_lists,
                "shapes": w_shapes,
                "version": self.version,
            }

        gradients = decode_update(gradients_lists, gradients_shapes, gradients_meta)
        payload_bytes = int(gradients_meta.get("tx_bytes", 0))

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

        now_ms = int(time.time() * 1000)
        if self.round_started_at_ms is None:
            self.round_started_at_ms = now_ms
        self.round_updates.append(reconstructed)
        self.comm["pending_round_updates"] = len(self.round_updates)
        elapsed_ms = now_ms - int(self.round_started_at_ms)

        if len(self.round_updates) >= self.quorum_count:
            self._commit_round("quorum")
        elif elapsed_ms >= self.round_timeout_ms:
            self._commit_round("timeout")

        w_lists, w_shapes, down_bytes = weights_to_lists(self.global_weights)
        self.comm["messages"] += 1
        self.comm["bytes_worker_to_ps"] += payload_bytes
        self.comm["bytes_ps_to_worker"] += down_bytes
        return {"status": "ok", "weights": w_lists, "shapes": w_shapes, "version": self.version}

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
        self.residual = [np.zeros_like(w, dtype=np.float32) for w in self.model.get_weights()]
        self.local_steps = 0
        self.quantization = str(getattr(config, "quantization", "fp32")).lower()
        self.topk_pct = float(getattr(config, "topk_pct", 1.0))
        self.residual_feedback = bool(getattr(config, "residual_feedback", True))
        self.adaptive_sync = bool(getattr(config, "adaptive_sync", True))
        self.stale_hits = 0
        self.pending_pushed_weights: tuple[list[list[float]], list[tuple[int, ...]], int] | None = None
        self.node_ip = ray.util.get_node_ip_address()
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        self.recent_logs: list[str] = []

    def _log(self, msg: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.recent_logs.append(f"[{stamp}] {self.worker_id}@{self.node_ip}: {msg}")
        if len(self.recent_logs) > 300:
            self.recent_logs = self.recent_logs[-300:]

    def get_recent_logs(self, limit: int = 80) -> dict:
        return {
            "worker_id": self.worker_id,
            "node_ip": self.node_ip,
            "hostname": self.hostname,
            "pid": self.pid,
            "local_steps": self.local_steps,
            "logs": self.recent_logs[-max(1, int(limit)):],
        }

    def push_weights(self, lists: list[list[float]], shapes: list[tuple[int, ...]], version: int) -> dict:
        # PS push-first path: enqueue latest global snapshot for immediate apply in train loop.
        self.pending_pushed_weights = (lists, shapes, int(version))
        return {"status": "ok"}

    def _apply_pending_push(self, local_version: int) -> int:
        if self.pending_pushed_weights is None:
            return local_version
        lists, shapes, version = self.pending_pushed_weights
        if int(version) > int(local_version):
            self.model.set_weights(lists_to_weights(lists, shapes))
            local_version = int(version)
            self.last_sent_full_gradient = None
        self.pending_pushed_weights = None
        return local_version

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
        local_version = int(reg.get("version", 0))
        self._log(f"training start (version={local_version})")

        sync_n = max(1, int(self.config.sync_every_examples))
        if str(self.config.data_mode).lower() == "stream_from_head":
            part_size = int(ray.get(dataset_server.partition_size.remote(self.worker_idx))) if dataset_server else 0
        else:
            part_size = len(self.partition)

        for _ in range(int(self.config.epochs)):
            cursor = 0
            while cursor < part_size:
                local_version = self._apply_pending_push(local_version)
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

                dense_payload = payload
                if self.quantization == "topk" and self.residual_feedback:
                    dense_payload = add_weights(dense_payload, self.residual)
                lists, shapes, meta, tx_bytes = encode_update(dense_payload, self.quantization, self.topk_pct)
                meta["tx_bytes"] = tx_bytes
                # update residual after top-k encoding/decoding roundtrip
                if self.quantization == "topk" and self.residual_feedback:
                    sparse_back = decode_update(lists, shapes, meta)
                    self.residual = subtract_weights(dense_payload, sparse_back)

                reply = ray.get(ps.submit_gradients.remote(self.worker_id, lists, shapes, meta, is_delta, local_version))
                if reply.get("status") == "stale":
                    # Reset to latest global state and restart delta chain.
                    self.model.set_weights(lists_to_weights(reply["weights"], reply["shapes"]))
                    local_version = int(reply.get("version", local_version))
                    self.last_sent_full_gradient = None
                    self.stale_hits += 1
                    self._log("stale update rejected by PS; resync applied")
                    if self.adaptive_sync:
                        sync_n = min(sync_n + 1, max(1, int(self.config.batch_size)))
                    continue
                self.model.set_weights(lists_to_weights(reply["weights"], reply["shapes"]))
                local_version = int(reply.get("version", local_version))
                if self.adaptive_sync and self.stale_hits == 0 and sync_n > 1:
                    sync_n = max(1, sync_n - 1)
                self.stale_hits = 0
                self.local_steps += 1
                if self.local_steps % 5 == 0:
                    self._log(f"progress steps={self.local_steps} version={local_version}")

        self._log(f"training completed steps={self.local_steps}")
        return {
            "worker_id": self.worker_id,
            "steps": self.local_steps,
            "samples_seen": part_size * int(self.config.epochs),
            "node_ip": self.node_ip,
            "hostname": self.hostname,
            "pid": self.pid,
        }

