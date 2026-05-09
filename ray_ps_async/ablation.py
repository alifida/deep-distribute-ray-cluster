from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import Callable, Dict, List, Optional

from ray_ps_async.config import ExperimentConfig
from ray_ps_async.runner import run_experiment


def get_presets() -> Dict[str, Dict]:
    return {
        "baseline_async_ps": {
            "quantization": "fp32",
            "topk_pct": 100.0,
            "residual_feedback": False,
            "ssp_staleness": 9999,
            "quorum_fraction": 1.0,
            "round_timeout_ms": 1_000_000,
            "ps_optimizer": "sgd",
            "adaptive_sync": False,
        },
        "ssp_only": {
            "quantization": "fp32",
            "topk_pct": 100.0,
            "residual_feedback": False,
            "ssp_staleness": 2,
            "quorum_fraction": 1.0,
            "round_timeout_ms": 1_000_000,
            "ps_optimizer": "adam",
            "adaptive_sync": False,
        },
        "quorum_timeout": {
            "quantization": "fp32",
            "topk_pct": 100.0,
            "residual_feedback": False,
            "ssp_staleness": 2,
            "quorum_fraction": 0.6,
            "round_timeout_ms": 100,
            "ps_optimizer": "adam",
            "adaptive_sync": False,
        },
        "compression_topk": {
            "quantization": "topk",
            "topk_pct": 1.0,
            "residual_feedback": True,
            "ssp_staleness": 2,
            "quorum_fraction": 0.6,
            "round_timeout_ms": 100,
            "ps_optimizer": "adam",
            "adaptive_sync": False,
        },
        "full_system": {
            "quantization": "topk",
            "topk_pct": 1.0,
            "residual_feedback": True,
            "ssp_staleness": 2,
            "quorum_fraction": 0.6,
            "round_timeout_ms": 100,
            "ps_optimizer": "adam",
            "adaptive_sync": True,
        },
    }


def run_ablation_suite(
    base_cfg: ExperimentConfig,
    output_dir: str | None = None,
    should_continue: Optional[Callable[[], bool]] = None,
) -> Dict:
    presets = get_presets()
    rows: List[Dict] = []
    full_results: Dict[str, Dict] = {}
    aborted = False
    for name, overrides in presets.items():
        if should_continue is not None and not should_continue():
            aborted = True
            break
        cfg_dict = {**base_cfg.__dict__, **overrides}
        cfg = ExperimentConfig(**cfg_dict)
        result = run_experiment(cfg)
        full_results[name] = result
        tm = result.get("test_metrics", {})
        comm = result.get("communication_cost", {})
        rows.append(
            {
                "preset": name,
                "accuracy": tm.get("accuracy", 0.0),
                "precision": tm.get("precision", 0.0),
                "recall": tm.get("recall", 0.0),
                "f1_score": tm.get("f1_score", 0.0),
                "auc": tm.get("auc", 0.0),
                "total_mb": comm.get("total_mb", 0.0),
                "messages": comm.get("messages", 0),
            }
        )

    output: Dict = {"rows": rows, "results": full_results}
    if aborted:
        output["aborted"] = True
    if output_dir and rows:
        os.makedirs(output_dir, exist_ok=True)
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(output_dir, f"ablation_{stamp}.json")
        csv_path = os.path.join(output_dir, f"ablation_{stamp}.csv")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        output["artifacts"] = {"json": json_path, "csv": csv_path}
    return output
