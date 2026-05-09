from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from math import sqrt
from typing import Dict, List

import numpy as np

from ray_ps_async.ablation import run_ablation_suite
from ray_ps_async.config import ExperimentConfig


def _ci95(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(1.96 * np.std(arr, ddof=1) / sqrt(len(arr)))


def _to_md_table(rows: List[Dict]) -> str:
    if not rows:
        return "_No rows generated._"
    headers = list(rows[0].keys())
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(out)


def run_research_pack(
    base_cfg: ExperimentConfig,
    dataset_roots: List[str],
    repeats: int = 3,
    output_dir: str | None = None,
) -> Dict:
    datasets = dataset_roots or [base_cfg.dataset_root]
    repeats = max(1, int(repeats))

    all_runs: List[Dict] = []
    grouped: Dict[tuple[str, str], Dict[str, List[float]]] = {}

    for ds in datasets:
        for rep in range(repeats):
            cfg = ExperimentConfig(**{**base_cfg.__dict__, "dataset_root": ds, "random_seed": int(base_cfg.random_seed) + rep})
            out = run_ablation_suite(cfg, output_dir=None)
            for row in out.get("rows", []):
                preset = str(row.get("preset", "unknown"))
                run_entry = {"dataset_root": ds, "repeat": rep, **row}
                all_runs.append(run_entry)
                key = (ds, preset)
                if key not in grouped:
                    grouped[key] = {"accuracy": [], "f1_score": [], "auc": [], "total_mb": [], "messages": []}
                grouped[key]["accuracy"].append(float(row.get("accuracy", 0.0)))
                grouped[key]["f1_score"].append(float(row.get("f1_score", 0.0)))
                grouped[key]["auc"].append(float(row.get("auc", 0.0)))
                grouped[key]["total_mb"].append(float(row.get("total_mb", 0.0)))
                grouped[key]["messages"].append(float(row.get("messages", 0)))

    summary_rows: List[Dict] = []
    for (ds, preset), vals in sorted(grouped.items()):
        summary_rows.append(
            {
                "dataset_root": ds,
                "preset": preset,
                "accuracy_mean": round(float(np.mean(vals["accuracy"])), 6),
                "accuracy_ci95": round(_ci95(vals["accuracy"]), 6),
                "f1_mean": round(float(np.mean(vals["f1_score"])), 6),
                "f1_ci95": round(_ci95(vals["f1_score"]), 6),
                "auc_mean": round(float(np.mean(vals["auc"])), 6),
                "auc_ci95": round(_ci95(vals["auc"]), 6),
                "total_mb_mean": round(float(np.mean(vals["total_mb"])), 6),
                "messages_mean": round(float(np.mean(vals["messages"])), 2),
                "n_runs": len(vals["accuracy"]),
            }
        )

    best_by_dataset: Dict[str, Dict] = {}
    for row in summary_rows:
        ds = str(row["dataset_root"])
        if ds not in best_by_dataset or float(row["f1_mean"]) > float(best_by_dataset[ds]["f1_mean"]):
            best_by_dataset[ds] = row

    flat_rows = [
        {
            "preset": r["preset"],
            "accuracy": r["accuracy_mean"],
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": r["f1_mean"],
            "auc": r["auc_mean"],
            "total_mb": r["total_mb_mean"],
            "messages": r["messages_mean"],
        }
        for r in summary_rows
    ]

    report_md = [
        "# Research Evaluation Pack",
        "",
        f"- Generated at: {datetime.utcnow().isoformat()}Z",
        f"- Repeats per dataset: {repeats}",
        f"- Datasets: {len(datasets)}",
        "",
        "## Best preset per dataset (by mean F1)",
        "",
        _to_md_table(list(best_by_dataset.values())),
        "",
        "## Full aggregated summary (mean +/- CI95)",
        "",
        _to_md_table(summary_rows),
    ]

    output = {
        "datasets": datasets,
        "repeats": repeats,
        "rows": flat_rows,
        "summary_rows": summary_rows,
        "best_by_dataset": best_by_dataset,
        "all_runs": all_runs,
        "report_markdown": "\n".join(report_md),
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(output_dir, f"research_pack_{stamp}.json")
        runs_csv = os.path.join(output_dir, f"research_pack_runs_{stamp}.csv")
        summary_csv = os.path.join(output_dir, f"research_pack_summary_{stamp}.csv")
        md_path = os.path.join(output_dir, f"research_pack_{stamp}.md")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        with open(runs_csv, "w", encoding="utf-8", newline="") as f:
            if all_runs:
                writer = csv.DictWriter(f, fieldnames=list(all_runs[0].keys()))
                writer.writeheader()
                writer.writerows(all_runs)
        with open(summary_csv, "w", encoding="utf-8", newline="") as f:
            if summary_rows:
                writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(summary_rows)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(output["report_markdown"])
        output["artifacts"] = {
            "json": json_path,
            "runs_csv": runs_csv,
            "summary_csv": summary_csv,
            "markdown": md_path,
        }

    return output
