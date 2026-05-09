from datetime import datetime, timezone
from threading import Lock
from typing import Callable, Dict, Literal
import json
import os
import uuid

from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExperimentRequest(BaseModel):
    dataset_root: str = Field(min_length=1)
    num_workers: int = Field(default=2, ge=1)
    epochs: int = Field(default=1, ge=1)
    batch_size: int = Field(default=8, ge=1)
    image_height: int = Field(default=224, ge=32)
    image_width: int = Field(default=224, ge=32)
    learning_rate: float = Field(default=1e-3, gt=0)
    sync_every_examples: int = Field(default=8, ge=1)
    num_gpus_per_worker: float = Field(default=0.0, ge=0.0)
    ray_address: Literal["local", "auto"] = "local"
    data_mode: Literal["shared_path", "stream_from_head"] = "shared_path"
    allow_gpu_fallback: bool = True
    ssp_staleness: int = Field(default=2, ge=0)
    quorum_fraction: float = Field(default=0.6, gt=0.0, le=1.0)
    round_timeout_ms: int = Field(default=100, ge=10)
    quantization: Literal["fp32", "fp16", "q8", "topk"] = "fp32"
    topk_pct: float = Field(default=1.0, gt=0.0, le=100.0)
    residual_feedback: bool = True
    ps_optimizer: Literal["adam", "sgd"] = "adam"
    ps_momentum: float = Field(default=0.9, ge=0.0, le=0.999)
    adaptive_sync: bool = True
    random_seed: int = Field(default=42)
    train_split: float = Field(default=0.8, gt=0.0, lt=1.0)
    val_split: float = Field(default=0.1, ge=0.0, lt=1.0)
    model_name: str = "tiny"
    dataset_roots_csv: str = ""
    benchmark_repeats: int = Field(default=3, ge=1, le=30)


class CompareRequest(BaseModel):
    plain_job_id: str = Field(min_length=1)
    custom_job_id: str = Field(min_length=1)


app = FastAPI(title="Ray Async Parameter Server Trainer")
JOBS: Dict[str, Dict] = {}
JOBS_LOCK = Lock()
JOBS_DB_PATH = os.path.join(os.path.dirname(__file__), "new_updates", "jobs_history.json")


def _ensure_ray_connected() -> tuple[bool, str]:
    try:
        import ray

        if not ray.is_initialized():
            ray.init(address="auto", ignore_reinit_error=True)
            return True, ""

        # If initialized session is stale, force reconnect so dashboard can recover.
        nodes = ray.nodes()
        alive_nodes = [n for n in nodes if bool(n.get("Alive", False))]
        if alive_nodes:
            return True, ""
        ray.shutdown()
        ray.init(address="auto", ignore_reinit_error=True)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _load_jobs_from_disk() -> Dict[str, Dict]:
    try:
        if not os.path.exists(JOBS_DB_PATH):
            return {}
        with open(JOBS_DB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_jobs_to_disk_locked() -> None:
    os.makedirs(os.path.dirname(JOBS_DB_PATH), exist_ok=True)
    tmp_path = f"{JOBS_DB_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(JOBS, f, indent=2)
    os.replace(tmp_path, JOBS_DB_PATH)


@app.on_event("startup")
def _startup_load_jobs() -> None:
    loaded = _load_jobs_from_disk()
    with JOBS_LOCK:
        JOBS.clear()
        JOBS.update(loaded)


def _experiment_runner():
    from ray_ps_async.runner import run_experiment

    return run_experiment


def _build_config(request: ExperimentRequest):
    from ray_ps_async.config import ExperimentConfig

    return ExperimentConfig(
        dataset_root=request.dataset_root,
        num_workers=request.num_workers,
        epochs=request.epochs,
        batch_size=request.batch_size,
        image_size=(request.image_height, request.image_width),
        learning_rate=request.learning_rate,
        sync_every_examples=request.sync_every_examples,
        num_gpus_per_worker=request.num_gpus_per_worker,
        ray_address=request.ray_address,
        data_mode=request.data_mode,
        allow_gpu_fallback=request.allow_gpu_fallback,
        ssp_staleness=request.ssp_staleness,
        quorum_fraction=request.quorum_fraction,
        round_timeout_ms=request.round_timeout_ms,
        quantization=request.quantization,
        topk_pct=request.topk_pct,
        residual_feedback=request.residual_feedback,
        ps_optimizer=request.ps_optimizer,
        ps_momentum=request.ps_momentum,
        adaptive_sync=request.adaptive_sync,
        random_seed=request.random_seed,
        train_split=request.train_split,
        val_split=request.val_split,
        model_name=request.model_name,
    )


def _apply_strategy_to_request(request: ExperimentRequest, strategy_type: str) -> ExperimentRequest:
    if strategy_type != "plain_ps":
        return request
    # Plain PS baseline mapping: disable novelty-specific knobs.
    return request.model_copy(
        update={
            "ssp_staleness": 9999,
            "quorum_fraction": 1.0,
            "round_timeout_ms": 1_000_000,
            "quantization": "fp32",
            "topk_pct": 100.0,
            "residual_feedback": False,
            "ps_optimizer": "sgd",
            "adaptive_sync": False,
        }
    )


def _run_job(job_id: str, request: ExperimentRequest, strategy_type: str = "custom_strategy") -> None:
    with JOBS_LOCK:
        if job_id not in JOBS:
            return
        JOBS[job_id]["status"] = "running"
        JOBS[job_id]["started_at"] = _now_iso()
        _save_jobs_to_disk_locked()
    try:
        runner: Callable = _experiment_runner()
        cfg = _build_config(_apply_strategy_to_request(request, strategy_type))
        setattr(cfg, "run_id", job_id)
        result = runner(cfg)
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "completed"
            JOBS[job_id]["result"] = result
            JOBS[job_id]["finished_at"] = _now_iso()
            _save_jobs_to_disk_locked()
    except Exception as exc:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = str(exc)
            JOBS[job_id]["finished_at"] = _now_iso()
            _save_jobs_to_disk_locked()


def _compare_pair(plain_job: Dict, custom_job: Dict) -> Dict:
    plain_res = plain_job.get("result") or {}
    custom_res = custom_job.get("result") or {}
    plain_tm = plain_res.get("test_metrics") or {}
    custom_tm = custom_res.get("test_metrics") or {}
    plain_comm = plain_res.get("communication_cost") or {}
    custom_comm = custom_res.get("communication_cost") or {}

    def dval(a: float, b: float) -> float:
        return float(b) - float(a)

    metrics = {}
    for key in ["accuracy", "precision", "recall", "f1_score", "auc"]:
        metrics[key] = {
            "plain_ps": float(plain_tm.get(key, 0.0)),
            "custom_strategy": float(custom_tm.get(key, 0.0)),
            "delta_custom_minus_plain": dval(plain_tm.get(key, 0.0), custom_tm.get(key, 0.0)),
        }

    comm = {
        "total_mb": {
            "plain_ps": float(plain_comm.get("total_mb", 0.0)),
            "custom_strategy": float(custom_comm.get("total_mb", 0.0)),
            "delta_custom_minus_plain": dval(plain_comm.get("total_mb", 0.0), custom_comm.get("total_mb", 0.0)),
        },
        "messages": {
            "plain_ps": float(plain_comm.get("messages", 0.0)),
            "custom_strategy": float(custom_comm.get("messages", 0.0)),
            "delta_custom_minus_plain": dval(plain_comm.get("messages", 0.0), custom_comm.get("messages", 0.0)),
        },
    }

    plain_req = plain_job.get("request") or {}
    custom_req = custom_job.get("request") or {}
    compatibility = {
        "same_dataset_root": plain_req.get("dataset_root") == custom_req.get("dataset_root"),
        "same_model_name": plain_req.get("model_name") == custom_req.get("model_name"),
        "same_image_size": (
            plain_req.get("image_height") == custom_req.get("image_height")
            and plain_req.get("image_width") == custom_req.get("image_width")
        ),
        "same_train_val_split": (
            plain_req.get("train_split") == custom_req.get("train_split")
            and plain_req.get("val_split") == custom_req.get("val_split")
        ),
    }
    compatibility["strictly_comparable"] = all(compatibility.values())

    return {"metrics": metrics, "communication": comm, "compatibility": compatibility}


def _run_ablation(job_id: str, request: ExperimentRequest) -> None:
    with JOBS_LOCK:
        if job_id not in JOBS:
            return
        JOBS[job_id]["status"] = "running"
        JOBS[job_id]["started_at"] = _now_iso()
        _save_jobs_to_disk_locked()
    try:
        from ray_ps_async.ablation import run_ablation_suite

        cfg = _build_config(request)
        result = run_ablation_suite(cfg, output_dir=os.path.join(os.getcwd(), "new_updates"))
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "completed"
            JOBS[job_id]["result"] = result
            JOBS[job_id]["finished_at"] = _now_iso()
            _save_jobs_to_disk_locked()
    except Exception as exc:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = str(exc)
            JOBS[job_id]["finished_at"] = _now_iso()
            _save_jobs_to_disk_locked()


def _run_research_pack(job_id: str, request: ExperimentRequest) -> None:
    with JOBS_LOCK:
        if job_id not in JOBS:
            return
        JOBS[job_id]["status"] = "running"
        JOBS[job_id]["started_at"] = _now_iso()
        _save_jobs_to_disk_locked()
    try:
        from ray_ps_async.research_pack import run_research_pack

        cfg = _build_config(request)
        dataset_roots = [x.strip() for x in str(request.dataset_roots_csv).split(",") if x.strip()]
        result = run_research_pack(
            cfg,
            dataset_roots=dataset_roots or [request.dataset_root],
            repeats=int(request.benchmark_repeats),
            output_dir=os.path.join(os.getcwd(), "new_updates"),
        )
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "completed"
            JOBS[job_id]["result"] = result
            JOBS[job_id]["finished_at"] = _now_iso()
            _save_jobs_to_disk_locked()
    except Exception as exc:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = str(exc)
            JOBS[job_id]["finished_at"] = _now_iso()
            _save_jobs_to_disk_locked()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/cluster/resources")
def cluster_resources() -> Dict:
    try:
        import ray
        ok, err = _ensure_ray_connected()
        if not ok:
            return {"status": "not_connected", "error": err, "resources": {}}
        return {"status": "ok", "resources": ray.available_resources()}
    except Exception as exc:
        return {"status": "error", "error": str(exc), "resources": {}}


@app.get("/cluster/nodes")
def cluster_nodes() -> Dict:
    try:
        import ray
        ok, err = _ensure_ray_connected()
        if not ok:
            return {"status": "not_connected", "error": err, "nodes": []}

        out = []
        for n in ray.nodes():
            resources = n.get("Resources", {})
            out.append(
                {
                    "alive": bool(n.get("Alive", False)),
                    "node_id": n.get("NodeID"),
                    "node_manager_address": n.get("NodeManagerAddress"),
                    "cpus": float(resources.get("CPU", 0.0)),
                    "gpus": float(resources.get("GPU", 0.0)),
                }
            )
        return {"status": "ok", "nodes": out}
    except Exception as exc:
        return {"status": "error", "error": str(exc), "nodes": []}


@app.post("/cluster/reconnect")
def cluster_reconnect() -> Dict:
    try:
        import ray

        if ray.is_initialized():
            try:
                ray.shutdown()
            except Exception:
                pass
        ray.init(address="auto", ignore_reinit_error=True)
        return {"status": "ok", "resources": ray.available_resources()}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


@app.get("/experiments/presets")
def experiment_presets() -> Dict:
    from ray_ps_async.ablation import get_presets

    return {"status": "ok", "presets": get_presets()}


@app.get("/artifacts/ablation/latest")
def latest_ablation_artifacts() -> Dict:
    out_dir = os.path.join(os.getcwd(), "new_updates")
    if not os.path.isdir(out_dir):
        return {"status": "not_found"}
    files = []
    for name in os.listdir(out_dir):
        if name.startswith("ablation_") and (name.endswith(".csv") or name.endswith(".json")):
            path = os.path.join(out_dir, name)
            files.append((os.path.getmtime(path), name, path))
    if not files:
        return {"status": "not_found"}
    files.sort(key=lambda x: x[0], reverse=True)
    latest_json = next(({"name": n, "path": p} for _, n, p in files if n.endswith(".json")), None)
    latest_csv = next(({"name": n, "path": p} for _, n, p in files if n.endswith(".csv")), None)
    return {"status": "ok", "json": latest_json, "csv": latest_csv}


@app.get("/artifacts/ablation/download/{fmt}")
def download_latest_ablation(fmt: str):
    fmt = str(fmt).lower()
    if fmt not in {"csv", "json"}:
        return {"status": "error", "error": "fmt must be csv or json"}
    out_dir = os.path.join(os.getcwd(), "new_updates")
    if not os.path.isdir(out_dir):
        return {"status": "not_found"}
    candidates = []
    for name in os.listdir(out_dir):
        if name.startswith("ablation_") and name.endswith(f".{fmt}"):
            path = os.path.join(out_dir, name)
            candidates.append((os.path.getmtime(path), name, path))
    if not candidates:
        return {"status": "not_found"}
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, name, path = candidates[0]
    return FileResponse(path, media_type="application/octet-stream", filename=name)


@app.get("/artifacts/research-pack/latest")
def latest_research_pack_artifacts() -> Dict:
    out_dir = os.path.join(os.getcwd(), "new_updates")
    if not os.path.isdir(out_dir):
        return {"status": "not_found"}
    groups = {"json": "research_pack_", "markdown": "research_pack_", "summary_csv": "research_pack_summary_", "runs_csv": "research_pack_runs_"}
    found: Dict[str, Dict] = {}
    for key, prefix in groups.items():
        ext = ".md" if key == "markdown" else ".csv" if "csv" in key else ".json"
        candidates = []
        for name in os.listdir(out_dir):
            if name.startswith(prefix) and name.endswith(ext):
                path = os.path.join(out_dir, name)
                candidates.append((os.path.getmtime(path), name, path))
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            _, name, path = candidates[0]
            found[key] = {"name": name, "path": path}
    if not found:
        return {"status": "not_found"}
    return {"status": "ok", **found}


@app.get("/artifacts/research-pack/download/{kind}")
def download_latest_research_pack(kind: str):
    kind = str(kind).lower()
    mapping = {
        "json": ("research_pack_", ".json"),
        "markdown": ("research_pack_", ".md"),
        "summary_csv": ("research_pack_summary_", ".csv"),
        "runs_csv": ("research_pack_runs_", ".csv"),
    }
    if kind not in mapping:
        return {"status": "error", "error": "kind must be json|markdown|summary_csv|runs_csv"}
    prefix, suffix = mapping[kind]
    out_dir = os.path.join(os.getcwd(), "new_updates")
    if not os.path.isdir(out_dir):
        return {"status": "not_found"}
    candidates = []
    for name in os.listdir(out_dir):
        if name.startswith(prefix) and name.endswith(suffix):
            path = os.path.join(out_dir, name)
            candidates.append((os.path.getmtime(path), name, path))
    if not candidates:
        return {"status": "not_found"}
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, name, path = candidates[0]
    return FileResponse(path, media_type="application/octet-stream", filename=name)


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Ray Async Trainer Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body { font-family: Inter, Arial, sans-serif; margin: 18px; background: #f3f5f9; color: #111827; }
    h1 { margin: 0 0 6px 0; font-size: 26px; }
    h3 { margin: 0 0 10px 0; color: #0f172a; }
    h4 { margin: 8px 0; color: #1e293b; }
    .muted { color: #64748b; margin-bottom: 14px; }
    .section-title { font-size: 14px; font-weight: 700; letter-spacing: .03em; color: #334155; margin: 18px 0 10px 2px; text-transform: uppercase; }
    .card { background: #fff; border: 1px solid #dbe5f3; border-radius: 12px; padding: 14px; margin-bottom: 12px; box-shadow: 0 2px 6px rgba(15,23,42,.05); }
    .accordion-header-btn {
      width: 100%;
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: transparent;
      color: #0f172a;
      border: none;
      padding: 0;
      margin: 0 0 8px 0;
      font-size: 16px;
      font-weight: 700;
      cursor: pointer;
      text-align: left;
    }
    .accordion-header-btn:hover { background: transparent; color: #0f172a; }
    .accordion-icon {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 18px;
      height: 18px;
      border-radius: 999px;
      border: 1px solid #cbd5e1;
      color: #334155;
      font-size: 13px;
      font-weight: 700;
      line-height: 1;
      margin-left: 10px;
      flex-shrink: 0;
    }
    .card.collapsed .accordion-content { display: none; }
    .top-grid { display: grid; grid-template-columns: repeat(4, minmax(220px, 1fr)); gap: 10px; margin-bottom: 12px; }
    .main-two-col { display: grid; grid-template-columns: 1.05fr 0.95fr; gap: 12px; align-items: start; }
    .left-panel, .right-panel { display: flex; flex-direction: column; gap: 10px; }
    .row { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 10px; }
    .field { display: flex; flex-direction: column; min-width: 180px; }
    .field-group-card { border: 1px solid #e2e8f0; border-radius: 10px; padding: 10px; margin-bottom: 10px; background: #fafcff; }
    .button-group { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
    .btn-plain { background: #0f766e; }
    .btn-plain:hover { background: #115e59; }
    .btn-custom { background: #1d4ed8; }
    .btn-custom:hover { background: #1e40af; }
    .btn-ablation { background: #7c3aed; }
    .btn-ablation:hover { background: #6d28d9; }
    .btn-research { background: #334155; }
    .btn-research:hover { background: #1e293b; }
    .icon-btn {
      padding: 9px 12px;
      min-width: 38px;
      font-size: 17px;
      line-height: 1.1;
      border-radius: 6px;
      background: #1d4ed8;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }
    .icon-btn.delete { background: #dc2626; }
    .icon-btn.delete:hover { background: #b91c1c; }
    .action-icons { display: inline-flex; align-items: center; gap: 4px; white-space: nowrap; }
    .icon-btn.refresh { background: #475569; }
    .icon-btn.refresh:hover { background: #334155; }
    input { padding: 8px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 13px; }
    select { padding: 8px; border: 1px solid #cbd5e1; border-radius: 6px; background: #fff; font-size: 13px; }
    button { padding: 9px 12px; border: none; border-radius: 7px; background: #1d4ed8; color: #fff; cursor: pointer; font-weight: 600; }
    button:hover { background: #1e40af; }
    .kpi { font-size: 18px; font-weight: 700; }
    .grid3 { display: grid; grid-template-columns: repeat(2,minmax(95px,1fr)); gap: 8px; }
    .pill { display: inline-block; padding: 3px 8px; border-radius: 999px; font-size: 12px; font-weight: 700; }
    .pill-queued { background: #eef2ff; color: #3730a3; }
    .pill-running { background: #fff7ed; color: #b45309; }
    .pill-completed { background: #ecfdf5; color: #166534; }
    .pill-failed { background: #fef2f2; color: #b91c1c; }
    .charts { display: grid; grid-template-columns: repeat(auto-fit,minmax(320px,1fr)); gap: 12px; }
    canvas { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 8px; max-height: 180px; }
    .small { font-size: 12px; color: #555; }
    .label-wrap { display: flex; align-items: center; gap: 6px; }
    .hint {
      display: inline-block;
      width: 16px;
      height: 16px;
      border-radius: 999px;
      background: #dbeafe;
      color: #1d4ed8;
      text-align: center;
      line-height: 16px;
      font-size: 12px;
      cursor: help;
      position: relative;
      font-weight: 700;
    }
    .hint:hover::after {
      content: attr(data-tip);
      position: absolute;
      left: 20px;
      top: -6px;
      width: 260px;
      background: #0f172a;
      color: #fff;
      border-radius: 6px;
      padding: 8px;
      font-size: 12px;
      line-height: 1.3;
      z-index: 100;
      white-space: normal;
    }
    pre { background: #0b1020; color: #d8e1ff; padding: 10px; border-radius: 8px; overflow: auto; max-height: 280px; font-size: 12px; }
    #cluster_resources { max-height: 150px; font-size: 11px; }
    table { width: 100%; border-collapse: collapse; }
    th, td { border-bottom: 1px solid #e5e7eb; text-align: left; padding: 7px; font-size: 12px; }
    .compact-scroll { max-height: 170px; overflow: auto; border: 1px solid #e5e7eb; border-radius: 8px; }
    .table-scroll { max-height: 260px; overflow: auto; border: 1px solid #e5e7eb; border-radius: 8px; }
    .conn-banner { border-radius: 8px; padding: 10px 12px; margin-bottom: 12px; font-size: 13px; }
    .conn-ok { background: #ecfdf5; color: #166534; border: 1px solid #86efac; }
    .conn-warn { background: #fff7ed; color: #9a3412; border: 1px solid #fdba74; }
    .conn-bad { background: #fef2f2; color: #b91c1c; border: 1px solid #fca5a5; }
  </style>
</head>
<body>
  <h1>Ray Async Parameter-Server Dashboard</h1>
  <div class="muted">A professional, step-wise dashboard for training, comparing, and reporting experiments.</div>

  <div class="section-title">Step 0 - Cluster And Runtime Status</div>
  <div class="top-grid">
    <div class="card">
      <h3>Connect Ray</h3>
      <div id="conn_banner" class="conn-banner conn-ok">Backend status: checking...</div>
      <button onclick="reconnectRay()">Reconnect Ray</button>
      <span class="hint" data-tip="Forces backend to detach and re-attach to Ray cluster (address=auto) without restarting the UI server.">i</span>
    </div>
    <div class="card">
      <h3>Job Overview</h3>
      <div class="grid3">
        <div><div class="small">Queued</div><div id="kpi_queued" class="kpi">0</div></div>
        <div><div class="small">Running</div><div id="kpi_running" class="kpi">0</div></div>
        <div><div class="small">Completed</div><div id="kpi_completed" class="kpi">0</div></div>
        <div><div class="small">Failed</div><div id="kpi_failed" class="kpi">0</div></div>
      </div>
    </div>
    <div class="card">
      <h3>Cluster Resources</h3>
      <button class="icon-btn refresh" title="Refresh resources" aria-label="Refresh resources" onclick="refreshResources()">⟳</button>
      <pre id="cluster_resources">Not loaded yet...</pre>
    </div>
    <div class="card">
      <h3>Connected Nodes</h3>
      <button class="icon-btn refresh" title="Refresh nodes" aria-label="Refresh nodes" onclick="refreshNodes()">⟳</button>
      <div class="compact-scroll">
      <table>
      <thead>
        <tr><th>Address</th><th>Alive</th><th>CPU</th><th>GPU</th><th>Node ID</th></tr>
      </thead>
      <tbody id="nodes_body"></tbody>
      </table>
      </div>
    </div>
  </div>

  <div class="main-two-col">
    <div class="left-panel">
      <div class="section-title">Step 1 - Configure And Launch Training</div>
      <div class="card">
        <h3>Training Configuration</h3>
        <div class="small" style="margin-bottom:8px;">Fields are grouped for cleaner setup and easier flow.</div>

        <div class="field-group-card">
          <h4>Core Setup</h4>
          <div class="row">
            <div class="field"><label class="label-wrap">Dataset Root <span class="hint" data-tip="Root path of class-wise dataset. Used by head to prepare splits and partitions.">i</span></label><input id="dataset_root" placeholder="/path/to/dataset_root" /></div>
            <div class="field"><label class="label-wrap">Ray Address <span class="hint" data-tip="local: run on same machine. auto: connect to existing Ray cluster (LAN).">i</span></label><select id="ray_address"><option value="auto" selected>auto</option><option value="local">local</option></select></div>
            <div class="field"><label class="label-wrap">Data Mode <span class="hint" data-tip="shared_path: workers read dataset path directly. stream_from_head: head sends each worker subset at runtime.">i</span></label><select id="data_mode"><option value="shared_path" selected>shared_path</option><option value="stream_from_head">stream_from_head</option></select></div>
            <div class="field"><label class="label-wrap">Workers <span class="hint" data-tip="Number of distributed worker actors to start for this job.">i</span></label><input id="num_workers" type="number" value="2" /></div>
            <div class="field"><label class="label-wrap">Epochs <span class="hint" data-tip="How many passes each worker performs over its assigned training partition.">i</span></label><input id="epochs" type="number" value="1" /></div>
            <div class="field"><label class="label-wrap">Batch Size <span class="hint" data-tip="Mini-batch size used inside each local training step.">i</span></label><input id="batch_size" type="number" value="8" /></div>
            <div class="field"><label class="label-wrap">Sync Every Examples <span class="hint" data-tip="After this many examples, worker sends update to PS and pulls latest global weights.">i</span></label><input id="sync_every_examples" type="number" value="8" /></div>
            <div class="field"><label class="label-wrap">Model Name <span class="hint" data-tip="Model architecture key used to build network (e.g., tiny).">i</span></label><input id="model_name" value="tiny" /></div>
          </div>
        </div>

        <div class="field-group-card">
          <h4>Resources And Optimization</h4>
          <div class="row">
            <div class="field"><label class="label-wrap">Image Height <span class="hint" data-tip="Input image resize height used before feeding the model.">i</span></label><input id="image_height" type="number" value="224" /></div>
            <div class="field"><label class="label-wrap">Image Width <span class="hint" data-tip="Input image resize width used before feeding the model.">i</span></label><input id="image_width" type="number" value="224" /></div>
            <div class="field"><label class="label-wrap">Learning Rate <span class="hint" data-tip="Optimizer step size used by workers in local updates.">i</span></label><input id="learning_rate" type="number" step="0.0001" value="0.001" /></div>
            <div class="field"><label class="label-wrap">GPUs per Worker <span class="hint" data-tip="GPU resource requested per worker actor. Use 1 for one-GPU-per-worker.">i</span></label><input id="num_gpus_per_worker" type="number" step="0.5" value="1" /></div>
            <div class="field"><label class="label-wrap">Allow GPU Fallback <span class="hint" data-tip="If true, job falls back to CPU when required GPUs are unavailable.">i</span></label><select id="allow_gpu_fallback"><option value="true" selected>true</option><option value="false">false</option></select></div>
            <div class="field"><label class="label-wrap">Random Seed <span class="hint" data-tip="Controls reproducibility for data shuffle and model randomness across runs.">i</span></label><input id="random_seed" type="number" value="42" /></div>
          </div>
        </div>

        <div class="field-group-card">
          <h4>Dataset And Evaluation</h4>
          <div class="row">
            <div class="field"><label class="label-wrap">Train Split <span class="hint" data-tip="Fraction of each class used for worker training partitions.">i</span></label><input id="train_split" type="number" step="0.01" value="0.8" /></div>
            <div class="field"><label class="label-wrap">Validation Split <span class="hint" data-tip="Fraction used for validation after training. Test split is computed from remainder.">i</span></label><input id="val_split" type="number" step="0.01" value="0.1" /></div>
            <div class="field"><label class="label-wrap">Benchmark Repeats <span class="hint" data-tip="Used by research pack: number of repeated runs per dataset for CI estimation.">i</span></label><input id="benchmark_repeats" type="number" value="3" min="1" max="30" /></div>
            <div class="field"><label class="label-wrap">Dataset Roots CSV <span class="hint" data-tip="Optional: comma-separated dataset roots for multi-dataset validation pack.">i</span></label><input id="dataset_roots_csv" placeholder="/path/ds1,/path/ds2" /></div>
          </div>
        </div>

        <div class="field-group-card">
          <h4>Custom Strategy Parameters</h4>
          <div class="row">
            <div class="field"><label class="label-wrap">SSP Staleness <span class="hint" data-tip="Maximum version lag allowed for worker updates. Older updates are rejected and worker must resync.">i</span></label><input id="ssp_staleness" type="number" value="2" min="0" /></div>
            <div class="field"><label class="label-wrap">Quorum Fraction <span class="hint" data-tip="Micro-round commit threshold as fraction of workers (e.g., 0.6 means commit after ~60% updates).">i</span></label><input id="quorum_fraction" type="number" step="0.01" value="0.6" min="0.01" max="1" /></div>
            <div class="field"><label class="label-wrap">Round Timeout (ms) <span class="hint" data-tip="If quorum is not reached, PS commits round once this timeout expires.">i</span></label><input id="round_timeout_ms" type="number" value="100" min="10" /></div>
            <div class="field"><label class="label-wrap">Quantization <span class="hint" data-tip="Update encoding method: fp32/fp16/q8/topk to reduce payload size.">i</span></label><select id="quantization"><option value="fp32">fp32</option><option value="fp16">fp16</option><option value="q8">q8</option><option value="topk">topk</option></select></div>
            <div class="field"><label class="label-wrap">Top-k % <span class="hint" data-tip="When quantization=topk, only this percentage of largest-magnitude elements is transmitted.">i</span></label><input id="topk_pct" type="number" step="0.1" value="1.0" min="0.1" max="100" /></div>
            <div class="field"><label class="label-wrap">Residual Feedback <span class="hint" data-tip="Carries top-k compression error into next update to preserve convergence.">i</span></label><select id="residual_feedback"><option value="true" selected>true</option><option value="false">false</option></select></div>
            <div class="field"><label class="label-wrap">PS Optimizer <span class="hint" data-tip="Central optimizer running at PS for applying aggregated updates.">i</span></label><select id="ps_optimizer"><option value="adam" selected>adam</option><option value="sgd">sgd</option></select></div>
            <div class="field"><label class="label-wrap">PS Momentum <span class="hint" data-tip="Momentum factor used when PS optimizer is SGD.">i</span></label><input id="ps_momentum" type="number" step="0.01" value="0.9" min="0" max="0.999" /></div>
            <div class="field"><label class="label-wrap">Adaptive Sync <span class="hint" data-tip="Automatically adapts sync interval when staleness/rejections are observed.">i</span></label><select id="adaptive_sync"><option value="true" selected>true</option><option value="false">false</option></select></div>
          </div>
        </div>

        <div class="button-group">
          <button class="btn-plain" onclick="startPlainTraining()">Train with Plain PS</button>
          <button class="btn-custom" onclick="startCustomTraining()">Train with Our Custom Strategy</button>
          <button class="btn-ablation" onclick="startAblation()">Run Ablation Suite</button>
          <button class="btn-research" onclick="startResearchPack()">Run Research Pack</button>
        </div>
        <p id="start_msg"></p>
      </div>

      <div class="section-title">Step 2 - Monitor And Select Jobs</div>
      <div class="card">
        <h3>Jobs And Strategy Selection</h3>
        <button class="icon-btn refresh" title="Refresh jobs" aria-label="Refresh jobs" onclick="refreshJobs()">⟳</button>
        <button onclick="deleteFinishedJobs()">Delete Finished Jobs</button>
        <div class="table-scroll" style="margin-top:8px;">
          <table>
            <thead>
              <tr><th>Job ID</th><th>Status</th><th>Type</th><th>Strategy</th><th>Created</th><th>Finished</th><th>Action</th></tr>
            </thead>
            <tbody id="jobs_body"></tbody>
          </table>
        </div>
        <div class="small" style="margin-top:10px;">Pick one completed Plain PS and one completed Custom job, then compare.</div>
        <div class="row" style="margin-top:8px;">
          <div style="flex:1; min-width:260px;">
            <h4>Plain PS</h4>
            <div class="table-scroll" style="max-height:150px;">
              <table>
                <thead><tr><th>Select</th><th>Job ID</th><th>Finished</th></tr></thead>
                <tbody id="plain_jobs_body"></tbody>
              </table>
            </div>
          </div>
          <div style="flex:1; min-width:260px;">
            <h4>Custom Strategy</h4>
            <div class="table-scroll" style="max-height:150px;">
              <table>
                <thead><tr><th>Select</th><th>Job ID</th><th>Finished</th></tr></thead>
                <tbody id="custom_jobs_body"></tbody>
              </table>
            </div>
          </div>
        </div>
        <div style="margin-top:10px;">
          <button onclick="compareSelectedRuns()">Compare Selected Plain vs Custom</button>
          <span id="compare_msg" class="small"></span>
        </div>
      </div>
    </div>

    <div class="right-panel">
      <div class="section-title">Step 3 - Results And Analysis</div>
      <div class="card" id="insights_card" style="display:none;">
        <h3>Selected Job Insights</h3>
        <div id="result_summary" class="small">Click View on a job to load summary and charts.</div>
        <div class="charts" style="margin-top:8px;">
          <div><h4>Model Metrics</h4><canvas id="metrics_chart" height="130"></canvas></div>
          <div><h4>Communication Cost (MB)</h4><canvas id="comm_chart" height="130"></canvas></div>
        </div>
      </div>

      <div class="card" id="ablation_card" style="display:none;">
        <h3>Ablation Comparison</h3>
        <div class="small">Visible when an ablation job is selected.</div>
        <div style="margin:10px 0;">
          <button onclick="downloadAblationArtifact('csv')">Download Latest CSV</button>
          <button onclick="downloadAblationArtifact('json')">Download Latest JSON</button>
          <span id="ablation_download_msg" class="small"></span>
        </div>
        <table>
          <thead><tr><th>Preset</th><th>Accuracy</th><th>F1</th><th>AUC</th><th>Total MB</th><th>Messages</th></tr></thead>
          <tbody id="ablation_body"><tr><td colspan="6">No ablation result selected.</td></tr></tbody>
        </table>
        <div style="margin-top:12px;"><canvas id="ablation_chart" height="130"></canvas></div>
        <div style="margin-top:10px;">
          <button onclick="downloadResearchArtifact('markdown')">Download Latest Research MD</button>
          <button onclick="downloadResearchArtifact('summary_csv')">Download Latest Research Summary CSV</button>
          <button onclick="downloadResearchArtifact('runs_csv')">Download Latest Research Runs CSV</button>
          <span id="research_download_msg" class="small"></span>
        </div>
      </div>

      <div class="card" id="pair_card" style="display:none;">
        <h3>Selected Pair Comparison</h3>
        <div id="pair_compare_summary" class="small">Select one Plain PS and one Custom completed job, then click Compare.</div>
        <table style="margin-top:10px;"><thead><tr><th>Metric</th><th>Plain PS</th><th>Custom Strategy</th><th>Delta (Custom - Plain)</th></tr></thead><tbody id="pair_metrics_body"><tr><td colspan="4">No comparison yet.</td></tr></tbody></table>
        <table style="margin-top:10px;"><thead><tr><th>Communication</th><th>Plain PS</th><th>Custom Strategy</th><th>Delta (Custom - Plain)</th></tr></thead><tbody id="pair_comm_body"><tr><td colspan="4">No comparison yet.</td></tr></tbody></table>
        <div class="charts" style="margin-top:12px;">
          <div><h4>Pair Metrics Comparison</h4><canvas id="pair_metrics_chart" height="130"></canvas></div>
          <div><h4>Pair Communication Comparison</h4><canvas id="pair_comm_chart" height="130"></canvas></div>
        </div>
        <pre id="pair_compare_details">Raw comparison payload will appear here.</pre>
      </div>

      <div class="card" id="worker_logs_card" style="display:none;">
        <h3>Live Worker Logs</h3>
        <div class="small">Use the running-job logs button (📡) to stream active worker logs.</div>
        <div style="margin:8px 0;">
          <button onclick="refreshLiveLogsNow()">Refresh Live Logs</button>
          <span id="worker_logs_msg" class="small"></span>
        </div>
        <pre id="worker_logs_view">No live logs selected.</pre>
      </div>

      <div class="card" id="job_details_card" style="display:none;">
        <h3>Job Details</h3>
        <pre id="job_details">Select a job to see details...</pre>
      </div>
    </div>
  </div>

  <script>
    function setBanner(state, text) {
      const el = document.getElementById("conn_banner");
      el.className = "conn-banner " + (state === "ok" ? "conn-ok" : state === "warn" ? "conn-warn" : "conn-bad");
      el.textContent = text;
    }

    async function safeFetch(url, options = {}) {
      try {
        const r = await fetch(url, options);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        setBanner("ok", "Backend status: connected");
        return await r.json();
      } catch (e) {
        setBanner("bad", `Backend status: disconnected (${e.message}). Check start_ui.sh and Ray head.`);
        throw e;
      }
    }

    async function probeConnection() {
      try {
        const h = await safeFetch("/health");
        if (h.status === "ok") setBanner("ok", "Backend status: connected");
      } catch (_) {
        // banner already updated in safeFetch
      }
    }

    async function reconnectRay() {
      try {
        const r = await safeFetch("/cluster/reconnect", {method: "POST"});
        if (r.status === "ok") {
          setBanner("ok", "Backend status: connected (Ray reconnected)");
          await refreshResources();
          await refreshNodes();
        } else {
          setBanner("warn", "Reconnect attempted but Ray is still unavailable.");
        }
      } catch (e) {
        setBanner("bad", "Reconnect failed: " + e.message);
      }
    }

    function buildPayloadFromForm() {
      return {
        dataset_root: document.getElementById("dataset_root").value,
        num_workers: parseInt(document.getElementById("num_workers").value || "2"),
        epochs: parseInt(document.getElementById("epochs").value || "1"),
        batch_size: parseInt(document.getElementById("batch_size").value || "8"),
        sync_every_examples: parseInt(document.getElementById("sync_every_examples").value || "8"),
        image_height: parseInt(document.getElementById("image_height").value || "224"),
        image_width: parseInt(document.getElementById("image_width").value || "224"),
        learning_rate: parseFloat(document.getElementById("learning_rate").value || "0.001"),
        num_gpus_per_worker: parseFloat(document.getElementById("num_gpus_per_worker").value || "1"),
        ray_address: document.getElementById("ray_address").value || "auto",
        data_mode: document.getElementById("data_mode").value || "shared_path",
        allow_gpu_fallback: (document.getElementById("allow_gpu_fallback").value || "true").toLowerCase() !== "false",
        ssp_staleness: parseInt(document.getElementById("ssp_staleness").value || "2"),
        quorum_fraction: parseFloat(document.getElementById("quorum_fraction").value || "0.6"),
        round_timeout_ms: parseInt(document.getElementById("round_timeout_ms").value || "100"),
        quantization: document.getElementById("quantization").value || "fp32",
        topk_pct: parseFloat(document.getElementById("topk_pct").value || "1.0"),
        residual_feedback: (document.getElementById("residual_feedback").value || "true").toLowerCase() !== "false",
        ps_optimizer: document.getElementById("ps_optimizer").value || "adam",
        ps_momentum: parseFloat(document.getElementById("ps_momentum").value || "0.9"),
        adaptive_sync: (document.getElementById("adaptive_sync").value || "true").toLowerCase() !== "false",
        random_seed: parseInt(document.getElementById("random_seed").value || "42"),
        benchmark_repeats: parseInt(document.getElementById("benchmark_repeats").value || "3"),
        dataset_roots_csv: document.getElementById("dataset_roots_csv").value || "",
        train_split: parseFloat(document.getElementById("train_split").value || "0.8"),
        val_split: parseFloat(document.getElementById("val_split").value || "0.1"),
        model_name: document.getElementById("model_name").value || "tiny"
      };
    }

    function validateBasePayload(payload, msgEl) {
      if (!payload.dataset_root) {
        msgEl.textContent = "dataset_root is required";
        return false;
      }
      if (payload.train_split + payload.val_split >= 1.0) {
        msgEl.textContent = "Invalid split: train_split + val_split must be < 1.0";
        return false;
      }
      if (payload.quorum_fraction <= 0 || payload.quorum_fraction > 1) {
        msgEl.textContent = "Invalid quorum_fraction: must be in (0, 1].";
        return false;
      }
      if (payload.topk_pct <= 0 || payload.topk_pct > 100) {
        msgEl.textContent = "Invalid topk_pct: must be in (0, 100].";
        return false;
      }
      return true;
    }

    async function startPlainTraining() {
      const payload = buildPayloadFromForm();
      const msg = document.getElementById("start_msg");
      if (!validateBasePayload(payload, msg)) return;
      try {
        const j = await safeFetch("/experiments/train/plain", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload)
        });
        msg.textContent = "Plain PS job queued: " + j.job_id;
        await refreshJobs();
      } catch (e) {
        msg.textContent = "Failed to start Plain PS job: " + e.message;
      }
    }

    async function startCustomTraining() {
      const payload = buildPayloadFromForm();
      const msg = document.getElementById("start_msg");
      if (!validateBasePayload(payload, msg)) return;
      try {
        const j = await safeFetch("/experiments/train/custom", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload)
        });
        msg.textContent = "Custom strategy job queued: " + j.job_id;
        await refreshJobs();
      } catch (e) {
        msg.textContent = "Failed to start Custom strategy job: " + e.message;
      }
    }

    async function startAblation() {
      const msg = document.getElementById("start_msg");
      const payload = buildPayloadFromForm();
      if (!validateBasePayload(payload, msg)) return;
      try {
        const j = await safeFetch("/experiments/ablation", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload)
        });
        msg.textContent = "Ablation queued: " + j.job_id;
        await refreshJobs();
      } catch (e) {
        msg.textContent = "Failed to start ablation: " + e.message;
      }
    }

    async function startResearchPack() {
      const msg = document.getElementById("start_msg");
      const payload = buildPayloadFromForm();
      if (!validateBasePayload(payload, msg)) return;
      try {
        const j = await safeFetch("/experiments/research-pack", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload)
        });
        msg.textContent = "Research pack queued: " + j.job_id;
        await refreshJobs();
      } catch (e) {
        msg.textContent = "Failed to start research pack: " + e.message;
      }
    }

    let metricsChart = null;
    let commChart = null;
    let ablationChart = null;
    let pairMetricsChart = null;
    let pairCommChart = null;
    let selectedPlainJobId = null;
    let selectedCustomJobId = null;
    let liveLogsJobId = null;
    let liveLogsPoller = null;

    function setCardVisible(id, visible) {
      const el = document.getElementById(id);
      if (!el) return;
      el.style.display = visible ? "block" : "none";
    }

    function initAccordionCards() {
      const cards = document.querySelectorAll(".card");
      cards.forEach((card) => {
        const h3 = card.querySelector(":scope > h3");
        if (!h3 || card.dataset.accordionReady === "1") return;

        const headerBtn = document.createElement("button");
        headerBtn.type = "button";
        headerBtn.className = "accordion-header-btn";
        const title = document.createElement("span");
        title.textContent = h3.textContent || "Section";
        const icon = document.createElement("span");
        icon.className = "accordion-icon";
        icon.textContent = "−";
        headerBtn.appendChild(title);
        headerBtn.appendChild(icon);

        const content = document.createElement("div");
        content.className = "accordion-content";

        let node = h3.nextSibling;
        while (node) {
          const next = node.nextSibling;
          content.appendChild(node);
          node = next;
        }

        h3.replaceWith(headerBtn);
        card.appendChild(content);
        card.dataset.accordionReady = "1";

        headerBtn.addEventListener("click", () => {
          const collapsed = card.classList.toggle("collapsed");
          icon.textContent = collapsed ? "+" : "−";
          headerBtn.setAttribute("aria-expanded", collapsed ? "false" : "true");
        });
      });
    }

    function fmtDelta(v) {
      const n = Number(v || 0);
      if (n > 0) return `+${n.toFixed(4)} (custom better/higher)`;
      if (n < 0) return `${n.toFixed(4)} (plain better/lower)`;
      return "0.0000 (equal)";
    }

    function badge(status) {
      return `<span class="pill pill-${status}">${status}</span>`;
    }

    function updateKpis(jobs) {
      const counts = {queued: 0, running: 0, completed: 0, failed: 0};
      Object.values(jobs).forEach((j) => {
        const st = j.status || "queued";
        if (counts[st] !== undefined) counts[st] += 1;
      });
      document.getElementById("kpi_queued").textContent = counts.queued;
      document.getElementById("kpi_running").textContent = counts.running;
      document.getElementById("kpi_completed").textContent = counts.completed;
      document.getElementById("kpi_failed").textContent = counts.failed;
    }

    async function refreshJobs() {
      let jobs = {};
      try {
        jobs = await safeFetch("/experiments");
      } catch (_) {
        return;
      }
      updateKpis(jobs);
      const body = document.getElementById("jobs_body");
      const plainBody = document.getElementById("plain_jobs_body");
      const customBody = document.getElementById("custom_jobs_body");
      body.innerHTML = "";
      plainBody.innerHTML = "";
      customBody.innerHTML = "";
      Object.entries(jobs).forEach(([jobId, info]) => {
        const tr = document.createElement("tr");
        const status = info.status || "unknown";
        const jobType = info.job_type || "training";
        const strategy = info.strategy_type || "unknown";
        let actionBtn = `<span class="small">-</span>`;
        if (status === "completed") {
          actionBtn = `<span class="action-icons"><button class="icon-btn" title="View job" aria-label="View job" onclick="viewJob('${jobId}')">👁</button><button class="icon-btn delete" title="Delete job" aria-label="Delete job" onclick="deleteJob('${jobId}')">🗑</button></span>`;
        } else if (status === "running") {
          actionBtn = `<span class="action-icons"><button class="icon-btn" title="Live worker logs" aria-label="Live worker logs" onclick="viewLiveLogs('${jobId}')">📡</button></span>`;
        } else if (status === "failed") {
          actionBtn = `<span class="action-icons"><button class="icon-btn delete" title="Delete job" aria-label="Delete job" onclick="deleteJob('${jobId}')">🗑</button></span>`;
        }
        tr.innerHTML = `
          <td>${jobId}</td>
          <td>${badge(status)}</td>
          <td>${jobType}</td>
          <td>${strategy}</td>
          <td>${info.created_at || "-"}</td>
          <td>${info.finished_at || "-"}</td>
          <td>${actionBtn}</td>
        `;
        body.appendChild(tr);

        if (status === "completed" && jobType === "training" && strategy === "plain_ps") {
          const s = document.createElement("tr");
          s.innerHTML = `
            <td><input type="radio" name="plain_select" value="${jobId}" ${selectedPlainJobId === jobId ? "checked" : ""} onchange="selectedPlainJobId='${jobId}'" /></td>
            <td>${jobId}</td>
            <td>${info.finished_at || "-"}</td>
          `;
          plainBody.appendChild(s);
        }
        if (status === "completed" && jobType === "training" && strategy === "custom_strategy") {
          const s = document.createElement("tr");
          s.innerHTML = `
            <td><input type="radio" name="custom_select" value="${jobId}" ${selectedCustomJobId === jobId ? "checked" : ""} onchange="selectedCustomJobId='${jobId}'" /></td>
            <td>${jobId}</td>
            <td>${info.finished_at || "-"}</td>
          `;
          customBody.appendChild(s);
        }
      });
      if (!plainBody.children.length) {
        const tr = document.createElement("tr");
        tr.innerHTML = "<td colspan='3'>No completed Plain PS jobs</td>";
        plainBody.appendChild(tr);
      }
      if (!customBody.children.length) {
        const tr = document.createElement("tr");
        tr.innerHTML = "<td colspan='3'>No completed Custom strategy jobs</td>";
        customBody.appendChild(tr);
      }
    }

    async function deleteJob(jobId) {
      const ok = window.confirm(`Delete job ${jobId}?`);
      if (!ok) return;
      try {
        const res = await safeFetch(`/experiments/${jobId}`, { method: "DELETE" });
        if (res.status !== "ok") {
          alert(res.error || "Delete failed.");
          return;
        }
        if (selectedPlainJobId === jobId) selectedPlainJobId = null;
        if (selectedCustomJobId === jobId) selectedCustomJobId = null;
        await refreshJobs();
      } catch (e) {
        alert("Delete failed: " + e.message);
      }
    }

    async function deleteFinishedJobs() {
      const ok = window.confirm("Delete all completed and failed jobs?");
      if (!ok) return;
      try {
        const res = await safeFetch("/experiments", { method: "DELETE" });
        if (res.status !== "ok") {
          alert(res.error || "Delete finished failed.");
          return;
        }
        selectedPlainJobId = null;
        selectedCustomJobId = null;
        await refreshJobs();
      } catch (e) {
        alert("Delete finished failed: " + e.message);
      }
    }

    async function compareSelectedRuns() {
      const msg = document.getElementById("compare_msg");
      const out = document.getElementById("pair_compare_details");
      if (!selectedPlainJobId || !selectedCustomJobId) {
        msg.textContent = "Select one completed Plain job and one completed Custom job first.";
        return;
      }
      try {
        const data = await safeFetch("/experiments/compare-selected", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({plain_job_id: selectedPlainJobId, custom_job_id: selectedCustomJobId}),
        });
        if (data.status !== "ok") {
          msg.textContent = data.error || "Comparison failed.";
          return;
        }
        msg.textContent = "Comparison ready.";
        out.textContent = JSON.stringify(data, null, 2);
        setCardVisible("pair_card", true);
        renderPairComparison(data);
      } catch (e) {
        msg.textContent = "Comparison failed: " + e.message;
      }
    }

    function renderPairComparison(payload) {
      const summaryEl = document.getElementById("pair_compare_summary");
      const metricsBody = document.getElementById("pair_metrics_body");
      const commBody = document.getElementById("pair_comm_body");

      const cmp = payload.comparison || {};
      const metrics = cmp.metrics || {};
      const comm = cmp.communication || {};
      const compat = cmp.compatibility || {};
      const comparable = !!compat.strictly_comparable;

      summaryEl.innerHTML = comparable
        ? "<b>Comparison quality:</b> strict apples-to-apples (same dataset/model/size/splits)."
        : "<b>Comparison quality:</b> partial (settings differ). Interpret deltas carefully.";

      metricsBody.innerHTML = "";
      Object.entries(metrics).forEach(([name, v]) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${name}</td>
          <td>${Number(v.plain_ps || 0).toFixed(4)}</td>
          <td>${Number(v.custom_strategy || 0).toFixed(4)}</td>
          <td>${fmtDelta(v.delta_custom_minus_plain)}</td>
        `;
        metricsBody.appendChild(tr);
      });
      if (!metricsBody.children.length) {
        metricsBody.innerHTML = "<tr><td colspan='4'>No metric data available.</td></tr>";
      }

      commBody.innerHTML = "";
      Object.entries(comm).forEach(([name, v]) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${name}</td>
          <td>${Number(v.plain_ps || 0).toFixed(4)}</td>
          <td>${Number(v.custom_strategy || 0).toFixed(4)}</td>
          <td>${fmtDelta(v.delta_custom_minus_plain)}</td>
        `;
        commBody.appendChild(tr);
      });
      if (!commBody.children.length) {
        commBody.innerHTML = "<tr><td colspan='4'>No communication data available.</td></tr>";
      }

      const metricKeys = Object.keys(metrics);
      const metricPlain = metricKeys.map((k) => Number((metrics[k] || {}).plain_ps || 0));
      const metricCustom = metricKeys.map((k) => Number((metrics[k] || {}).custom_strategy || 0));

      const commKeys = Object.keys(comm);
      const commPlain = commKeys.map((k) => Number((comm[k] || {}).plain_ps || 0));
      const commCustom = commKeys.map((k) => Number((comm[k] || {}).custom_strategy || 0));

      if (pairMetricsChart) pairMetricsChart.destroy();
      if (pairCommChart) pairCommChart.destroy();

      pairMetricsChart = new Chart(document.getElementById("pair_metrics_chart"), {
        type: "bar",
        data: {
          labels: metricKeys,
          datasets: [
            { label: "Plain PS", data: metricPlain, backgroundColor: "#f59e0b" },
            { label: "Custom Strategy", data: metricCustom, backgroundColor: "#2563eb" }
          ]
        },
        options: {scales: {x: {grid: {display: false}}, y: {beginAtZero: true, max: 1, grid: {display: false}}}}
      });

      pairCommChart = new Chart(document.getElementById("pair_comm_chart"), {
        type: "bar",
        data: {
          labels: commKeys,
          datasets: [
            { label: "Plain PS", data: commPlain, backgroundColor: "#f43f5e" },
            { label: "Custom Strategy", data: commCustom, backgroundColor: "#10b981" }
          ]
        },
        options: {scales: {x: {grid: {display: false}}, y: {beginAtZero: true, grid: {display: false}}}}
      });
    }

    async function refreshResources() {
      let data = {};
      try {
        data = await safeFetch("/cluster/resources");
      } catch (_) {
        document.getElementById("cluster_resources").textContent = "Unable to reach backend.";
        return;
      }
      if (data.status !== "ok") {
        setBanner("warn", "Backend up, but Ray cluster is not connected.");
      }
      document.getElementById("cluster_resources").textContent = JSON.stringify(data, null, 2);
    }

    async function refreshNodes() {
      let data = {};
      try {
        data = await safeFetch("/cluster/nodes");
      } catch (_) {
        return;
      }
      const body = document.getElementById("nodes_body");
      body.innerHTML = "";
      const nodes = data.nodes || [];
      if (!nodes.length) {
        const tr = document.createElement("tr");
        tr.innerHTML = "<td colspan='5'>No connected nodes</td>";
        body.appendChild(tr);
        return;
      }
      nodes.forEach((n) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${n.node_manager_address || "-"}</td>
          <td>${n.alive ? "yes" : "no"}</td>
          <td>${(n.cpus !== undefined && n.cpus !== null) ? n.cpus : 0}</td>
          <td>${(n.gpus !== undefined && n.gpus !== null) ? n.gpus : 0}</td>
          <td>${(n.node_id || "").slice(0, 12)}</td>
        `;
        body.appendChild(tr);
      });
    }

    async function viewJob(jobId) {
      let data = {};
      try {
        data = await safeFetch("/experiments/" + jobId);
      } catch (_) {
        return;
      }
      document.getElementById("job_details").textContent = JSON.stringify({job_id: jobId, ...data}, null, 2);
      setCardVisible("insights_card", true);
      setCardVisible("job_details_card", true);
      renderSummary(data);
      renderCharts(data);
      renderAblation(data);
    }

    async function refreshLiveLogsNow() {
      if (!liveLogsJobId) return;
      const msg = document.getElementById("worker_logs_msg");
      const out = document.getElementById("worker_logs_view");
      try {
        const data = await safeFetch(`/experiments/${liveLogsJobId}/live-logs`);
        if (data.status !== "ok") {
          msg.textContent = data.message || "No live logs available.";
          if (data.status === "not_running" && liveLogsPoller) {
            clearInterval(liveLogsPoller);
            liveLogsPoller = null;
          }
          return;
        }
        msg.textContent = `Live stream for ${liveLogsJobId} (${(data.workers || []).length} workers)`;
        const lines = [];
        (data.workers || []).forEach((w) => {
          const hdr = `${w.worker_id || "unknown"} | ${w.node_ip || "-"} | ${w.hostname || "-"} | pid=${w.pid || "-"}`;
          lines.push(hdr);
          if (w.error) {
            lines.push(`ERROR: ${w.error}`);
          } else {
            (w.logs || []).forEach((ln) => lines.push("  " + ln));
          }
          lines.push("");
        });
        out.textContent = lines.join("\\n") || "No worker logs yet.";
      } catch (e) {
        msg.textContent = "Failed to load logs: " + e.message;
      }
    }

    async function viewLiveLogs(jobId) {
      liveLogsJobId = jobId;
      setCardVisible("worker_logs_card", true);
      document.getElementById("worker_logs_view").textContent = `Loading live logs for ${jobId}...`;
      await refreshLiveLogsNow();
      if (liveLogsPoller) clearInterval(liveLogsPoller);
      liveLogsPoller = setInterval(refreshLiveLogsNow, 3000);
    }

    function clearAblationView(message = "No ablation result selected.") {
      const body = document.getElementById("ablation_body");
      body.innerHTML = `<tr><td colspan="6">${message}</td></tr>`;
      setCardVisible("ablation_card", false);
      if (ablationChart) {
        ablationChart.destroy();
        ablationChart = null;
      }
    }

    function renderSummary(data) {
      const target = document.getElementById("result_summary");
      if (!data || data.status !== "completed" || !data.result) {
        target.textContent = data && data.error ? `Error: ${data.error}` : "No completed result selected.";
        clearAblationView();
        return;
      }
      if (Array.isArray(data.result.rows)) {
        target.innerHTML = `<b>Ablation completed:</b> ${data.result.rows.length} preset runs compared.`;
        return;
      }
      const tm = data.result.test_metrics || {};
      const comm = data.result.communication_cost || {};
      target.innerHTML = `
        <b>Accuracy:</b> ${Number(tm.accuracy || 0).toFixed(4)} |
        <b>Precision:</b> ${Number(tm.precision || 0).toFixed(4)} |
        <b>Recall:</b> ${Number(tm.recall || 0).toFixed(4)} |
        <b>F1:</b> ${Number(tm.f1_score || 0).toFixed(4)} |
        <b>AUC:</b> ${Number(tm.auc || 0).toFixed(4)} <br/>
        <b>Total Communication:</b> ${Number(comm.total_mb || 0).toFixed(3)} MB |
        <b>Messages:</b> ${comm.messages || 0}
      `;
    }

    function renderCharts(data) {
      if (!data || data.status !== "completed" || !data.result) return;
      if (Array.isArray(data.result.rows)) {
        if (metricsChart) metricsChart.destroy();
        if (commChart) commChart.destroy();
        return;
      }
      const tm = data.result.test_metrics || {};
      const comm = data.result.communication_cost || {};

      const metricsCtx = document.getElementById("metrics_chart");
      const commCtx = document.getElementById("comm_chart");

      if (metricsChart) metricsChart.destroy();
      if (commChart) commChart.destroy();

      metricsChart = new Chart(metricsCtx, {
        type: "bar",
        data: {
          labels: ["accuracy", "precision", "recall", "f1_score", "auc"],
          datasets: [{
            label: "Metric value",
            data: [
              tm.accuracy || 0,
              tm.precision || 0,
              tm.recall || 0,
              tm.f1_score || 0,
              tm.auc || 0
            ],
            backgroundColor: "#3b82f6"
          }]
        },
        options: {scales: {x: {grid: {display: false}}, y: {beginAtZero: true, max: 1, grid: {display: false}}}}
      });

      commChart = new Chart(commCtx, {
        type: "doughnut",
        data: {
          labels: ["worker->ps", "ps->worker"],
          datasets: [{
            data: [
              (comm.bytes_worker_to_ps || 0) / (1024 * 1024),
              (comm.bytes_ps_to_worker || 0) / (1024 * 1024)
            ],
            backgroundColor: ["#ef4444", "#10b981"]
          }]
        }
      });
    }

    function renderAblation(data) {
      if (!data || data.status !== "completed" || !data.result || !Array.isArray(data.result.rows)) {
        clearAblationView();
        return;
      }
      setCardVisible("ablation_card", true);
      const rows = data.result.rows;
      const body = document.getElementById("ablation_body");
      body.innerHTML = "";
      rows.forEach((r) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${r.preset}</td>
          <td>${Number(r.accuracy || 0).toFixed(4)}</td>
          <td>${Number(r.f1_score || 0).toFixed(4)}</td>
          <td>${Number(r.auc || 0).toFixed(4)}</td>
          <td>${Number(r.total_mb || 0).toFixed(3)}</td>
          <td>${r.messages || 0}</td>
        `;
        body.appendChild(tr);
      });
      const ablationCtx = document.getElementById("ablation_chart");
      if (ablationChart) ablationChart.destroy();
      ablationChart = new Chart(ablationCtx, {
        type: "bar",
        data: {
          labels: rows.map((r) => r.preset),
          datasets: [
            {
              label: "Accuracy",
              data: rows.map((r) => r.accuracy || 0),
              backgroundColor: "#2563eb",
            },
            {
              label: "F1",
              data: rows.map((r) => r.f1_score || 0),
              backgroundColor: "#16a34a",
            },
          ],
        },
        options: {scales: {x: {grid: {display: false}}, y: {beginAtZero: true, max: 1, grid: {display: false}}}},
      });
    }

    async function downloadAblationArtifact(fmt) {
      const msg = document.getElementById("ablation_download_msg");
      try {
        const meta = await safeFetch("/artifacts/ablation/latest");
        if (meta.status !== "ok" || !meta[fmt]) {
          msg.textContent = "No ablation artifacts available yet.";
          return;
        }
        window.open(`/artifacts/ablation/download/${fmt}`, "_blank");
        msg.textContent = `Downloading latest ${fmt.toUpperCase()} (${meta[fmt].name})`;
      } catch (e) {
        msg.textContent = "Download failed: " + e.message;
      }
    }

    async function downloadResearchArtifact(kind) {
      const msg = document.getElementById("research_download_msg");
      try {
        const meta = await safeFetch("/artifacts/research-pack/latest");
        if (meta.status !== "ok" || !meta[kind]) {
          msg.textContent = "No research-pack artifacts available yet.";
          return;
        }
        window.open(`/artifacts/research-pack/download/${kind}`, "_blank");
        msg.textContent = `Downloading ${kind} (${meta[kind].name})`;
      } catch (e) {
        msg.textContent = "Download failed: " + e.message;
      }
    }

    initAccordionCards();
    probeConnection();
    refreshJobs();
    refreshResources();
    refreshNodes();
    setInterval(probeConnection, 5000);
    setInterval(refreshJobs, 5000);
    setInterval(refreshResources, 5000);
    setInterval(refreshNodes, 5000);
  </script>
</body>
</html>"""


@app.post("/experiments")
def start_experiment(request: ExperimentRequest, background_tasks: BackgroundTasks) -> Dict[str, str]:
    job_id = str(uuid.uuid4())
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "queued",
            "created_at": _now_iso(),
            "started_at": None,
            "finished_at": None,
            "request": request.model_dump(),
            "result": None,
            "error": None,
            "job_type": "training",
            "strategy_type": "custom_strategy",
        }
        _save_jobs_to_disk_locked()
    background_tasks.add_task(_run_job, job_id, request, "custom_strategy")
    return {"job_id": job_id, "status": "queued"}


@app.post("/experiments/train/plain")
def start_plain_training(request: ExperimentRequest, background_tasks: BackgroundTasks) -> Dict[str, str]:
    job_id = str(uuid.uuid4())
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "queued",
            "created_at": _now_iso(),
            "started_at": None,
            "finished_at": None,
            "request": request.model_dump(),
            "result": None,
            "error": None,
            "job_type": "training",
            "strategy_type": "plain_ps",
        }
        _save_jobs_to_disk_locked()
    background_tasks.add_task(_run_job, job_id, request, "plain_ps")
    return {"job_id": job_id, "status": "queued"}


@app.post("/experiments/train/custom")
def start_custom_training(request: ExperimentRequest, background_tasks: BackgroundTasks) -> Dict[str, str]:
    job_id = str(uuid.uuid4())
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "queued",
            "created_at": _now_iso(),
            "started_at": None,
            "finished_at": None,
            "request": request.model_dump(),
            "result": None,
            "error": None,
            "job_type": "training",
            "strategy_type": "custom_strategy",
        }
        _save_jobs_to_disk_locked()
    background_tasks.add_task(_run_job, job_id, request, "custom_strategy")
    return {"job_id": job_id, "status": "queued"}


@app.post("/experiments/ablation")
def start_ablation(request: ExperimentRequest, background_tasks: BackgroundTasks) -> Dict[str, str]:
    job_id = str(uuid.uuid4())
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "queued",
            "created_at": _now_iso(),
            "started_at": None,
            "finished_at": None,
            "request": request.model_dump(),
            "result": None,
            "error": None,
            "job_type": "ablation",
            "strategy_type": "n/a",
        }
        _save_jobs_to_disk_locked()
    background_tasks.add_task(_run_ablation, job_id, request)
    return {"job_id": job_id, "status": "queued"}


@app.post("/experiments/research-pack")
def start_research_pack(request: ExperimentRequest, background_tasks: BackgroundTasks) -> Dict[str, str]:
    job_id = str(uuid.uuid4())
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "queued",
            "created_at": _now_iso(),
            "started_at": None,
            "finished_at": None,
            "request": request.model_dump(),
            "result": None,
            "error": None,
            "job_type": "research_pack",
            "strategy_type": "n/a",
        }
        _save_jobs_to_disk_locked()
    background_tasks.add_task(_run_research_pack, job_id, request)
    return {"job_id": job_id, "status": "queued"}


@app.get("/experiments/{job_id}")
def get_experiment(job_id: str) -> Dict:
    with JOBS_LOCK:
        return JOBS.get(job_id, {"status": "not_found"})


@app.get("/experiments/{job_id}/live-logs")
def get_experiment_live_logs(job_id: str) -> Dict:
    with JOBS_LOCK:
        payload = JOBS.get(job_id)
        if not payload:
            return {"status": "not_found"}
        status = str(payload.get("status", ""))
    if status != "running":
        return {"status": "not_running", "workers": [], "message": "Live logs are only available for running jobs."}
    from ray_ps_async.runner import get_live_worker_logs

    return get_live_worker_logs(job_id, limit=120)


@app.get("/experiments")
def list_experiments() -> Dict[str, Dict]:
    with JOBS_LOCK:
        return {
            job_id: {
                "status": payload.get("status"),
                "created_at": payload.get("created_at"),
                "started_at": payload.get("started_at"),
                "finished_at": payload.get("finished_at"),
                "error": payload.get("error"),
                "job_type": payload.get("job_type", "training"),
                "strategy_type": payload.get("strategy_type", "unknown"),
            }
            for job_id, payload in JOBS.items()
        }


@app.delete("/experiments/{job_id}")
def delete_experiment(job_id: str) -> Dict[str, str]:
    with JOBS_LOCK:
        payload = JOBS.get(job_id)
        if not payload:
            return {"status": "not_found"}
        status = str(payload.get("status", ""))
        if status not in {"completed", "failed"}:
            return {"status": "error", "error": "Only completed/failed jobs can be deleted."}
        del JOBS[job_id]
        _save_jobs_to_disk_locked()
    return {"status": "ok", "job_id": job_id}


@app.delete("/experiments")
def delete_finished_experiments() -> Dict[str, int | str]:
    with JOBS_LOCK:
        removable = [jid for jid, payload in JOBS.items() if str(payload.get("status", "")) in {"completed", "failed"}]
        for jid in removable:
            del JOBS[jid]
        _save_jobs_to_disk_locked()
    return {"status": "ok", "deleted": len(removable)}


@app.post("/experiments/compare-selected")
def compare_selected_runs(request: CompareRequest) -> Dict:
    with JOBS_LOCK:
        plain_job = JOBS.get(request.plain_job_id)
        custom_job = JOBS.get(request.custom_job_id)
    if not plain_job or not custom_job:
        return {"status": "error", "error": "One or both selected job IDs do not exist."}
    if plain_job.get("status") != "completed" or custom_job.get("status") != "completed":
        return {"status": "error", "error": "Both jobs must be completed before comparison."}
    if plain_job.get("strategy_type") != "plain_ps":
        return {"status": "error", "error": "plain_job_id must point to a Plain PS training job."}
    if custom_job.get("strategy_type") != "custom_strategy":
        return {"status": "error", "error": "custom_job_id must point to a Custom Strategy training job."}
    return {
        "status": "ok",
        "plain_job_id": request.plain_job_id,
        "custom_job_id": request.custom_job_id,
        "comparison": _compare_pair(plain_job, custom_job),
    }

