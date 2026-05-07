from datetime import datetime, timezone
from threading import Lock
from typing import Callable, Dict, Literal
import uuid

from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import HTMLResponse
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
    train_split: float = Field(default=0.8, gt=0.0, lt=1.0)
    val_split: float = Field(default=0.1, ge=0.0, lt=1.0)
    model_name: str = "tiny"


app = FastAPI(title="Ray Async Parameter Server Trainer")
JOBS: Dict[str, Dict] = {}
JOBS_LOCK = Lock()


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
        train_split=request.train_split,
        val_split=request.val_split,
        model_name=request.model_name,
    )


def _run_job(job_id: str, request: ExperimentRequest) -> None:
    with JOBS_LOCK:
        if job_id not in JOBS:
            return
        JOBS[job_id]["status"] = "running"
        JOBS[job_id]["started_at"] = _now_iso()
    try:
        runner: Callable = _experiment_runner()
        cfg = _build_config(request)
        result = runner(cfg)
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "completed"
            JOBS[job_id]["result"] = result
            JOBS[job_id]["finished_at"] = _now_iso()
    except Exception as exc:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = str(exc)
            JOBS[job_id]["finished_at"] = _now_iso()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/cluster/resources")
def cluster_resources() -> Dict:
    try:
        import ray

        if not ray.is_initialized():
            try:
                ray.init(address="auto", ignore_reinit_error=True)
            except Exception:
                return {"status": "not_connected", "resources": {}}
        return {"status": "ok", "resources": ray.available_resources()}
    except Exception as exc:
        return {"status": "error", "error": str(exc), "resources": {}}


@app.get("/cluster/nodes")
def cluster_nodes() -> Dict:
    try:
        import ray

        if not ray.is_initialized():
            try:
                ray.init(address="auto", ignore_reinit_error=True)
            except Exception:
                return {"status": "not_connected", "nodes": []}

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
    body { font-family: Arial, sans-serif; margin: 24px; background: #f4f6fb; color: #111; }
    h1 { margin-bottom: 8px; }
    .muted { color: #555; margin-bottom: 16px; }
    .card { background: #fff; border: 1px solid #d9e1f2; border-radius: 10px; padding: 16px; margin-bottom: 16px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }
    .row { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 10px; }
    .field { display: flex; flex-direction: column; min-width: 190px; }
    input { padding: 8px; border: 1px solid #bbb; border-radius: 6px; }
    select { padding: 8px; border: 1px solid #bbb; border-radius: 6px; background: #fff; }
    button { padding: 10px 14px; border: none; border-radius: 6px; background: #2563eb; color: #fff; cursor: pointer; }
    button:hover { background: #1d4ed8; }
    .kpi { font-size: 24px; font-weight: 700; }
    .grid3 { display: grid; grid-template-columns: repeat(auto-fit,minmax(220px,1fr)); gap: 10px; }
    .pill { display: inline-block; padding: 3px 8px; border-radius: 999px; font-size: 12px; font-weight: 700; }
    .pill-queued { background: #eef2ff; color: #3730a3; }
    .pill-running { background: #fff7ed; color: #b45309; }
    .pill-completed { background: #ecfdf5; color: #166534; }
    .pill-failed { background: #fef2f2; color: #b91c1c; }
    .charts { display: grid; grid-template-columns: repeat(auto-fit,minmax(320px,1fr)); gap: 16px; }
    canvas { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 8px; }
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
    pre { background: #0b1020; color: #d8e1ff; padding: 12px; border-radius: 8px; overflow: auto; max-height: 340px; }
    table { width: 100%; border-collapse: collapse; }
    th, td { border-bottom: 1px solid #eee; text-align: left; padding: 8px; font-size: 14px; }
    .conn-banner { border-radius: 8px; padding: 10px 12px; margin-bottom: 12px; font-size: 13px; }
    .conn-ok { background: #ecfdf5; color: #166534; border: 1px solid #86efac; }
    .conn-warn { background: #fff7ed; color: #9a3412; border: 1px solid #fdba74; }
    .conn-bad { background: #fef2f2; color: #b91c1c; border: 1px solid #fca5a5; }
  </style>
</head>
<body>
  <h1>Ray Async Parameter-Server Dashboard</h1>
  <div class="muted">Production-focused dashboard for launching, monitoring, and analyzing experiments.</div>
  <div id="conn_banner" class="conn-banner conn-ok">Backend status: checking...</div>
  <div style="margin-bottom:12px;">
    <button onclick="reconnectRay()">Reconnect Ray</button>
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
    <button onclick="refreshResources()">Refresh Resources</button>
    <pre id="cluster_resources">Not loaded yet...</pre>
  </div>

  <div class="card">
    <h3>Connected Nodes</h3>
    <button onclick="refreshNodes()">Refresh Nodes</button>
    <table>
      <thead>
        <tr><th>Address</th><th>Alive</th><th>CPU</th><th>GPU</th><th>Node ID</th></tr>
      </thead>
      <tbody id="nodes_body"></tbody>
    </table>
  </div>

  <div class="card">
    <h3>Start Experiment</h3>
    <div class="row">
      <div class="field"><label class="label-wrap">Dataset Root <span class="hint" data-tip="Root path of class-wise dataset. Used by head to prepare splits and partitions.">i</span></label><input id="dataset_root" placeholder="/path/to/dataset_root" /></div>
      <div class="field"><label class="label-wrap">Workers <span class="hint" data-tip="Number of distributed worker actors to start for this job.">i</span></label><input id="num_workers" type="number" value="2" /></div>
      <div class="field"><label class="label-wrap">Epochs <span class="hint" data-tip="How many passes each worker performs over its assigned training partition.">i</span></label><input id="epochs" type="number" value="1" /></div>
      <div class="field"><label class="label-wrap">Batch Size <span class="hint" data-tip="Mini-batch size used inside each local training step.">i</span></label><input id="batch_size" type="number" value="8" /></div>
      <div class="field"><label class="label-wrap">Sync Every Examples <span class="hint" data-tip="After this many examples, worker sends update to PS and pulls latest global weights.">i</span></label><input id="sync_every_examples" type="number" value="8" /></div>
      <div class="field"><label class="label-wrap">Ray Address <span class="hint" data-tip="local: run on same machine. auto: connect to existing Ray cluster (LAN).">i</span></label><select id="ray_address"><option value="auto" selected>auto</option><option value="local">local</option></select></div>
      <div class="field"><label class="label-wrap">Data Mode <span class="hint" data-tip="shared_path: workers read dataset path directly. stream_from_head: head sends each worker subset at runtime.">i</span></label><select id="data_mode"><option value="shared_path" selected>shared_path</option><option value="stream_from_head">stream_from_head</option></select></div>
    </div>
    <div class="row">
      <div class="field"><label class="label-wrap">Image Height <span class="hint" data-tip="Input image resize height used before feeding the model.">i</span></label><input id="image_height" type="number" value="224" /></div>
      <div class="field"><label class="label-wrap">Image Width <span class="hint" data-tip="Input image resize width used before feeding the model.">i</span></label><input id="image_width" type="number" value="224" /></div>
      <div class="field"><label class="label-wrap">Learning Rate <span class="hint" data-tip="Optimizer step size used by workers in local updates.">i</span></label><input id="learning_rate" type="number" step="0.0001" value="0.001" /></div>
      <div class="field"><label class="label-wrap">GPUs per Worker <span class="hint" data-tip="GPU resource requested per worker actor. Use 1 for one-GPU-per-worker.">i</span></label><input id="num_gpus_per_worker" type="number" step="0.5" value="1" /></div>
      <div class="field"><label class="label-wrap">Allow GPU Fallback <span class="hint" data-tip="If true, job falls back to CPU when required GPUs are unavailable.">i</span></label><select id="allow_gpu_fallback"><option value="true" selected>true</option><option value="false">false</option></select></div>
      <div class="field"><label class="label-wrap">Model Name <span class="hint" data-tip="Model architecture key used to build network (e.g., tiny).">i</span></label><input id="model_name" value="tiny" /></div>
    </div>
    <div class="row">
      <div class="field"><label class="label-wrap">Train Split <span class="hint" data-tip="Fraction of each class used for worker training partitions.">i</span></label><input id="train_split" type="number" step="0.01" value="0.8" /></div>
      <div class="field"><label class="label-wrap">Validation Split <span class="hint" data-tip="Fraction used for validation after training. Test split is computed from remainder.">i</span></label><input id="val_split" type="number" step="0.01" value="0.1" /></div>
    </div>
    <button onclick="startExperiment()">Start</button>
    <p id="start_msg"></p>
  </div>

  <div class="card">
    <h3>Jobs</h3>
    <button onclick="refreshJobs()">Refresh</button>
    <table>
      <thead>
        <tr><th>Job ID</th><th>Status</th><th>Created</th><th>Finished</th><th>Action</th></tr>
      </thead>
      <tbody id="jobs_body"></tbody>
    </table>
  </div>

  <div class="card">
    <h3>Result Summary</h3>
    <div id="result_summary" class="small">Select a completed job to view summary.</div>
  </div>

  <div class="card charts">
    <div>
      <h3>Model Metrics</h3>
      <canvas id="metrics_chart" height="210"></canvas>
    </div>
    <div>
      <h3>Communication Cost (MB)</h3>
      <canvas id="comm_chart" height="210"></canvas>
    </div>
  </div>

  <div class="card">
    <h3>Job Details</h3>
    <pre id="job_details">Select a job to see details...</pre>
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

    async function startExperiment() {
      const payload = {
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
        train_split: parseFloat(document.getElementById("train_split").value || "0.8"),
        val_split: parseFloat(document.getElementById("val_split").value || "0.1"),
        model_name: document.getElementById("model_name").value || "tiny"
      };
      const msg = document.getElementById("start_msg");
      if (!payload.dataset_root) {
        msg.textContent = "dataset_root is required";
        return;
      }
      if (payload.train_split + payload.val_split >= 1.0) {
        msg.textContent = "Invalid split: train_split + val_split must be < 1.0";
        return;
      }
      try {
        const j = await safeFetch("/experiments", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload)
        });
        msg.textContent = "Job queued: " + j.job_id;
        await refreshJobs();
      } catch (e) {
        msg.textContent = "Failed to start job: " + e.message;
      }
    }

    let metricsChart = null;
    let commChart = null;

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
      body.innerHTML = "";
      Object.entries(jobs).forEach(([jobId, info]) => {
        const tr = document.createElement("tr");
        const status = info.status || "unknown";
        tr.innerHTML = `
          <td>${jobId}</td>
          <td>${badge(status)}</td>
          <td>${info.created_at || "-"}</td>
          <td>${info.finished_at || "-"}</td>
          <td><button onclick="viewJob('${jobId}')">View</button></td>
        `;
        body.appendChild(tr);
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
          <td>${n.cpus ?? 0}</td>
          <td>${n.gpus ?? 0}</td>
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
      renderSummary(data);
      renderCharts(data);
    }

    function renderSummary(data) {
      const target = document.getElementById("result_summary");
      if (!data || data.status !== "completed" || !data.result) {
        target.textContent = data && data.error ? `Error: ${data.error}` : "No completed result selected.";
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
        options: {scales: {y: {beginAtZero: true, max: 1}}}
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
        }
    background_tasks.add_task(_run_job, job_id, request)
    return {"job_id": job_id, "status": "queued"}


@app.get("/experiments/{job_id}")
def get_experiment(job_id: str) -> Dict:
    with JOBS_LOCK:
        return JOBS.get(job_id, {"status": "not_found"})


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
            }
            for job_id, payload in JOBS.items()
        }

