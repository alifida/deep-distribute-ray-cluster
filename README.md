# Ray Async Parameter-Server Training (Zero-Config Starter)

This project is a Ray implementation of your current idea in `latest_impl/latest_impl`:
- asynchronous parallel worker training,
- dataset partitioning across workers,
- central parameter server updates,
- first full-gradient submission, then delta-gradient submissions,
- worker pull of latest global state after every submission,
- configurable sync cadence (`sync_every_examples`),
- communication/network cost metrics in results.

## 1) Install

```bash
cd /home/ali/Documents/phd/latest_impl/ray_ps_async
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or use one command:

```bash
bash scripts/setup_node.sh
```

## 2) Dataset format

Use folder-per-class under one root:

```text
dataset_root/
  class0/
    img1.jpg
    img2.jpg
  class1/
    img3.jpg
    img4.jpg
```

Binary classification is expected by default (`0/1` labels from class index order).

## 3) Run from CLI

```bash
python cli.py \
  --dataset-root "/path/to/dataset_root" \
  --num-workers 2 \
  --epochs 2 \
  --sync-every-examples 8 \
  --num-gpus-per-worker 1
```

Run ablation suite (baseline -> full novelty):

```bash
python ablation_cli.py \
  --dataset-root "/path/to/dataset_root" \
  --num-workers 2 \
  --epochs 1 \
  --output-dir new_updates
```

Run repeated research evaluation pack (CI + multi-dataset):

```bash
python research_pack_cli.py \
  --dataset-root "/path/to/dataset_root" \
  --dataset-roots-csv "/path/to/ds1,/path/to/ds2" \
  --repeats 3 \
  --num-workers 2 \
  --epochs 1 \
  --output-dir new_updates
```

Artifacts generated in `new_updates/`:
- `research_pack_*.json` (full structured output)
- `research_pack_runs_*.csv` (all raw runs)
- `research_pack_summary_*.csv` (mean + CI95 summary)
- `research_pack_*.md` (paper-ready markdown tables)

Outputs JSON with:
- per-worker run stats,
- validation/test metrics,
- communication cost (`bytes_worker_to_ps`, `bytes_ps_to_worker`, `total_bytes`, message counts),
- dataset partition sizes.

## 4) Run as API interface

```bash
uvicorn api:app --host 0.0.0.0 --port 8080
```

Open UI dashboard:

```text
http://127.0.0.1:8080/
```

Start experiment:

```bash
curl -X POST "http://127.0.0.1:8080/experiments" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_root": "/path/to/dataset_root",
    "num_workers": 2,
    "epochs": 2,
    "batch_size": 8,
    "sync_every_examples": 8,
    "num_gpus_per_worker": 1
  }'
```

Check status/result:

```bash
curl "http://127.0.0.1:8080/experiments/<job_id>"
```

## Notes

- Local zero-config mode works with `ray.init()` automatically.
- For a cluster, start Ray head/worker nodes normally and keep this code unchanged.
- This implementation avoids external DB dependency for faster setup and lower runtime overhead.

## LAN Multi-Worker (Simple Scripts)

Shared config file:
- Edit `scripts/cluster_config.env` once (HEAD_IP, ports, GPU defaults).
- Linux and Windows scripts auto-read this file.
- For mixed Linux head + Windows worker clusters, keep `RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1`.

You need the project folder on every machine that will run Ray (head + workers), because workers must import the training code and Python dependencies.

### 1) On every machine (once)

```bash
cd /home/ali/Documents/phd/deep-distribute-ray-cluster
bash scripts/setup_node.sh
```

### 2) On head machine

```bash
cd /home/ali/Documents/phd/deep-distribute-ray-cluster
bash scripts/start_head.sh
bash scripts/start_ui.sh
```

Stop head runtime:

```bash
bash scripts/stop_head.sh
```

### 3) On each worker machine (1 GPU each)

```bash
cd /home/ali/Documents/phd/deep-distribute-ray-cluster
bash scripts/start_worker.sh
```

You can still override explicitly:

```bash
NUM_GPUS=1 bash scripts/start_worker.sh <HEAD_LAN_IP> 6379
```

Stop worker runtime:

```bash
bash scripts/stop_worker.sh
```

### 4) In UI

- Open `http://<HEAD_LAN_IP>:8080/`
- Set `Ray Address` = `auto`
- Set `GPUs per Worker` = `1`
- Set `Workers` = number of GPUs/nodes you want to use

### Dataset path requirement

`dataset_root` must be reachable from worker nodes too (shared filesystem path or same absolute path copied to all nodes).

## Windows (PowerShell) Scripts

For Windows machines, use the PowerShell versions in `scripts/`:

### 1) Setup node (once)

```powershell
cd C:\path\to\deep-distribute-ray-cluster
.\scripts\setup_node.ps1
```

### 2) Start head node

```powershell
cd C:\path\to\deep-distribute-ray-cluster
.\scripts\start_head.ps1
.\scripts\start_ui.ps1
```

### 3) Start worker node

```powershell
cd C:\path\to\deep-distribute-ray-cluster
$env:NUM_GPUS=1
.\scripts\start_worker.ps1 <HEAD_LAN_IP> 6379
```

### 4) Optional: check ports

```powershell
.\scripts\check_ports.ps1 quick
.\scripts\check_ports.ps1 full
```

### Notes for Windows

- If PowerShell blocks scripts, run once as admin:
  - `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
- Use Python Launcher (`py -3`) or a Python 3 install in PATH.
- Recommended Python on Windows for Ray: **3.12** (preferred), then **3.11** or **3.10**.
- `scripts/setup_node.ps1` now automatically prefers Python **3.12**, then **3.11**, then **3.10**.
- Keep project path consistent on all nodes where possible.

### Optional `.bat` wrappers (double-click / cmd-friendly)

Batch wrappers are included in `scripts/` and call the PowerShell scripts with safe flags:

- `setup_node.bat`
- `start_head.bat`
- `start_ui.bat`
- `stop_head.bat`
- `start_worker.bat <HEAD_IP> [HEAD_PORT]`
- `stop_worker.bat`
- `check_ports.bat [TARGET_IP] [quick|full]`

Wrappers also work without IP args when `HEAD_IP` is set in `cluster_config.env`.

