import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import api
from fastapi.testclient import TestClient


def test_health():
    client = TestClient(api.app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_experiment_lifecycle_monkeypatched(monkeypatch):
    def fake_runner():
        def _run(_cfg):
            return {
                "test_metrics": {"accuracy": 0.9, "precision": 0.8, "recall": 0.85, "f1_score": 0.82, "auc": 0.91},
                "communication_cost": {"bytes_worker_to_ps": 1024, "bytes_ps_to_worker": 2048, "total_mb": 0.0029, "messages": 10},
            }

        return _run

    monkeypatch.setattr(api, "_experiment_runner", fake_runner)
    client = TestClient(api.app)

    payload = {
        "dataset_root": "/tmp/dummy",
        "num_workers": 1,
        "epochs": 1,
        "batch_size": 2,
        "image_height": 224,
        "image_width": 224,
        "learning_rate": 0.001,
        "sync_every_examples": 2,
        "num_gpus_per_worker": 0.0,
        "ray_address": "local",
        "data_mode": "shared_path",
        "allow_gpu_fallback": True,
        "model_name": "tiny",
    }
    start = client.post("/experiments", json=payload)
    assert start.status_code == 200
    job_id = start.json()["job_id"]

    # Background task completes before response with TestClient.
    get_one = client.get(f"/experiments/{job_id}")
    assert get_one.status_code == 200
    body = get_one.json()
    assert body["status"] == "completed"
    assert body["result"]["test_metrics"]["accuracy"] == 0.9

    list_jobs = client.get("/experiments")
    assert list_jobs.status_code == 200
    assert job_id in list_jobs.json()
