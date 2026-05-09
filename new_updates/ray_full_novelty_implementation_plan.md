# Ray Full Novelty Implementation Plan

This plan maps the remaining novelty points into concrete implementation phases for `deep-distribute-ray-cluster`.

## Phase 1 (Implemented now): SSP Versioning + Resync

- Add `ssp_staleness` configuration.
- Workers attach `base_version` with each submitted update.
- PS rejects stale updates (`base_version < global_version - ssp_staleness`).
- PS returns `resync_required` + latest weights.
- Worker performs immediate resync and restarts delta chain.

Status: implemented in:
- `ray_ps_async/config.py`
- `ray_ps_async/actors.py`

## Phase 2: Quorum + Timeout Micro-Round Scheduler

- Move from per-arrival immediate apply to micro-round buffering.
- Introduce round buffers keyed by `global_version`.
- Aggregate when either:
  - quorum reached (`ceil(quorum_fraction * num_workers)`) or
  - timeout reached (`round_timeout_ms`).
- Persist round diagnostics:
  - number of updates, wait time, stale drops.

Target files:
- `ray_ps_async/actors.py` (PS state + scheduling loop)
- `ray_ps_async/config.py` (add quorum_fraction, round_timeout_ms)

## Phase 3: Compression + Error Feedback

- Add pluggable codecs:
  - `fp16`
  - `q8`
  - `topk` sparsification
- Add residual error feedback on worker to recover convergence.
- Add compression stats in communication report:
  - raw bytes vs transmitted bytes
  - compression ratio by method.

Target files:
- `ray_ps_async/serialization.py` (codec interfaces)
- `ray_ps_async/actors.py` (encode/decode paths)

## Phase 4: Push-First Synchronization

- Replace purely request-response weight pull with push-style update stream.
- In Ray actor model, PS can asynchronously notify worker actors after round commit.
- Workers can apply pushed global updates immediately to reduce staleness.

Target files:
- `ray_ps_async/actors.py` (PS maintains worker handles)

## Phase 5: Central Optimizer State on PS

- Replace direct `w += lr * grad` with central optimizer state:
  - SGD momentum / Adam moments.
- Workers compute local gradient signal only.
- PS maintains optimizer slots and applies updates consistently.

Target files:
- `ray_ps_async/actors.py` (PS optimizer states)

## Phase 6: Adaptive Sync/Throttle

- Dynamic `sync_every_examples` per worker based on:
  - observed stale rejections,
  - network delay,
  - queue pressure.
- Expose adaptive decisions in run telemetry.

Target files:
- `ray_ps_async/actors.py`
- `api.py` (surface adaptive stats)

## Phase 7: Evaluation and Ablation Protocol

- Add experiment presets and compare:
  1. baseline async PS
  2. +SSP
  3. +quorum/timeout
  4. +compression
  5. full system
- Produce clean tables and graphs in UI and exported JSON/CSV.

Target files:
- `api.py` (preset selection + reporting)
- `new_updates/` (analysis scripts/results templates)
