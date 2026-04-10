"""
Pipeline orchestrator: sequences all services in order with per-stage error
isolation so that a non-critical failure (e.g. summarization) does not abort
the entire run.

Critical stages (abort on failure): ingestion, processing, clustering, ranking
Non-critical stages (log and continue): summarization, firestore_write, notify

Writes a run log to data/pipeline_run_log.json for inspection after each run.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

# Allow running as `python backend/workers/pipeline.py` from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import shared.config as cfg
from shared.utils import get_logger, utcnow

logger = get_logger(__name__)


def _run_stage(
    name: str,
    fn,
    critical: bool = True,
    log: dict[str, Any] | None = None,
) -> Any:
    """
    Run a single pipeline stage, logging timing and handling errors.

    Args:
        name:     Stage name for logging
        fn:       Zero-argument callable that runs the stage
        critical: If True, re-raise exceptions. If False, log and return None.
        log:      Mutable dict to record stage metrics into

    Returns:
        The return value of fn(), or None on non-critical failure.
    """
    logger.info("--- Stage: %s ---", name)
    start = time.monotonic()
    try:
        result = fn()
        elapsed = time.monotonic() - start
        logger.info("Stage %s completed in %.1fs", name, elapsed)
        if log is not None:
            log[name] = {"status": "ok", "elapsed_s": round(elapsed, 2)}
        return result
    except Exception as exc:
        elapsed = time.monotonic() - start
        logger.error("Stage %s FAILED after %.1fs: %s", name, elapsed, exc, exc_info=True)
        if log is not None:
            log[name] = {"status": "error", "elapsed_s": round(elapsed, 2), "error": str(exc)}
        if critical:
            raise
        return None


def run_pipeline() -> dict[str, Any]:
    """
    Execute the full pipeline end-to-end.
    Returns a summary dict with per-stage status and timing.
    """
    run_start = time.monotonic()
    run_id = utcnow().strftime("%Y%m%dT%H%M%SZ")
    logger.info("========== Pipeline run %s started ==========", run_id)

    stage_log: dict[str, Any] = {}

    # ── Critical stages ─────────────────────────────────────────────────────

    try:
        from services.ingestion.run import run as ingest_run
        raw_articles = _run_stage("ingestion", ingest_run, critical=True, log=stage_log)
        stage_log["ingestion"]["article_count"] = len(raw_articles) if raw_articles else 0
    except Exception:
        _write_log(run_id, stage_log, run_start, "aborted:ingestion")
        raise

    try:
        from services.processing.run import run as processing_run
        normalized = _run_stage("processing", processing_run, critical=True, log=stage_log)
        stage_log["processing"]["article_count"] = len(normalized) if normalized else 0
    except Exception:
        _write_log(run_id, stage_log, run_start, "aborted:processing")
        raise

    try:
        from services.clustering.run import run as clustering_run
        clusters = _run_stage("clustering", clustering_run, critical=True, log=stage_log)
        stage_log["clustering"]["cluster_count"] = len(clusters) if clusters else 0
    except Exception:
        _write_log(run_id, stage_log, run_start, "aborted:clustering")
        raise

    try:
        from services.ranking.run import run as ranking_run
        ranked = _run_stage("ranking", ranking_run, critical=True, log=stage_log)
        stage_log["ranking"]["ranked_count"] = len(ranked) if ranked else 0
    except Exception:
        _write_log(run_id, stage_log, run_start, "aborted:ranking")
        raise

    # ── Non-critical stages ─────────────────────────────────────────────────

    from services.summarization.run import run as summarization_run
    events = _run_stage("summarization", summarization_run, critical=False, log=stage_log)
    if stage_log.get("summarization", {}).get("status") == "ok":
        stage_log["summarization"]["event_count"] = len(events) if events else 0

    from backend.workers.firestore_writer import write_events_from_file
    _run_stage("firestore_write", write_events_from_file, critical=False, log=stage_log)

    from backend.workers.notifier import run as notify_run
    _run_stage("notify", notify_run, critical=False, log=stage_log)

    total_elapsed = time.monotonic() - run_start
    logger.info(
        "========== Pipeline run %s complete in %.1fs ==========",
        run_id,
        total_elapsed,
    )

    return _write_log(run_id, stage_log, run_start, "ok")


def _write_log(
    run_id: str,
    stage_log: dict[str, Any],
    run_start: float,
    status: str,
) -> dict[str, Any]:
    summary = {
        "run_id": run_id,
        "status": status,
        "total_elapsed_s": round(time.monotonic() - run_start, 2),
        "stages": stage_log,
    }
    try:
        os.makedirs(os.path.dirname(cfg.PIPELINE_LOG_PATH) or ".", exist_ok=True)
        with open(cfg.PIPELINE_LOG_PATH, "w") as f:
            json.dump(summary, f, indent=2)
    except Exception as exc:
        logger.warning("Failed to write pipeline log: %s", exc)
    return summary


if __name__ == "__main__":
    try:
        result = run_pipeline()
        failed_stages = [k for k, v in result["stages"].items() if v.get("status") == "error"]
        if failed_stages:
            logger.warning("Pipeline completed with non-critical failures: %s", failed_stages)
            sys.exit(0)  # Don't fail the GitHub Actions run for non-critical failures
    except Exception as exc:
        logger.error("Pipeline aborted: %s", exc)
        sys.exit(1)
