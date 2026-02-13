import pytest
import json
import threading
from pathlib import Path
from datetime import datetime, timezone
import uuid

from config import DEFAULT_CONFIG, config_to_dict
from metrics.database import MetricsDatabase


@pytest.fixture
def test_db():
    """
    Fixture to create a fresh test database for each test and clean it up afterwards.
    """
    db_path = "logs/test_metrics.db"
    # Ensure old test DB is removed
    if Path(db_path).exists():
        Path(db_path).unlink()
    db = MetricsDatabase(db_path)
    yield db
    # Clean up after test
    if Path(db_path).exists():
        Path(db_path).unlink()


def test_default_config_values():
    """
    Test 1 — Default config values are set correctly.
    """
    assert DEFAULT_CONFIG.model.model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert DEFAULT_CONFIG.agent.memory_decrease_threshold_pct == 85.0
    assert DEFAULT_CONFIG.runtime.min_batch_size < DEFAULT_CONFIG.runtime.initial_batch_size < DEFAULT_CONFIG.runtime.max_batch_size


def test_config_to_dict_is_serializable():
    """
    Test 2 — config_to_dict returns a dict and is JSON-serializable.
    """
    cfg_dict = config_to_dict(DEFAULT_CONFIG)
    assert isinstance(cfg_dict, dict)
    # Test JSON serialization
    try:
        json.dumps(cfg_dict)
    except Exception:
        pytest.fail("config_to_dict output is not JSON-serializable")


def test_database_creates_file():
    """
    Test 3 — MetricsDatabase creates the database file.
    """
    db_path = "logs/test_metrics.db"
    if Path(db_path).exists():
        Path(db_path).unlink()
    db = MetricsDatabase(db_path)
    assert Path(db_path).exists()
    # Clean up
    Path(db_path).unlink()


def test_insert_and_retrieve_cycle(test_db):
    """
    Test 4 — Insert a record and retrieve it.
    """
    run_id = str(uuid.uuid4())
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "batch_size": 4,
        "sequence_length": 256,
        "precision": "fp16",
        "tokens_generated": 100,
        "tokens_per_second": 50.0,
        "latency_ms": 200.0,
        "gpu_memory_allocated_mb": 1024.0,
        "gpu_memory_peak_mb": 1200.0,
        "gpu_utilization_pct": 75.0,
        "oom_event": 0,
        "agent_decision": None,
        "run_id": run_id
    }
    row_id = test_db.insert_inference_cycle(record)
    assert isinstance(row_id, int) and row_id > 0

    recent = test_db.get_recent_cycles(limit=1)
    assert len(recent) == 1
    assert recent[0]["tokens_per_second"] == 50.0


def test_run_summary_aggregates_correctly(test_db):
    """
    Test 5 — Insert multiple cycles and verify aggregated stats.
    """
    run_id = str(uuid.uuid4())
    values = [10.0, 20.0, 30.0]
    for v in values:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "batch_size": 4,
            "sequence_length": 256,
            "precision": "fp16",
            "tokens_generated": 100,
            "tokens_per_second": v,
            "latency_ms": 200.0,
            "gpu_memory_allocated_mb": 1024.0,
            "gpu_memory_peak_mb": 1200.0,
            "gpu_utilization_pct": 75.0,
            "oom_event": 0,
            "agent_decision": None,
            "run_id": run_id
        }
        test_db.insert_inference_cycle(record)

    summary = test_db.get_run_summary(run_id)
    assert summary["cycle_count"] == 3
    assert abs(summary["avg_tokens_per_second"] - 20.0) < 1e-5


def test_concurrent_inserts_do_not_corrupt(test_db):
    """
    Test 6 — Verify threading lock prevents DB corruption.
    """
    run_id = str(uuid.uuid4())
    def insert_dummy(i):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "batch_size": 4,
            "sequence_length": 256,
            "precision": "fp16",
            "tokens_generated": 100,
            "tokens_per_second": float(i),
            "latency_ms": 200.0,
            "gpu_memory_allocated_mb": 1024.0,
            "gpu_memory_peak_mb": 1200.0,
            "gpu_utilization_pct": 75.0,
            "oom_event": 0,
            "agent_decision": None,
            "run_id": run_id
        }
        test_db.insert_inference_cycle(record)

    threads = [threading.Thread(target=insert_dummy, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    recent = test_db.get_recent_cycles(limit=20)
    count = sum(1 for r in recent if r["run_id"] == run_id)
    assert count == 10
