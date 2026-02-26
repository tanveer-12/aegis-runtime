import socket
import subprocess
import uuid

import pynvml
import torch
import transformers

from aegis_runtime.config import RuntimeConfig
from aegis_runtime.metrics.database import DatabaseManager
from aegis_runtime.metrics.logger import setup_trial_logger
from aegis_runtime.metrics.tracker import MetricsTracker
from aegis_runtime.model.inference import run_inference
from aegis_runtime.model.loader import load_model_and_tokenizer
from aegis_runtime.runtime.monitor import GPUMonitor


def run_single_trial_benchmark(config: RuntimeConfig):
    # ── Initialization ────────────────────────────────────────────────────────
    db = DatabaseManager(config.db_path)
    db.initialize_schema()

    experiment_id = str(uuid.uuid4())

    git_result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    )
    git_commit_hash = git_result.stdout.strip()

    monitor = GPUMonitor()
    gpu_name = torch.cuda.get_device_name(0)

    db.insert_experiment({
        "experiment_id": experiment_id,
        "model_name": config.model_name,
        "gpu_name": gpu_name,
        "gpu_memory_total_gb": monitor.get_total_memory_mb() / 1024,
        "hostname": socket.gethostname(),
        "cuda_version": torch.version.cuda or "unknown",
        "driver_version": str(pynvml.nvmlSystemGetDriverVersion()),
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "git_commit_hash": git_commit_hash,
        "config_description": config.model_dump().get("config_description"),
    })

    config.total_gpu_memory_mb = monitor.get_total_memory_mb()

    trial_logger = setup_trial_logger(experiment_id, config.log_dir)

    # ── Model loading ─────────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(config)
    param_count = sum(p.numel() for p in model.parameters())
    trial_logger.info(
        "Model loaded",
        extra={"model_name": config.model_name, "param_count": param_count},
    )

    # ── Trial setup ───────────────────────────────────────────────────────────
    trial_id = str(uuid.uuid4())
    config_hash = config.get_hash()
    random_seed = config.random_seed_base

    db.insert_trial({
        "trial_id": trial_id,
        "experiment_id": experiment_id,
        "trail_number": 1,
        "config_hash": config_hash,
        "random_seed": random_seed,
        "status": "running",
    })

    tracker = MetricsTracker()

    # ── Cycle loop ────────────────────────────────────────────────────────────
    for cycle_num in range(config.num_cycles):
        monitor.get_snapshot()  # pre-inference snapshot (unused but documents intent)
        inference_result = run_inference(model, tokenizer, config)
        post_snapshot = monitor.get_snapshot()

        cycle_data = {
            "cycle_number": cycle_num + 1,
            "batch_size": config.batch_size,
            "max_seq_length": config.max_seq_length,
            "precision": config.precision,
            "tokens_generated": inference_result["tokens_generated"],
            "tokens_per_second": inference_result["tokens_per_second"],
            "latency_ms": inference_result["latency_ms"],
            "gpu_memory_allocated_mb": post_snapshot["memory_allocated_mb"],
            "gpu_memory_reserved_mb": inference_result["memory_reserved_mb"],
            "gpu_utilization_percent": post_snapshot["utilization_percent"],
            "estimated_memory_mb": None,
            "estimation_error_pct": None,
            "agent_action": None,
            "agent_reason": None,
            "oom_event": False,
        }
        tracker.record_cycle(cycle_data)
        trial_logger.debug(
            "Cycle complete",
            extra={"cycle": cycle_num + 1, "tokens_per_second": inference_result["tokens_per_second"]},
        )

    # ── Wrap up ───────────────────────────────────────────────────────────────
    tracker.flush_to_database(db, trial_id)
    db.update_trial_status(trial_id, status="completed")

    cycles = tracker._cycles
    mean_tps = sum(c["tokens_per_second"] for c in cycles) / len(cycles)
    mean_lat = sum(c["latency_ms"] for c in cycles) / len(cycles)
    peak_mem = max(c["gpu_memory_allocated_mb"] for c in cycles)

    print(
        f"\n{'='*60}\n"
        f"  experiment_id : {experiment_id}\n"
        f"  cycles        : {len(cycles)}\n"
        f"  mean tps      : {mean_tps:.1f} tokens/sec\n"
        f"  mean latency  : {mean_lat:.1f} ms\n"
        f"  peak memory   : {peak_mem:.1f} MB\n"
        f"{'='*60}\n"
    )
