"""
AEGIS Runtime — database.py

Responsibility: All SQLite database operations.

Owns:
- Database connection management
- Schema initialization from schema.sql
- Insert operations for experiments, trials, cycles, summary
- Query operations for dashboard and analysis
- Transaction handling

Does NOT own:
- Statistical computation (analyzer.py)
- Metric accumulation logic (tracker.py)
- Logging to files (logger.py)
"""

import sqlite3
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages all SQLite operations for AEGIS Runtime."""

    def __init__(self, db_path: str):
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.schema_path = Path(__file__).parent / "schema.sql"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def initialize_schema(self):
        """Create all tables from schema.sql if they don't exist."""
        schema_sql = self.schema_path.read_text()
        with self.get_connection() as conn:
            conn.executescript(schema_sql)
        logger.info("Schema initialized from %s", self.schema_path)

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def insert_experiment(self, experiment_data: Dict[str, Any]) -> str:
        """Insert a new experiment and return its ID."""
        sql = """
            INSERT INTO experiments (
                experiment_id, model_name, gpu_name, gpu_memory_total_gb,
                hostname, cuda_version, driver_version, pytorch_version,
                transformers_version, git_commit_hash, config_description
            ) VALUES (
                :experiment_id, :model_name, :gpu_name, :gpu_memory_total_gb,
                :hostname, :cuda_version, :driver_version, :pytorch_version,
                :transformers_version, :git_commit_hash, :config_description
            )
        """
        with self.get_connection() as conn:
            conn.execute(sql, experiment_data)
            conn.commit()
        logger.debug("Inserted experiment %s", experiment_data["experiment_id"])
        return experiment_data["experiment_id"]

    def insert_trial(self, trial_data: Dict[str, Any]) -> str:
        """Insert a new trial and return its ID."""
        sql = """
            INSERT INTO trials (
                trial_id, experiment_id, trail_number, config_hash,
                random_seed, status
            ) VALUES (
                :trial_id, :experiment_id, :trail_number, :config_hash,
                :random_seed, :status
            )
        """
        with self.get_connection() as conn:
            conn.execute(sql, trial_data)
            conn.commit()
        logger.debug("Inserted trial %s", trial_data["trial_id"])
        return trial_data["trial_id"]

    def insert_inference_cycle(self, cycle_data: Dict[str, Any]):
        """Insert a new inference cycle record."""
        sql = """
            INSERT INTO inference_cycles (
                trial_id, cycle_number, batch_size, max_seq_length, precision,
                tokens_generated, tokens_per_second, latency_ms,
                gpu_memory_allocated_mb, gpu_memory_reserved_mb,
                gpu_utilization_percent, agent_action, agent_reason, oom_event
            ) VALUES (
                :trial_id, :cycle_number, :batch_size, :max_seq_length, :precision,
                :tokens_generated, :tokens_per_second, :latency_ms,
                :gpu_memory_allocated_mb, :gpu_memory_reserved_mb,
                :gpu_utilization_percent, :agent_action, :agent_reason, :oom_event
            )
        """
        with self.get_connection() as conn:
            conn.execute(sql, cycle_data)
            conn.commit()
        logger.debug("Inserted cycle %s for trial %s",
                     cycle_data["cycle_number"], cycle_data["trial_id"])

    def update_trial_status(self, trial_id: str, status: str,
                            completed_at: str = None,
                            failure_reason: str = None):
        """Update trial status when it completes or fails."""
        sql = """
            UPDATE trials
            SET status = :status,
                completed_at = :completed_at,
                failure_reason = :failure_reason
            WHERE trial_id = :trial_id
        """
        with self.get_connection() as conn:
            conn.execute(sql, {
                "trial_id": trial_id,
                "status": status,
                "completed_at": completed_at,
                "failure_reason": failure_reason,
            })
            conn.commit()
        logger.debug("Updated trial %s → status=%s", trial_id, status)

    def insert_experiment_summary(self, summary_data: Dict[str, Any]):
        """Insert computed statistics for an experiment."""
        sql = """
            INSERT INTO experiment_summary (
                experiment_id, num_trials,
                mean_latency_ms, std_latency_ms, min_latency_ms, max_latency_ms,
                mean_tokens_per_sec, std_tokens_per_sec,
                min_tokens_per_sec, max_tokens_per_sec,
                peak_memory_allocated_mb, peak_memory_reserved_mb,
                total_oom_events, oom_frequency,
                successful_trials, failed_trials
            ) VALUES (
                :experiment_id, :num_trials,
                :mean_latency_ms, :std_latency_ms, :min_latency_ms, :max_latency_ms,
                :mean_tokens_per_sec, :std_tokens_per_sec,
                :min_tokens_per_sec, :max_tokens_per_sec,
                :peak_memory_allocated_mb, :peak_memory_reserved_mb,
                :total_oom_events, :oom_frequency,
                :successful_trials, :failed_trials
            )
        """
        with self.get_connection() as conn:
            conn.execute(sql, summary_data)
            conn.commit()
        logger.debug("Inserted summary for experiment %s",
                     summary_data["experiment_id"])

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve experiment metadata."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?",
                (experiment_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_trials(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Get all trials for an experiment, ordered by trial number."""
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM trials WHERE experiment_id = ? ORDER BY trail_number",
                (experiment_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_inference_cycles(self, trial_id: str) -> List[Dict[str, Any]]:
        """Get all cycles for a trial, ordered by cycle number."""
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM inference_cycles WHERE trial_id = ? ORDER BY cycle_number",
                (trial_id,)
            ).fetchall()
        return [dict(r) for r in rows]
