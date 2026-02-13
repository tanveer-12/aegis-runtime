"""
SQLite Database Module for Aegis Runtime Metrics
=================================================

This module owns all SQLite operations. No other module writes raw SQL —
they call MetricsDatabase methods. This separation means you can swap
SQLite for PostgreSQL in the future by changing exactly this file.

Thread Safety:
    Uses a threading.Lock to serialize all writes. Reads can parallelize
    safely in SQLite.

Database Schema:
    - inference_cycles: One row per inference run
    - agent_decisions: One row per agent control variable change
"""

import sqlite3
import threading
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

# Configure module logger
logger = logging.getLogger(__name__)

# Thread lock for serializing database writes
_db_lock = threading.Lock()


class MetricsDatabase:
    """
    Thread-safe SQLite database interface for Aegis metrics.
    
    Provides methods for inserting inference cycles and agent decisions,
    as well as querying recent data and run summaries.
    """
    
    def __init__(self, db_path: str = "logs/aegis_metrics.db"):
        """
        Initialize the database connection and create schema.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        # Create parent directory if it doesn't exist
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initializing database at {db_path}")
        self._initialize_schema()
    
    def _initialize_schema(self) -> None:
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table: inference_cycles
        # Every single inference run — one row per cycle
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inference_cycles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                batch_size INTEGER NOT NULL,
                sequence_length INTEGER NOT NULL,
                precision TEXT NOT NULL,
                tokens_generated INTEGER NOT NULL,
                tokens_per_second REAL NOT NULL,
                latency_ms REAL NOT NULL,
                gpu_memory_allocated_mb REAL NOT NULL,
                gpu_memory_peak_mb REAL NOT NULL,
                gpu_utilization_pct REAL NOT NULL,
                oom_event INTEGER NOT NULL DEFAULT 0,
                agent_decision INTEGER,
                run_id TEXT NOT NULL,
                FOREIGN KEY (agent_decision) REFERENCES agent_decisions(id)
            )
        """)
        
        # Table: agent_decisions
        # Every time the agent changes a control variable, one row here
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                trigger TEXT NOT NULL,
                old_batch_size INTEGER,
                new_batch_size INTEGER,
                old_precision TEXT,
                new_precision TEXT,
                old_seq_length INTEGER,
                new_seq_length INTEGER,
                memory_pct_at_decision REAL,
                reason TEXT NOT NULL,
                run_id TEXT NOT NULL
            )
        """)
        
        # Create indexes for common query patterns
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_inference_cycles_run_id 
            ON inference_cycles(run_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_inference_cycles_timestamp 
            ON inference_cycles(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_decisions_run_id 
            ON agent_decisions(run_id)
        """)
        
        conn.commit()
        conn.close()
        logger.info("Database schema initialized successfully")
    
    def insert_inference_cycle(self, record: dict) -> int:
        """
        Insert an inference cycle record into the database.
        
        Args:
            record: Dictionary with keys matching inference_cycles columns:
                - timestamp (str): ISO 8601 UTC timestamp
                - batch_size (int): Number of sequences in batch
                - sequence_length (int): Token length cap used
                - precision (str): fp16 or bf16
                - tokens_generated (int): Total tokens output
                - tokens_per_second (float): Performance metric
                - latency_ms (float): Wall-clock time for inference
                - gpu_memory_allocated_mb (float): PyTorch allocated at end
                - gpu_memory_peak_mb (float): PyTorch peak reserved
                - gpu_utilization_pct (float): GPU utilization
                - oom_event (bool/int): 1 if OOM occurred
                - run_id (str): UUID for the benchmark run
                - agent_decision (int, optional): FK to agent_decisions
        
        Returns:
            The auto-generated ID of the inserted row
            
        Raises:
            Exception: Re-raises any database error with details
        """
        with _db_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    INSERT INTO inference_cycles (
                        timestamp, batch_size, sequence_length, precision,
                        tokens_generated, tokens_per_second, latency_ms,
                        gpu_memory_allocated_mb, gpu_memory_peak_mb,
                        gpu_utilization_pct, oom_event, agent_decision, run_id
                    ) VALUES (
                        :timestamp, :batch_size, :sequence_length, :precision,
                        :tokens_generated, :tokens_per_second, :latency_ms,
                        :gpu_memory_allocated_mb, :gpu_memory_peak_mb,
                        :gpu_utilization_pct, :oom_event, :agent_decision, :run_id
                    )
                """, record)
                
                conn.commit()
                row_id = cursor.lastrowid
                logger.debug(f"Inserted inference cycle with id={row_id}")
                return row_id
                
            except Exception as e:
                logger.error(f"Error inserting inference cycle: {e}")
                logger.error(f"Record that caused error: {record}")
                raise
                
            finally:
                conn.close()
    
    def insert_agent_decision(self, record: dict) -> int:
        """
        Insert an agent decision record into the database.
        
        Args:
            record: Dictionary with keys matching agent_decisions columns:
                - timestamp (str): ISO 8601 UTC timestamp
                - trigger (str): What caused the decision
                - old_batch_size (int, optional): Before change
                - new_batch_size (int, optional): After change
                - old_precision (str, optional): Before change
                - new_precision (str, optional): After change
                - old_seq_length (int, optional): Before change
                - new_seq_length (int, optional): After change
                - memory_pct_at_decision (float, optional): Memory % trigger
                - reason (str): Human-readable explanation
                - run_id (str): Links to inference_cycles
        
        Returns:
            The auto-generated ID of the inserted row
            
        Raises:
            Exception: Re-raises any database error with details
        """
        with _db_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    INSERT INTO agent_decisions (
                        timestamp, trigger,
                        old_batch_size, new_batch_size,
                        old_precision, new_precision,
                        old_seq_length, new_seq_length,
                        memory_pct_at_decision, reason, run_id
                    ) VALUES (
                        :timestamp, :trigger,
                        :old_batch_size, :new_batch_size,
                        :old_precision, :new_precision,
                        :old_seq_length, :new_seq_length,
                        :memory_pct_at_decision, :reason, :run_id
                    )
                """, record)
                
                conn.commit()
                row_id = cursor.lastrowid
                logger.debug(f"Inserted agent decision with id={row_id}")
                return row_id
                
            except Exception as e:
                logger.error(f"Error inserting agent decision: {e}")
                logger.error(f"Record that caused error: {record}")
                raise
                
            finally:
                conn.close()
    
    def get_recent_cycles(
        self, 
        run_id: Optional[str] = None, 
        limit: int = 100
    ) -> list[dict]:
        """
        Get recent inference cycles, optionally filtered by run_id.
        
        Args:
            run_id: Optional UUID to filter by specific run
            limit: Maximum number of records to return (default 100)
            
        Returns:
            List of dictionaries representing inference cycles
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if run_id:
            cursor.execute("""
                SELECT * FROM inference_cycles 
                WHERE run_id = :run_id
                ORDER BY timestamp DESC
                LIMIT :limit
            """, {"run_id": run_id, "limit": limit})
        else:
            cursor.execute("""
                SELECT * FROM inference_cycles 
                ORDER BY timestamp DESC
                LIMIT :limit
            """, {"limit": limit})
        
        rows = cursor.fetchall()
        conn.close()
        
        # Convert sqlite3.Row to plain dict for JSON serialization
        return [dict(row) for row in rows]
    
    def get_run_summary(self, run_id: str) -> dict:
        """
        Get aggregate statistics for a specific run.
        
        Args:
            run_id: UUID of the run to summarize
            
        Returns:
            Dictionary with aggregate statistics:
                - cycle_count: Total number of cycles
                - avg_tokens_per_second: Average throughput
                - max_tokens_per_second: Peak throughput
                - avg_latency_ms: Average latency
                - oom_count: Number of OOM events
                - avg_gpu_memory_mb: Average GPU memory usage
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as cycle_count,
                AVG(tokens_per_second) as avg_tokens_per_second,
                MAX(tokens_per_second) as max_tokens_per_second,
                AVG(latency_ms) as avg_latency_ms,
                SUM(oom_event) as oom_count,
                AVG(gpu_memory_allocated_mb) as avg_gpu_memory_mb
            FROM inference_cycles 
            WHERE run_id = :run_id
        """, {"run_id": run_id})
        
        row = cursor.fetchone()
        conn.close()
        
        return dict(row) if row else {}
    
    def list_runs(self) -> list[dict]:
        """
        List all distinct runs with their metadata.
        
        Returns:
            List of run summaries containing:
                - run_id: Unique identifier
                - start_time: First cycle timestamp
                - end_time: Last cycle timestamp
                - cycle_count: Number of cycles in run
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                run_id,
                MIN(timestamp) as start_time,
                MAX(timestamp) as end_time,
                COUNT(*) as cycle_count
            FROM inference_cycles 
            GROUP BY run_id
            ORDER BY start_time DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]


# Module-level singleton instance for convenience
_default_db: Optional[MetricsDatabase] = None


def get_database(db_path: str = "logs/aegis_metrics.db") -> MetricsDatabase:
    """
    Get or create the default database instance.
    
    Args:
        db_path: Path to the SQLite database
        
    Returns:
        A MetricsDatabase instance
    """
    global _default_db
    if _default_db is None:
        _default_db = MetricsDatabase(db_path)
    return _default_db
