"""
AEGIS Runtime — config.py

Single source of truth for all runtime parameters.

This module defines RuntimeConfig, the only configuration class for the AEGIS
benchmark pipeline. Every parameter that controls a benchmark run lives here.
No tunable value is hard-coded anywhere else in the codebase.

Usage:
    from aegis_runtime.config import RuntimeConfig

    # Build programmatically (all fields have sensible defaults)
    cfg = RuntimeConfig(model_name="gpt2", batch_size=8, precision="bf16")

    # Load from a JSON file written by a previous run
    cfg = RuntimeConfig.from_json("experiments/baseline.json")

    # Produce a deterministic fingerprint for trial isolation checks
    trial_hash = cfg.get_hash()

Does NOT own:
- Database operations (metrics/database.py)
- Metric collection (metrics/tracker.py)
- Agent decision logic (runtime/agent.py)
"""

import json
import hashlib
import os
from pathlib import Path

from pydantic import BaseModel, field_validator

_ALLOWED_PRECISIONS = ("fp16", "bf16", "fp32")
# Get the package root directory (where aegis_runtime/ lives)
HOME = Path.home()
DATA_ROOT = HOME / "aegis-data"

# Always use home directory for permanent storage
DEFAULT_DB_PATH = str(DATA_ROOT / "experiments" / "aegis_metrics.db")
DEFAULT_LOG_DIR = str(DATA_ROOT / "logs")

# HuggingFace auto-manages model cache in ~/.cache
DEFAULT_MODEL_CACHE = None


class RuntimeConfig(BaseModel):
    """
    Flat, validated configuration for a single AEGIS benchmark experiment.

    Validation is enforced by Pydantic on every instantiation — constructing
    an invalid RuntimeConfig raises a ValidationError immediately.

    Use get_hash() to produce a deterministic MD5 fingerprint of the config,
    stored in the database as config_hash to detect drift across trials.
    """

    # ── Model ─────────────────────────────────────────────────────────────────
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # ── Benchmark parameters ──────────────────────────────────────────────────
    batch_size: int = 4
    precision: str = "fp16"
    max_seq_length: int = 256
    num_trials: int = 3
    random_seed_base: int = 42

    # ── Storage paths ─────────────────────────────────────────────────────────
    # Default paths anchored to package root
    db_path: str = DEFAULT_DB_PATH
    log_dir: str = DEFAULT_LOG_DIR
    model_cache_dir : str = str(Path(HOME) / "aegis-data" / "models") if HOME else None

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("batch_size")
    @classmethod
    def _batch_size_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"batch_size must be > 0, got {v}")
        return v

    @field_validator("max_seq_length")
    @classmethod
    def _max_seq_length_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"max_seq_length must be > 0, got {v}")
        return v

    @field_validator("num_trials")
    @classmethod
    def _num_trials_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"num_trials must be > 0, got {v}")
        return v

    @field_validator("precision")
    @classmethod
    def _precision_allowed(cls, v: str) -> str:
        if v not in _ALLOWED_PRECISIONS:
            raise ValueError(
                f"precision must be one of {_ALLOWED_PRECISIONS}, got {v!r}"
            )
        return v

    @field_validator("db_path", "log_dir")
    @classmethod
    def _path_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Path fields must be non-empty strings")
        return v

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return all fields as a plain dict (suitable for database storage)."""
        return self.model_dump()

    def to_json(self) -> str:
        """Return a pretty-printed JSON string of this configuration."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, filepath: str) -> "RuntimeConfig":
        """
        Load a RuntimeConfig from a JSON file on disk.

        Args:
            filepath: Path to a JSON file previously written by to_json().

        Returns:
            A fully validated RuntimeConfig instance.

        Raises:
            FileNotFoundError: If filepath does not exist.
            pydantic.ValidationError: If the JSON contains invalid field values.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        with path.open() as fh:
            data = json.load(fh)
        return cls(**data)

    # ── Hashing ───────────────────────────────────────────────────────────────

    def get_hash(self) -> str:
        """
        Return an MD5 hex digest of the serialized configuration.

        Keys are sorted before hashing so the digest is deterministic
        regardless of field insertion order. This value is stored in the
        database as config_hash to detect parameter drift across trials.
        """
        canonical = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(canonical.encode()).hexdigest()