"""
Aegis Runtime Configuration
===========================

This module defines the single source of truth for all tunable parameters
in the Aegis Runtime project. Every configuration value lives here.

Usage:
    from config import DEFAULT_CONFIG, config_to_dict
    
    # Access config values
    batch_size = DEFAULT_CONFIG.runtime.initial_batch_size
    model_name = DEFAULT_CONFIG.model.model_name
    
    # Get full config as dict for logging
    config_dict = config_to_dict(DEFAULT_CONFIG)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the model itself."""
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    max_new_tokens: int = 128
    device: str = "cuda"


@dataclass
class RuntimeConfig:
    """Configuration for runtime parameters."""
    initial_batch_size: int = 4
    min_batch_size: int = 1
    max_batch_size: int = 16
    initial_precision: str = "fp16"
    initial_seq_length: int = 256
    min_seq_length: int = 64
    max_seq_length: int = 512


@dataclass
class AgentConfig:
    """Configuration for the agent's decision thresholds."""
    memory_increase_threshold_pct: float = 60.0
    memory_decrease_threshold_pct: float = 85.0
    oom_recovery_divisor: int = 2


@dataclass
class LoggingConfig:
    """Configuration for logging and metrics."""
    db_path: str = "logs/aegis_metrics.db"
    log_dir: str = "logs"
    console_log_level: str = "INFO"


@dataclass
class AegisConfig:
    """Top-level configuration containing all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# Default configuration instance - import this anywhere config is needed
DEFAULT_CONFIG = AegisConfig()


def config_to_dict(config: AegisConfig) -> dict:
    """
    Convert an AegisConfig to a nested dictionary.
    
    Uses dataclasses.asdict() for recursive conversion.
    Useful for logging the full configuration at the start of each run.
    
    Args:
        config: An AegisConfig instance
        
    Returns:
        A nested dictionary with all configuration values
    """
    return {
        "model": {
            "model_name": config.model.model_name,
            "max_new_tokens": config.model.max_new_tokens,
            "device": config.model.device,
        },
        "runtime": {
            "initial_batch_size": config.runtime.initial_batch_size,
            "min_batch_size": config.runtime.min_batch_size,
            "max_batch_size": config.runtime.max_batch_size,
            "initial_precision": config.runtime.initial_precision,
            "initial_seq_length": config.runtime.initial_seq_length,
            "min_seq_length": config.runtime.min_seq_length,
            "max_seq_length": config.runtime.max_seq_length,
        },
        "agent": {
            "memory_increase_threshold_pct": config.agent.memory_increase_threshold_pct,
            "memory_decrease_threshold_pct": config.agent.memory_decrease_threshold_pct,
            "oom_recovery_divisor": config.agent.oom_recovery_divisor,
        },
        "logging": {
            "db_path": config.logging.db_path,
            "log_dir": config.logging.log_dir,
            "console_log_level": config.logging.console_log_level,
        }
    }
