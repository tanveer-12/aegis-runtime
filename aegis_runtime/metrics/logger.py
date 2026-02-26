import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_STANDARD_LOG_KEYS = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "funcName", "lineno", "created", "msecs", "relativeCreated",
    "thread", "threadName", "processName", "process", "message",
    "exc_info", "exc_text", "stack_info",
})


class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        record.getMessage()  # populates record.message
        extra = {k: v for k, v in record.__dict__.items() if k not in _STANDARD_LOG_KEYS}
        return json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
            "context": extra,
        })


def setup_trial_logger(trial_id: str, log_dir: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(Path(log_dir) / f"{trial_id}.log")
    file_handler.setFormatter(_JSONFormatter())
    file_handler.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(logging.WARNING)

    logger = logging.getLogger(f"aegis.trial.{trial_id}")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False

    return logger
