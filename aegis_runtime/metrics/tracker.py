import logging

logger = logging.getLogger(__name__)


class MetricsTracker:
    def __init__(self):
        self._cycles = []
        self._total_recorded = 0

    def record_cycle(self, cycle_data: dict):
        self._cycles.append(cycle_data)
        self._total_recorded += 1

    def flush_to_database(self, db, trial_id: str):
        if not self._cycles:
            return
        for cycle in self._cycles:
            db.insert_inference_cycle({**cycle, "trial_id": trial_id})
        logger.info("Flushed %d cycles to database for trial %s", len(self._cycles), trial_id)

    def reset(self):
        self._cycles = []

    @property
    def count(self) -> int:
        return len(self._cycles)
