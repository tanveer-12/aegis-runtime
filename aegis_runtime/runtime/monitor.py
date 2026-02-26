import torch
import pynvml


class GPUMonitor:
    def __init__(self):
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            self._total_memory_mb = memory_info.total / (1024 ** 2)
        except Exception as e:
            raise RuntimeError(
                "NVML initialization failed. Ensure you are on a machine with "
                f"NVIDIA drivers installed. Original error: {e}"
            ) from e

    def get_snapshot(self) -> dict:
        return {
            "memory_allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
            "memory_reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
            "utilization_percent": pynvml.nvmlDeviceGetUtilizationRates(self._handle).gpu,
        }

    def get_total_memory_mb(self) -> float:
        return self._total_memory_mb

    def __del__(self):
        pynvml.nvmlShutdown()
