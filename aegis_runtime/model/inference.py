import time

import torch

from aegis_runtime.config import RuntimeConfig


def run_inference(model, tokenizer, config: RuntimeConfig) -> dict:
    torch.cuda.reset_peak_memory_stats()

    device = next(model.parameters()).device
    input_ids = torch.ones(
        (config.batch_size, config.max_seq_length), dtype=torch.long
    ).to(device)

    torch.cuda.synchronize()
    start = time.perf_counter()

    try:
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise

    torch.cuda.synchronize()
    end = time.perf_counter()

    elapsed = end - start
    tokens_generated = output.shape[-1] - config.max_seq_length

    return {
        "latency_ms": elapsed * 1000,
        "tokens_generated": tokens_generated,
        "tokens_per_second": tokens_generated / elapsed,
        "memory_allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
        "peak_memory_allocated_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
        "memory_reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
    }
