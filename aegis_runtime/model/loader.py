import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple
from aegis_runtime.config import RuntimeConfig

logger = logging.getLogger(__name__)

_PRECISION_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def load_model_and_tokenizer(config: RuntimeConfig) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA-capable GPU detected. aegis-runtime requires a GPU to load models."
        )

    torch_dtype = _PRECISION_MAP.get(config.precision)
    if torch_dtype is None:
        raise ValueError(
            f"Invalid precision {config.precision!r}. "
            f"Accepted values: {list(_PRECISION_MAP.keys())}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        cache_dir=config.model_cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer loaded: %s", config.model_name)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        cache_dir=config.model_cache_dir,
    )
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model loaded | name=%s | params=%.2fM | dtype=%s | device=%s",
        config.model_name,
        param_count / 1e6,
        next(model.parameters()).dtype,
        next(model.parameters()).device,
    )

    return model, tokenizer