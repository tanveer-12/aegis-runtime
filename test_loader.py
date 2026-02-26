"""
Smoke test for the model loader.

Run from the project root after activating your environment:
    python test_loader.py

Expected output (GPT-2 on a CUDA node):
    Device : cuda:0
    Params : 124,439,808
    Dtype  : torch.float16
    Training mode: False
"""

from aegis_runtime.config import RuntimeConfig
from aegis_runtime.model.loader import load_model_and_tokenizer

config = RuntimeConfig(model_name="gpt2", precision="fp16")

model, tokenizer = load_model_and_tokenizer(config)

param_count = sum(p.numel() for p in model.parameters())
first_param = next(model.parameters())

print(f"Device : {first_param.device}")
print(f"Params : {param_count:,}")
print(f"Dtype  : {first_param.dtype}")
print(f"Training mode: {model.training}")
