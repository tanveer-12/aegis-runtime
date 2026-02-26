# AEGIS Runtime — Master Project Specification

**Version:** 2.0  
**Last Updated:** February 18, 2026  
**Status:** Active Development

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Development Environment](#development-environment)
4. [Git Workflow](#git-workflow)
5. [Directory Structure](#directory-structure)
6. [4-Week Implementation Roadmap](#4-week-implementation-roadmap)
7. [Module Responsibilities](#module-responsibilities)
8. [Database Schema](#database-schema)
9. [Coding Standards](#coding-standards)
10. [Testing Protocol](#testing-protocol)
11. [Deployment Checklist](#deployment-checklist)

---

## Project Overview

### Mission Statement

AEGIS Runtime is a **modular, GPU-aware adaptive LLM inference runtime** with controlled benchmarking capabilities. The system dynamically adjusts batch size, precision, and sequence length to maximize tokens-per-second while preventing OOM errors, with statistically validated multi-trial experiments.

### Core Principles

1. **Model Agnostic** — Support any HuggingFace causal language model
2. **Observable** — Every decision, every cycle, every trial logged with timestamps
3. **Isolated** — Trials are hermetically sealed units with memory/seed resets
4. **Reproducible** — Git commit hash + config = identical results months later
5. **Research-Grade** — Statistical validation across multiple trials mandatory

### Key Features

- **Adaptive Agent:** Dynamically adjusts batch size and precision based on GPU state
- **Multi-Trial Framework:** 5-10 trials per experiment with statistical analysis
- **Comprehensive Logging:** SQLite database captures experiment → trial → cycle hierarchy
- **HPC Ready:** SLURM integration for cluster execution
- **Model Comparison:** Benchmark multiple models under identical constraints

---

## System Architecture

### Execution Model

- **Mode:** Autoregressive inference only (no training)
- **Supported Models:** Decoder-only transformers (GPT-2, GPT-Neo, Llama, Mistral, TinyLlama, etc.)
- **Optimization Goal:** Maximize tokens/sec with zero OOM crashes

### Agent Control Variables

- Batch size
- Precision (fp16, bf16, fp32)
- Maximum sequence length cap

### Agent Observables

- GPU memory allocated
- GPU max memory reserved
- GPU utilization %
- Tokens/sec
- Latency (ms)
- OOM events

### Statistical Requirements

**Per Experiment (across all trials):**
- Mean latency ± std deviation
- Mean tokens/sec ± std deviation
- Peak memory usage
- OOM frequency
- Trial success rate

**Validity Threshold:** Std deviation < 20% of mean (indicates stable, reproducible performance)

---

## Development Environment

### Local Development (Windows + WSL)

**Hardware:**
- Windows 11
- WSL2 (Ubuntu 20.04/22.04)
- NVIDIA GPU (for local testing, optional)

**Software:**
- VS Code with Remote WSL extension
- GitHub Desktop (for git operations)
- Miniconda/Anaconda
- Git

**Purpose:**
- Code development
- Local testing (optional)
- Dashboard visualization of results

### Production Execution (HPC Cluster)

**Hardware:**
- University HPC cluster
- SLURM scheduler
- NVIDIA GPUs: A30 (24GB), A100 (40-80GB), L40S (48GB), H100 (80GB), H200 (141GB)
- CUDA 12.1+

**Software:**
- Linux (CentOS/Ubuntu)
- Miniconda
- CUDA-enabled PyTorch
- Module system (for CUDA, conda)

**Purpose:**
- All GPU experiments
- Model loading and inference
- Database generation

### Storage Strategy

**On HPC:**
```
/home/<username>/
├── aegis-runtime/              # Git repo (source code)
│   ├── aegis_runtime/          # Python package
│   ├── requirements.txt        # Package dependencies
│   └── README.md
│
└── aegis-data/                 # Data directory (NOT in git)
    ├── experiments/            # SQLite databases (permanent)
    │   ├── exp_001_tinyllama.db
    │   └── exp_002_gpt2.db
    └── logs/                   # Runtime logs (permanent)
        ├── exp_001/
        └── exp_002/
```

**Model Cache:**
- HuggingFace auto-caches to `~/.cache/huggingface/hub/`
- First download: slow (downloads from internet)
- Subsequent runs: fast (loads from cache)

**On Windows:**
```
C:\Users\<username>\DevProjects\
└── aegis-runtime/              # Git repo (managed by GitHub Desktop)
    ├── aegis_runtime/
    ├── requirements.txt
    └── downloaded_dbs/         # Databases downloaded from HPC (gitignored)
```

---

## Git Workflow

### Branch Structure

```
main
  ├─ Stable, reproducible, validated release snapshots only
  ├─ Only merged from dev
  └─ Tagged for experiment milestones (v0.1-baseline, v0.2-multitrial, etc.)

dev
  ├─ Integration branch
  └─ All validated feature branches merge here

feature/*
  ├─ Modular system components
  └─ Examples: feature/model-loader, feature/runtime-controller, feature/agent-logic

exp/*
  ├─ Experimental performance investigations
  └─ Examples: exp/batch-scaling, exp/precision-comparison, exp/model-comparison
```

### Commit Message Standard

```
[MODULE] Short description

What changed
Why
Measurable impact (if applicable)
```

**Example:**
```
[RUNTIME] Add peak memory reset per trial

Ensures trial isolation
Fixes memory accumulation bias across trials
Verified: std deviation reduced from 15% to 3%
```

### Tagging Policy

Every validated experiment milestone must be tagged:
- `v0.1-baseline` — Single-trial benchmark working
- `v0.2-multitrial` — Multi-trial statistics working
- `v0.3-agent` — Adaptive agent working
- `v0.4-hpc` — Full HPC deployment working

### Daily Workflow

**On Windows:**
1. Pull latest changes via GitHub Desktop
2. Edit code in VS Code
3. Commit via GitHub Desktop (write descriptive message)
4. Push via GitHub Desktop

**On HPC:**
1. SSH to cluster
2. `cd ~/aegis-runtime`
3. `git pull origin feature/baseline` (or relevant branch)
4. Run experiments
5. (Optional) Commit results from HPC or download to Windows first

**Never:**
- ❌ Commit directly to `main`
- ❌ Push without testing
- ❌ Commit database files or logs
- ❌ Force push to shared branches

---

## Directory Structure

```
aegis-runtime/                          # Repository root
├── aegis_runtime/                      # Python package (all source code)
│   │
│   ├── __init__.py
│   ├── main.py                         # Entry point
│   ├── config.py                       # Configuration system
│   ├── validate_environment.py         # GPU/environment validator
│   │
│   ├── model/                          # Model loading and inference
│   │   ├── __init__.py
│   │   ├── loader.py                   # Load model + tokenizer from HF
│   │   └── inference.py                # Single forward pass logic
│   │
│   ├── runtime/                        # Runtime control and monitoring
│   │   ├── __init__.py
│   │   ├── monitor.py                  # GPU state observation (pynvml)
│   │   ├── agent.py                    # Decision logic (adjust batch/precision)
│   │   ├── controller.py               # Orchestrate agent + inference + monitor
│   │   └── scheduler.py                # Trial sequencing and isolation
│   │
│   ├── metrics/                        # Logging and database
│   │   ├── __init__.py
│   │   ├── tracker.py                  # In-memory metric accumulation
│   │   ├── logger.py                   # Structured file logging
│   │   ├── database.py                 # SQLite CRUD operations
│   │   └── schema.sql                  # Database schema (single source of truth)
│   │
│   ├── experiments/                    # Experiment orchestration
│   │   ├── __init__.py
│   │   ├── benchmark.py                # Experiment parameter definitions
│   │   ├── runner.py                   # Execute trials with isolation
│   │   └── analyzer.py                 # Post-experiment statistics
│   │
│   ├── dashboard/                      # Visualization (Streamlit)
│   │   ├── __init__.py
│   │   ├── app.py                      # Streamlit UI
│   │   └── api.py                      # Data access layer
│   │
│   ├── slurm/                          # HPC job scripts
│   │   └── run_experiment.sh           # SLURM submission script
│   │
│   └── logs/                           # Runtime logs (gitignored)
│       └── .gitkeep
│
├── requirements.txt                    # Package dependencies (pinned versions)
├── environment-dev.yml                 # (DEPRECATED) Conda spec (optional reference)
├── .gitignore                          # Exclude logs, databases, pycache
├── README.md                           # Project documentation
└── LICENSE

```

### File Responsibility Summary

| File | Responsibility | Does NOT Own |
|------|---------------|--------------|
| `config.py` | All runtime parameters, validation, serialization | Database ops, metric collection |
| `model/loader.py` | Load model + tokenizer from HF, move to GPU | Inference, config, monitoring |
| `model/inference.py` | Single forward pass, token generation | Agent logic, GPU monitoring |
| `runtime/monitor.py` | GPU state observation (memory, utilization) | Decisions, inference |
| `runtime/agent.py` | Decision logic (adjust batch/precision) | Execution, monitoring |
| `runtime/controller.py` | Orchestrate agent + inference + monitor per cycle | Trial sequencing |
| `runtime/scheduler.py` | Trial loop, isolation enforcement | Experiment definition, stats |
| `metrics/tracker.py` | In-memory metric accumulation during trial | Database writes, logging |
| `metrics/logger.py` | Structured file logging (JSON) | Database, metrics |
| `metrics/database.py` | SQLite CRUD operations | Computation, logging |
| `experiments/benchmark.py` | Experiment parameter definitions | Execution, statistics |
| `experiments/runner.py` | Execute trials, enforce isolation | Stats, visualization |
| `experiments/analyzer.py` | Post-experiment statistical computation | Execution, database schema |
| `dashboard/app.py` | Streamlit UI for visualization | Data retrieval |
| `dashboard/api.py` | Data access layer (queries database) | UI rendering |

---

## 4-Week Implementation Roadmap

---

## **WEEK 1: Baseline GPU Benchmark + Structured Logging**

**Goal:** Single-trial benchmark that measures inference performance and logs everything to SQLite.

**Status After Week 1:** ✅ Environment validated, ✅ Database schema designed, ✅ GPU working on HPC

---

### **Week 1, Stage 1: Environment Setup** ✅ COMPLETE

**Deliverables:**
- Conda environment with PyTorch, Transformers, Pydantic, nvidia-ml-py
- `validate_environment.py` script
- Verified on Windows and HPC

**Key Files:**
- `requirements.txt`
- `aegis_runtime/validate_environment.py`

**Verification:**
```bash
python aegis_runtime/validate_environment.py | python -m json.tool
echo $?  # Should be 0
```

---

### **Week 1, Stage 2: SQLite Schema Design** ✅ COMPLETE

**Deliverables:**
- Four-table hierarchy: experiments → trials → inference_cycles → experiment_summary
- Foreign key constraints
- Indexes for query performance
- `DatabaseManager` class for all CRUD operations

**Key Files:**
- `aegis_runtime/metrics/schema.sql`
- `aegis_runtime/metrics/database.py`

**Schema:**

```sql
-- experiments: High-level metadata
CREATE TABLE experiments (
    experiment_id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    gpu_name TEXT NOT NULL,
    cuda_version TEXT NOT NULL,
    git_commit_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- trials: Independent trial runs
CREATE TABLE trials (
    trial_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    trial_number INTEGER NOT NULL,
    config_hash TEXT NOT NULL,
    random_seed INTEGER NOT NULL,
    status TEXT DEFAULT 'running',
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

-- inference_cycles: Individual generation loops
CREATE TABLE inference_cycles (
    cycle_id INTEGER PRIMARY KEY AUTOINCREMENT,
    trial_id TEXT NOT NULL,
    batch_size INTEGER NOT NULL,
    precision TEXT NOT NULL,
    tokens_per_second REAL NOT NULL,
    latency_ms REAL NOT NULL,
    gpu_memory_allocated_mb REAL NOT NULL,
    agent_action TEXT,
    agent_reason TEXT,
    oom_event BOOLEAN DEFAULT 0,
    FOREIGN KEY (trial_id) REFERENCES trials(trial_id)
);

-- experiment_summary: Statistics across trials
CREATE TABLE experiment_summary (
    experiment_id TEXT PRIMARY KEY,
    mean_latency_ms REAL NOT NULL,
    std_latency_ms REAL NOT NULL,
    mean_tokens_per_sec REAL NOT NULL,
    std_tokens_per_sec REAL NOT NULL,
    peak_memory_allocated_mb REAL NOT NULL,
    total_oom_events INTEGER DEFAULT 0,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);
```

---

### **Week 1, Stage 3: Configuration System** ✅ COMPLETE

**Deliverables:**
- `RuntimeConfig` class with Pydantic validation
- Serialization methods (to_dict, to_json, from_json)
- Config hash for trial isolation verification
- Path auto-detection (HPC vs local)

**Key Files:**
- `aegis_runtime/config.py`

**Usage:**
```python
from aegis_runtime.config import RuntimeConfig

# Create config
config = RuntimeConfig(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    batch_size=4,
    precision="fp16",
    num_trials=5
)

# Serialize
config_dict = config.to_dict()
config_json = config.to_json()

# Hash (for trial isolation)
config_hash = config.get_hash()
```

---

### **Week 1, Stage 4: Model Loader** 🔄 IN PROGRESS

**Deliverables:**
- Load any HuggingFace causal LM from config
- Load tokenizer
- Support fp16/bf16/fp32 precision
- Move model to GPU
- Return model + tokenizer ready for inference

**Key Files:**
- `aegis_runtime/model/loader.py`

**Implementation Guide:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from aegis_runtime.config import RuntimeConfig

def load_model_and_tokenizer(config: RuntimeConfig):
    """
    Load model and tokenizer from HuggingFace.
    
    Steps:
    1. Validate GPU available
    2. Load tokenizer with cache_dir
    3. Load model with:
       - torch_dtype based on config.precision
       - device_map="auto" (auto GPU placement)
       - cache_dir from config
    4. Set model.eval()
    5. Log model info (params, device)
    6. Return (model, tokenizer)
    """
    # YOUR IMPLEMENTATION HERE
    pass
```

**Test:**
```python
config = RuntimeConfig(model_name="gpt2", precision="fp16")
model, tokenizer = load_model_and_tokenizer(config)
print(f"Model on: {model.device}")
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
```

**Commit Message:**
```
[MODEL] Implement model and tokenizer loader

- Loads any HuggingFace causal LM from config.model_name
- Supports fp16/bf16/fp32 precision
- Auto-places model on GPU via device_map
- Returns model in eval mode + tokenizer

Tested with GPT-2 on A30 GPU (24GB)
```

---

### **Week 1, Stage 5: Inference Engine**

**Deliverables:**
- Single forward pass function
- Generate N tokens with configured batch size
- Return: generated tokens, tokens/sec, latency
- Memory tracking before/after generation

**Key Files:**
- `aegis_runtime/model/inference.py`

**Implementation Guide:**

```python
import torch
import time
from typing import Tuple, Dict

def run_inference(model, tokenizer, config: RuntimeConfig) -> Dict[str, float]:
    """
    Run one inference cycle.
    
    Args:
        model: Loaded model on GPU
        tokenizer: Loaded tokenizer
        config: RuntimeConfig with batch_size, max_seq_length
        
    Returns:
        Dict with:
            - tokens_generated: int
            - tokens_per_second: float
            - latency_ms: float
            - memory_allocated_mb: float
            - memory_reserved_mb: float
    
    Steps:
    1. Create dummy input_ids (batch_size x seq_len)
    2. Move to GPU
    3. Record start time and memory
    4. Generate tokens (model.generate)
    5. Record end time and memory
    6. Calculate metrics
    7. Return dict
    """
    # YOUR IMPLEMENTATION HERE
    pass
```

**Test:**
```python
model, tokenizer = load_model_and_tokenizer(config)
result = run_inference(model, tokenizer, config)
print(f"Tokens/sec: {result['tokens_per_second']:.2f}")
print(f"Latency: {result['latency_ms']:.2f} ms")
```

---

### **Week 1, Stage 6: GPU Monitor**

**Deliverables:**
- Poll GPU state using pynvml
- Return: allocated memory, reserved memory, utilization %
- Snapshot method (capture state at a moment)
- No side effects — observation only

**Key Files:**
- `aegis_runtime/runtime/monitor.py`

**Implementation Guide:**

```python
import pynvml
from typing import Dict

class GPUMonitor:
    """Monitor GPU state via nvidia-ml-py."""
    
    def __init__(self):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    
    def get_snapshot(self) -> Dict[str, float]:
        """
        Get current GPU state.
        
        Returns:
            Dict with:
                - memory_allocated_mb: float
                - memory_reserved_mb: float
                - utilization_percent: float
        """
        # YOUR IMPLEMENTATION HERE
        pass
    
    def __del__(self):
        pynvml.nvmlShutdown()
```

---

### **Week 1, Stage 7: Metrics Tracker**

**Deliverables:**
- In-memory accumulation of cycle metrics during trial
- Store: batch size, tokens/sec, latency, GPU state per cycle
- Flush to database at end of trial
- Reset between trials

**Key Files:**
- `aegis_runtime/metrics/tracker.py`

**Implementation Guide:**

```python
from typing import List, Dict

class MetricsTracker:
    """Accumulate metrics during a trial."""
    
    def __init__(self):
        self.cycles = []
    
    def record_cycle(self, cycle_data: Dict):
        """Record one inference cycle."""
        self.cycles.append(cycle_data)
    
    def flush_to_database(self, db_manager, trial_id: str):
        """Write all cycles to database."""
        for cycle in self.cycles:
            db_manager.insert_inference_cycle(trial_id, cycle)
    
    def reset(self):
        """Clear all recorded cycles."""
        self.cycles = []
```

---

### **Week 1, Stage 8: File Logger**

**Deliverables:**
- Structured logging to `aegis_runtime/logs/`
- One log file per trial
- JSON-formatted log entries
- Include: timestamp, level, module, message, context

**Key Files:**
- `aegis_runtime/metrics/logger.py`

**Implementation Guide:**

```python
import logging
import json
from pathlib import Path

def setup_trial_logger(trial_id: str, log_dir: str) -> logging.Logger:
    """
    Create a JSON logger for a trial.
    
    Args:
        trial_id: Unique trial identifier
        log_dir: Directory for log files
        
    Returns:
        Configured logger instance
    """
    log_path = Path(log_dir) / f"{trial_id}.log"
    
    # YOUR IMPLEMENTATION HERE
    # Use logging.FileHandler
    # Use custom JSON formatter
    pass
```

---

### **Week 1, Stage 9: Single-Trial Benchmark**

**Deliverables:**
- Define experiment parameters
- Create experiment in database
- Run ONE trial
- Log all cycles to database
- Verify data integrity

**Key Files:**
- `aegis_runtime/experiments/benchmark.py`

**Implementation Guide:**

```python
from aegis_runtime.config import RuntimeConfig
from aegis_runtime.model.loader import load_model_and_tokenizer
from aegis_runtime.model.inference import run_inference
from aegis_runtime.runtime.monitor import GPUMonitor
from aegis_runtime.metrics.tracker import MetricsTracker
from aegis_runtime.metrics.database import DatabaseManager
import uuid

def run_single_trial_benchmark(config: RuntimeConfig):
    """
    Run a complete single-trial benchmark.
    
    Steps:
    1. Initialize database
    2. Create experiment record
    3. Load model
    4. Create trial record
    5. Initialize monitor and tracker
    6. Run N inference cycles
    7. Record each cycle
    8. Flush to database
    9. Print summary
    """
    # YOUR IMPLEMENTATION HERE
    pass

if __name__ == "__main__":
    config = RuntimeConfig(
        model_name="gpt2",
        batch_size=4,
        precision="fp16",
        num_trials=1
    )
    run_single_trial_benchmark(config)
```

**Test on HPC:**
```bash
srun --partition=academic --gres=gpu:A30:1 --time=01:00:00 --mem=32G --pty bash
conda activate aegis
cd ~/aegis-runtime
python aegis_runtime/experiments/benchmark.py
```

**Verify:**
```bash
sqlite3 ~/aegis-data/experiments/aegis_metrics.db
> SELECT * FROM experiments;
> SELECT * FROM trials;
> SELECT COUNT(*) FROM inference_cycles;
> .quit
```

---

### **Week 1 Deliverable**

**Tag:** `v0.1-baseline`

**What Works:**
- Single-trial benchmark runs on HPC
- Model loads (GPT-2, TinyLlama, etc.)
- 50-100 inference cycles logged to database
- Database contains: 1 experiment, 1 trial, N cycles
- All metrics captured: tokens/sec, latency, GPU memory

**Verification:**
```bash
cd ~/aegis-runtime
git tag v0.1-baseline
git push origin v0.1-baseline
```

---

## **WEEK 2: Multi-Trial Statistical Framework**

**Goal:** Run 5-10 isolated trials per experiment, compute statistics, validate reproducibility.

---

### **Week 2, Stage 1: Trial Scheduler**

**Deliverables:**
- Execute N trials sequentially
- Enforce isolation protocol before each trial
- Catch OOM events gracefully
- Update trial status in database

**Key Files:**
- `aegis_runtime/runtime/scheduler.py`

**Isolation Protocol:**
```python
def reset_trial_environment():
    """
    Reset environment before each trial.
    
    Required steps:
    1. torch.cuda.empty_cache()
    2. torch.cuda.reset_peak_memory_stats()
    3. Reset RNG seed: torch.manual_seed(seed)
    4. Clear metrics tracker
    """
    import torch
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # Set seed based on trial number
```

**Implementation:**
```python
class TrialScheduler:
    def __init__(self, config: RuntimeConfig, db_manager, logger):
        self.config = config
        self.db = db_manager
        self.logger = logger
    
    def run_all_trials(self, experiment_id: str):
        """
        Run config.num_trials trials with full isolation.
        
        For each trial:
        1. Create trial record in database
        2. Reset environment
        3. Run trial (call run_single_trial)
        4. Handle OOM gracefully
        5. Update trial status
        6. Log completion
        """
        for trial_num in range(self.config.num_trials):
            # YOUR IMPLEMENTATION
            pass
```

---

### **Week 2, Stage 2: Experiment Runner**

**Deliverables:**
- High-level orchestrator
- Create experiment in database
- Call scheduler to run all trials
- Handle failures gracefully

**Key Files:**
- `aegis_runtime/experiments/runner.py`

**Implementation:**
```python
def run_experiment(config: RuntimeConfig):
    """
    Run complete multi-trial experiment.
    
    Steps:
    1. Initialize database
    2. Create experiment record (with git commit hash)
    3. Create trial scheduler
    4. Run all trials
    5. Call analyzer to compute statistics
    6. Write summary to database
    7. Return experiment_id
    """
    # YOUR IMPLEMENTATION
    pass
```

---

### **Week 2, Stage 3: Statistical Analyzer**

**Deliverables:**
- Query all trials for an experiment
- Compute mean ± std for latency and tokens/sec
- Compute peak memory across trials
- Calculate OOM frequency
- Write to experiment_summary table

**Key Files:**
- `aegis_runtime/experiments/analyzer.py`

**Implementation:**
```python
import numpy as np

class ExperimentAnalyzer:
    def __init__(self, db_manager):
        self.db = db_manager
    
    def analyze_experiment(self, experiment_id: str):
        """
        Compute statistics across all trials.
        
        Steps:
        1. Query all successful trials
        2. For each trial, get all cycles
        3. Compute per-trial averages
        4. Compute across-trial statistics:
           - Mean latency ± std
           - Mean tokens/sec ± std
           - Peak memory (max across all trials)
           - OOM count
        5. Write to experiment_summary table
        """
        trials = self.db.get_trials(experiment_id)
        
        latencies = []
        throughputs = []
        peak_memories = []
        
        for trial in trials:
            cycles = self.db.get_inference_cycles(trial['trial_id'])
            # Compute trial-level stats
            # YOUR IMPLEMENTATION
        
        # Compute experiment-level stats
        summary = {
            'experiment_id': experiment_id,
            'mean_latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies),
            # ... etc
        }
        
        self.db.insert_experiment_summary(summary)
```

---

### **Week 2, Stage 4: Multi-Trial Verification**

**Test:**
```python
config = RuntimeConfig(
    model_name="gpt2",
    batch_size=4,
    precision="fp16",
    num_trials=5  # Run 5 trials
)

experiment_id = run_experiment(config)

# Query summary
summary = db.get_experiment_summary(experiment_id)
print(f"Mean tokens/sec: {summary['mean_tokens_per_sec']:.2f} ± {summary['std_tokens_per_sec']:.2f}")
print(f"Coefficient of variation: {summary['std_tokens_per_sec'] / summary['mean_tokens_per_sec'] * 100:.1f}%")
```

**Validation:**
- CV (coefficient of variation) should be < 20% for valid results
- If CV > 20%, environment is unstable (investigate)

---

### **Week 2, Stage 5: Experiment Matrix**

**Deliverable:** Run experiments varying one parameter

**Test Cases:**

```python
# Batch size scaling
for batch_size in [1, 2, 4, 8]:
    config = RuntimeConfig(model_name="gpt2", batch_size=batch_size, num_trials=5)
    run_experiment(config)

# Precision comparison
for precision in ["fp16", "bf16"]:
    config = RuntimeConfig(model_name="gpt2", precision=precision, num_trials=5)
    run_experiment(config)

# Sequence length scaling
for seq_len in [128, 256, 512]:
    config = RuntimeConfig(model_name="gpt2", max_seq_length=seq_len, num_trials=5)
    run_experiment(config)
```

**Analysis:**
```sql
SELECT 
    model_name,
    batch_size,
    mean_tokens_per_sec,
    std_tokens_per_sec
FROM experiment_summary
JOIN experiments USING (experiment_id)
ORDER BY batch_size;
```

---

### **Week 2 Deliverable**

**Tag:** `v0.2-multitrial`

**What Works:**
- 5 trials run automatically with isolation
- Statistics computed across trials
- Database contains experiment_summary table
- Coefficient of variation validates reproducibility
- Batch size scaling experiment completed

---

## **WEEK 3: Adaptive Agent Integration**

**Goal:** Agent dynamically adjusts batch size and precision based on GPU state.

---

### **Week 3, Stage 1: Agent Decision Logic**

**Deliverables:**
- Observe GPU memory, utilization, tokens/sec
- Decide: increase batch, decrease batch, switch precision, no change
- Return action + reason (logged to database)

**Key Files:**
- `aegis_runtime/runtime/agent.py`

**Implementation:**

```python
class AdaptiveAgent:
    """
    Makes decisions to optimize throughput while avoiding OOM.
    
    Policy:
    - If memory < 50% and utilization > 80% → increase batch
    - If memory > 85% → decrease batch
    - If OOM detected → decrease batch by 50% or switch to fp16
    - If utilization < 60% → increase batch
    """
    
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.current_batch_size = config.batch_size
        self.current_precision = config.precision
    
    def decide(self, gpu_state: Dict, metrics: Dict) -> Dict[str, str]:
        """
        Make decision based on current state.
        
        Args:
            gpu_state: {memory_percent, utilization_percent}
            metrics: {tokens_per_sec, latency_ms}
            
        Returns:
            {
                'action': 'increase_batch' | 'decrease_batch' | 'switch_precision' | 'no_change',
                'reason': 'memory < 50%, utilization high',
                'new_batch_size': 8,
                'new_precision': 'fp16'
            }
        """
        # YOUR IMPLEMENTATION
        # Decision tree based on memory and utilization
        pass
```

---

### **Week 3, Stage 2: Runtime Controller**

**Deliverables:**
- Orchestrate inference + monitor + agent per cycle
- Log each cycle to database with agent decision
- Handle OOM gracefully

**Key Files:**
- `aegis_runtime/runtime/controller.py`

**Implementation:**

```python
class RuntimeController:
    """Coordinates inference, monitoring, and agent decisions."""
    
    def __init__(self, model, tokenizer, config, agent, monitor, tracker, logger):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.agent = agent
        self.monitor = monitor
        self.tracker = tracker
        self.logger = logger
    
    def run_trial(self, trial_id: str, num_cycles: int = 100):
        """
        Run one trial with agent control.
        
        For each cycle:
        1. Get GPU state (before inference)
        2. Run inference
        3. Get GPU state (after inference)
        4. Ask agent for decision
        5. Apply decision (update batch size/precision)
        6. Record cycle with agent action
        7. Handle OOM
        """
        for cycle_num in range(num_cycles):
            try:
                # Pre-inference state
                gpu_before = self.monitor.get_snapshot()
                
                # Run inference
                result = run_inference(self.model, self.tokenizer, self.config)
                
                # Post-inference state
                gpu_after = self.monitor.get_snapshot()
                
                # Agent decision
                decision = self.agent.decide(gpu_after, result)
                
                # Record cycle
                cycle_data = {
                    'trial_id': trial_id,
                    'cycle_number': cycle_num,
                    'batch_size': self.config.batch_size,
                    'precision': self.config.precision,
                    'tokens_per_second': result['tokens_per_second'],
                    'latency_ms': result['latency_ms'],
                    'gpu_memory_allocated_mb': gpu_after['memory_allocated_mb'],
                    'agent_action': decision['action'],
                    'agent_reason': decision['reason'],
                    'oom_event': False
                }
                self.tracker.record_cycle(cycle_data)
                
                # Apply decision
                if decision['action'] == 'increase_batch':
                    self.config.batch_size = decision['new_batch_size']
                elif decision['action'] == 'decrease_batch':
                    self.config.batch_size = decision['new_batch_size']
                
            except torch.cuda.OutOfMemoryError:
                # Handle OOM
                self.logger.error(f"OOM at cycle {cycle_num}, batch={self.config.batch_size}")
                # Record OOM cycle
                # Ask agent to recover
                pass
```

---

### **Week 3, Stage 3: Agent Policy Testing**

**Test Different Policies:**

```python
# Policy 1: Aggressive (maximize batch)
class AggressiveAgent(AdaptiveAgent):
    def decide(self, gpu_state, metrics):
        if gpu_state['memory_percent'] < 70:
            return {'action': 'increase_batch', ...}
        # ...

# Policy 2: Conservative (keep memory < 60%)
class ConservativeAgent(AdaptiveAgent):
    def decide(self, gpu_state, metrics):
        if gpu_state['memory_percent'] > 60:
            return {'action': 'decrease_batch', ...}
        # ...

# Policy 3: Adaptive (balance memory and utilization)
class BalancedAgent(AdaptiveAgent):
    def decide(self, gpu_state, metrics):
        # Complex decision tree
        pass
```

**Run Experiments:**
```python
for agent_class in [AggressiveAgent, ConservativeAgent, BalancedAgent]:
    agent = agent_class(config)
    experiment_id = run_experiment_with_agent(config, agent)
    # Compare results
```

---

### **Week 3, Stage 4: Agent Logging Validation**

**Verify:**
```sql
SELECT 
    cycle_number,
    batch_size,
    agent_action,
    agent_reason,
    tokens_per_second
FROM inference_cycles
WHERE trial_id = 'trial_abc'
ORDER BY cycle_number;
```

**Expected:**
- Every cycle has agent_action and agent_reason
- Batch size changes visible over time
- Reasoning is descriptive

---

### **Week 3 Deliverable**

**Tag:** `v0.3-agent`

**What Works:**
- Agent successfully adjusts batch size without OOM
- All decisions logged to database with reasons
- 3 different agent policies tested and compared
- Dashboard visualizes agent decisions over time

---

## **WEEK 4: HPC Deployment + Model Comparison + Dashboard**

**Goal:** Full production deployment on HPC, stress testing, model comparison.

---

### **Week 4, Stage 1: SLURM Job Script**

**Deliverables:**
- Job script for HPC execution
- Conda environment activation
- Config file driven execution
- Email notification on completion

**Key Files:**
- `aegis_runtime/slurm/run_experiment.sh`

**Implementation:**

```bash
#!/bin/bash
#SBATCH --job-name=aegis_benchmark
#SBATCH --partition=academic
#SBATCH --gres=gpu:A30:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --output=/home/%u/aegis-data/logs/slurm-%j.out
#SBATCH --error=/home/%u/aegis-data/logs/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@university.edu

# Load conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate aegis

# Navigate to repo
cd ~/aegis-runtime

# Run experiment with config file
CONFIG_FILE=$1
python -m aegis_runtime.main --config "$CONFIG_FILE"

# Print completion
echo "Experiment completed at $(date)"
echo "Database: $(grep db_path $CONFIG_FILE)"
```

**Usage:**
```bash
sbatch slurm/run_experiment.sh configs/experiment_tinyllama.json
```

---

### **Week 4, Stage 2: Config-Driven Execution**

**Deliverable:** Experiments defined as JSON configs

**Example:** `configs/exp_tinyllama_baseline.json`

```json
{
  "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "batch_size": 4,
  "precision": "fp16",
  "max_seq_length": 256,
  "num_trials": 5,
  "random_seed_base": 42,
  "db_path": "/home/tfnu/aegis-data/experiments/exp_tinyllama_baseline.db",
  "log_dir": "/home/tfnu/aegis-data/logs/exp_tinyllama"
}
```

**Main Entry Point:** `aegis_runtime/main.py`

```python
import argparse
from aegis_runtime.config import RuntimeConfig
from aegis_runtime.experiments.runner import run_experiment

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config JSON')
    args = parser.parse_args()
    
    config = RuntimeConfig.from_json(args.config)
    experiment_id = run_experiment(config)
    print(f"Experiment complete: {experiment_id}")

if __name__ == "__main__":
    main()
```

---

### **Week 4, Stage 3: Stress Test**

**Deliverable:** Long-running experiment to test stability

```json
{
  "model_name": "meta-llama/Llama-2-7b-hf",
  "batch_size": 4,
  "precision": "bf16",
  "max_seq_length": 512,
  "num_trials": 10,
  "num_cycles_per_trial": 200
}
```

**Submit:**
```bash
sbatch slurm/run_experiment.sh configs/stress_test_llama7b.json
```

**Monitor:**
```bash
squeue -u $USER
tail -f ~/aegis-data/logs/slurm-JOBID.out
```

**Verify:**
- No crashes
- No memory leaks (memory stable across trials)
- CV < 20%

---

### **Week 4, Stage 4: Model Comparison**

**Deliverable:** Benchmark 3 models with identical config

```python
models = [
    "gpt2",                                    # 124M params
    "EleutherAI/gpt-neo-1.3B",                # 1.3B params
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"      # 1.1B params
]

base_config = {
    "batch_size": 4,
    "precision": "fp16",
    "max_seq_length": 256,
    "num_trials": 5
}

for model in models:
    config = RuntimeConfig(model_name=model, **base_config)
    # Generate config JSON
    # Submit SLURM job
```

**Analysis:**
```sql
SELECT 
    e.model_name,
    COUNT(DISTINCT t.trial_id) as num_trials,
    s.mean_tokens_per_sec,
    s.std_tokens_per_sec,
    s.peak_memory_allocated_mb,
    s.total_oom_events
FROM experiments e
JOIN experiment_summary s USING (experiment_id)
JOIN trials t USING (experiment_id)
WHERE e.model_name IN ('gpt2', 'EleutherAI/gpt-neo-1.3B', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')
GROUP BY e.model_name
ORDER BY s.mean_tokens_per_sec DESC;
```

---

### **Week 4, Stage 5: Dashboard Development**

**Deliverable:** Streamlit app for visualization

**Key Files:**
- `aegis_runtime/dashboard/app.py`
- `aegis_runtime/dashboard/api.py`

**Features:**

**Page 1: Experiment List**
- Show all experiments
- Filter by model, date, status
- Click to drill down

**Page 2: Experiment Overview**
- Summary statistics
- Trial success rate
- OOM events

**Page 3: Trial Comparison**
- Box plot: tokens/sec across trials
- Line plot: latency over cycles
- Error bars showing std deviation

**Page 4: Agent Timeline**
- Line plot: batch size over cycles
- Annotations: agent decisions
- Color-coded by action type

**Page 5: GPU Memory Heatmap**
- 2D heatmap: trial x cycle
- Color = memory usage %
- Highlight OOM events

**Run Dashboard:**
```bash
# On Windows (after downloading databases)
conda activate aegis
streamlit run aegis_runtime/dashboard/app.py
```

---

### **Week 4, Stage 6: Documentation**

**Deliverables:**

**README.md:**
- Project overview
- Installation instructions
- Quick start guide
- HPC deployment guide
- Example configs

**HPC_GUIDE.md:**
- SSH setup
- Conda environment creation
- SLURM job submission
- Retrieving results
- Common issues

**ARCHITECTURE.md:**
- System design
- Module responsibilities
- Database schema
- Extension points

---

### **Week 4 Deliverable**

**Tag:** `v0.4-hpc`

**What Works:**
- SLURM jobs submitted and complete successfully
- 3 models compared under identical conditions
- Dashboard visualizes all experiments
- Documentation complete
- Reproducible: any experiment can be re-run from git tag + config

---

## Module Responsibilities

### config.py

**Owns:**
- All runtime parameters (model, batch, precision, paths)
- Validation via Pydantic
- Serialization (to_dict, to_json, from_json)
- Config hash generation (MD5 for trial isolation)

**Does NOT Own:**
- Database operations
- Metric collection
- Agent logic

**Interface:**
```python
config = RuntimeConfig(model_name="gpt2", batch_size=4)
config_dict = config.to_dict()
config_hash = config.get_hash()
```

---

### model/loader.py

**Owns:**
- Loading model from HuggingFace
- Loading tokenizer
- Moving model to GPU
- Setting precision (fp16/bf16/fp32)

**Does NOT Own:**
- Inference/generation
- GPU monitoring
- Configuration

**Interface:**
```python
model, tokenizer = load_model_and_tokenizer(config)
```

---

### model/inference.py

**Owns:**
- Single forward pass
- Token generation
- Timing measurement
- Memory tracking (before/after)

**Does NOT Own:**
- Model loading
- Agent decisions
- GPU monitoring

**Interface:**
```python
result = run_inference(model, tokenizer, config)
# Returns: {tokens_per_second, latency_ms, memory_allocated_mb, ...}
```

---

### runtime/monitor.py

**Owns:**
- GPU state observation via pynvml
- Memory queries (allocated, reserved)
- Utilization queries

**Does NOT Own:**
- Decisions
- Inference
- Logging

**Interface:**
```python
monitor = GPUMonitor()
state = monitor.get_snapshot()
# Returns: {memory_allocated_mb, memory_reserved_mb, utilization_percent}
```

---

### runtime/agent.py

**Owns:**
- Decision logic (increase/decrease batch, switch precision)
- Policy implementation
- Reasoning generation

**Does NOT Own:**
- Execution of decisions
- GPU monitoring
- Inference

**Interface:**
```python
agent = AdaptiveAgent(config)
decision = agent.decide(gpu_state, metrics)
# Returns: {action, reason, new_batch_size, new_precision}
```

---

### runtime/controller.py

**Owns:**
- Orchestration of inference + monitor + agent
- Cycle loop
- OOM handling
- Decision application

**Does NOT Own:**
- Trial sequencing
- Statistical analysis
- Database schema

**Interface:**
```python
controller = RuntimeController(model, tokenizer, config, agent, monitor, tracker, logger)
controller.run_trial(trial_id, num_cycles=100)
```

---

### runtime/scheduler.py

**Owns:**
- Trial loop (run N trials)
- Isolation enforcement (cache clear, seed reset)
- Trial status updates

**Does NOT Own:**
- Experiment definition
- Statistical computation
- Visualization

**Interface:**
```python
scheduler = TrialScheduler(config, db, logger)
scheduler.run_all_trials(experiment_id)
```

---

### metrics/tracker.py

**Owns:**
- In-memory metric accumulation
- Cycle data storage (list)
- Flush to database

**Does NOT Own:**
- Database writes (delegates to database.py)
- File logging
- Statistical computation

**Interface:**
```python
tracker = MetricsTracker()
tracker.record_cycle(cycle_data)
tracker.flush_to_database(db, trial_id)
tracker.reset()
```

---

### metrics/logger.py

**Owns:**
- Structured file logging (JSON)
- Log file creation per trial
- Timestamping

**Does NOT Own:**
- Database operations
- Metric computation
- Visualization

**Interface:**
```python
logger = setup_trial_logger(trial_id, log_dir)
logger.info("Inference cycle completed", extra={'cycle': 1, 'tokens_per_sec': 245})
```

---

### metrics/database.py

**Owns:**
- All SQLite CRUD operations
- Schema initialization
- Transaction handling
- Query methods

**Does NOT Own:**
- Statistical computation (analyzer.py)
- Metric accumulation (tracker.py)
- Schema definition (schema.sql)

**Interface:**
```python
db = DatabaseManager("path/to/db.sqlite")
db.initialize_schema()
exp_id = db.insert_experiment(exp_data)
db.insert_trial(trial_data)
db.insert_inference_cycle(cycle_data)
```

---

### experiments/benchmark.py

**Owns:**
- Experiment parameter definitions
- Experiment creation logic
- High-level orchestration

**Does NOT Own:**
- Trial execution (scheduler.py)
- Statistics (analyzer.py)
- Visualization

**Interface:**
```python
experiment_id = run_single_trial_benchmark(config)
```

---

### experiments/runner.py

**Owns:**
- Full experiment execution
- Calls scheduler for trials
- Calls analyzer for stats
- Error handling

**Does NOT Own:**
- Trial isolation (scheduler.py)
- Statistical formulas (analyzer.py)
- Database schema

**Interface:**
```python
experiment_id = run_experiment(config)
```

---

### experiments/analyzer.py

**Owns:**
- Post-experiment statistics
- Mean/std computation
- Peak memory aggregation
- Writing experiment_summary

**Does NOT Own:**
- Trial execution
- Database schema
- Visualization

**Interface:**
```python
analyzer = ExperimentAnalyzer(db)
analyzer.analyze_experiment(experiment_id)
```

---

## Database Schema

See Week 1, Stage 2 for complete schema.

**Key Points:**
- Four tables: experiments, trials, inference_cycles, experiment_summary
- Foreign keys enforce referential integrity
- Indexes on experiment_id, trial_id, timestamp
- All writes append-only (no updates except trial status)

---

## Coding Standards

### Module Docstrings

Every file must have:

```python
"""
AEGIS Runtime — filename.py

Responsibility: [One sentence]

Owns:
- [Bullet list of what this module is responsible for]

Does NOT own:
- [What explicitly belongs elsewhere]
"""
```

### Function Docstrings

```python
def function_name(arg1: Type1, arg2: Type2) -> ReturnType:
    """
    Brief description.
    
    Args:
        arg1: Description
        arg2: Description
        
    Returns:
        Description of return value
        
    Raises:
        ExceptionType: When this happens
    """
```

### Type Hints

All functions must have type hints:

```python
from typing import Dict, List, Tuple, Optional

def load_model(config: RuntimeConfig) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    ...
```

### Error Handling

```python
try:
    result = operation()
except SpecificException as e:
    logger.error(f"Operation failed: {e}")
    raise RuntimeError(f"Failed to complete operation: {e}")
```

### Logging

```python
# Good
logger.info("Inference complete", extra={'tokens_per_sec': 245, 'latency_ms': 12})

# Bad
print(f"Inference complete, tokens/sec: {tps}")
```

---

## Testing Protocol

### Unit Tests

(Future addition, not Week 1-4)

```python
def test_config_validation():
    with pytest.raises(ValueError):
        RuntimeConfig(batch_size=-1)
```

### Integration Tests

Manual for now:

```bash
# Test model loading
python -c "from aegis_runtime.model.loader import load_model_and_tokenizer; from aegis_runtime.config import RuntimeConfig; config = RuntimeConfig(model_name='gpt2'); model, tok = load_model_and_tokenizer(config); print('OK')"

# Test inference
python test_inference.py

# Test full benchmark
python aegis_runtime/experiments/benchmark.py
```

### Validation Queries

After each experiment:

```sql
-- Check experiment created
SELECT * FROM experiments ORDER BY created_at DESC LIMIT 1;

-- Check trials
SELECT trial_id, trial_number, status FROM trials WHERE experiment_id = 'exp_abc';

-- Check cycle count
SELECT COUNT(*) FROM inference_cycles WHERE trial_id = 'trial_123';

-- Check summary
SELECT * FROM experiment_summary WHERE experiment_id = 'exp_abc';
```

---

## Deployment Checklist

### Before Running on HPC

- [ ] Code committed and pushed from Windows
- [ ] Git pulled on HPC
- [ ] Conda environment activated
- [ ] Config JSON created with correct paths
- [ ] Data directories exist (`~/aegis-data/experiments`, `~/aegis-data/logs`)
- [ ] GPU partition access confirmed

### Running Experiment

- [ ] Config file validated (paths exist, model name correct)
- [ ] SLURM script tested with short job first
- [ ] Job submitted with correct partition and GPU type
- [ ] Job ID recorded
- [ ] Monitoring logs (`tail -f slurm-JOBID.out`)

### After Experiment Completes

- [ ] Check SLURM output for errors
- [ ] Verify database file created
- [ ] Query database for expected number of trials
- [ ] Coefficient of variation < 20%
- [ ] Download database to Windows (optional)
- [ ] Commit results or summary to repo
- [ ] Tag git repo if milestone reached

---

## Common Issues and Solutions

### Issue: OOM During Inference

**Symptom:** `torch.cuda.OutOfMemoryError`

**Solutions:**
1. Decrease batch size
2. Switch to fp16 (if using bf16/fp32)
3. Decrease max_seq_length
4. Use smaller model

**Prevention:**
- Start with batch_size=1 for large models
- Monitor memory usage
- Agent should catch and recover

---

### Issue: High Coefficient of Variation

**Symptom:** CV > 20%

**Causes:**
1. Shared GPU node (other users)
2. Inconsistent thermal throttling
3. Background processes

**Solutions:**
1. Request exclusive GPU (`--exclusive`)
2. Run during off-peak hours
3. Increase num_trials to 10
4. Check `nvidia-smi` for other processes

---

### Issue: Model Download Fails

**Symptom:** `HTTPError` or `ConnectionError`

**Solutions:**
1. Check internet access from HPC
2. Pre-download model: `huggingface-cli download model_name`
3. Use `cache_dir` to specify location
4. Check HuggingFace Hub status

---

### Issue: SLURM Job Pending

**Symptom:** Job stuck in queue

**Solutions:**
1. Check partition access: `sacctmgr show user $USER`
2. Try different GPU type
3. Reduce time request
4. Reduce memory request
5. Check cluster status: `sinfo`

---

## Quick Reference Commands

### HPC Workflow

```bash
# SSH
ssh username@hpc.university.edu

# Navigate and pull
cd ~/aegis-runtime
git pull origin feature/baseline

# Activate environment
conda activate aegis

# Submit job
sbatch slurm/run_experiment.sh configs/my_experiment.json

# Monitor
squeue -u $USER
tail -f ~/aegis-data/logs/slurm-JOBID.out

# Query database
sqlite3 ~/aegis-data/experiments/my_experiment.db
> SELECT * FROM experiment_summary;
```

### Windows Workflow

```powershell
# Open repo
cd C:\Users\tanve\DevProjects\aegis-runtime

# Pull latest
# (Use GitHub Desktop)

# Edit code
code .

# Commit and push
# (Use GitHub Desktop)
```

---

## Next Steps After Week 4

### Immediate Extensions

1. **Precision Switching During Runtime**
   - Agent switches fp16 ↔ bf16 mid-trial
   - Requires model reload

2. **Sequence Length Adaptation**
   - Agent adjusts max_seq_length
   - Trade-off: longer sequences = smaller batches

3. **Multi-GPU Support**
   - DataParallel or tensor parallel
   - Monitor memory across all GPUs

### Advanced Features

1. **Real-Time Dashboard**
   - WebSocket updates
   - Live progress bars
   - Alert on OOM

2. **Experiment Comparison Tool**
   - CLI: `compare_experiments.py exp1 exp2`
   - Statistical significance testing
   - Automated report generation

3. **Anomaly Detection**
   - Flag trials with outlier performance
   - Automatic re-run of failed trials
   - OOM prediction based on memory trends

---

## Conclusion

This specification provides a complete roadmap for building AEGIS Runtime from scratch to a production-ready system in 4 weeks. Follow the stages sequentially, commit regularly, and validate at each milestone.

**Key Success Metrics:**

- ✅ Week 1: Single trial logs to database
- ✅ Week 2: 5 trials with CV < 20%
- ✅ Week 3: Agent prevents OOM
- ✅ Week 4: HPC production runs complete

**Remember:**
- Observability first (log everything)
- Isolation is critical (reset between trials)
- Reproducibility is non-negotiable (git tags + configs)
- Model agnostic always (no hardcoding)

Good luck building AEGIS Runtime!

---

**End of Specification**