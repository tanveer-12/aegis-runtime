-- AEGIS Runtime — schema.sql
-- Single source of truth for all SQLite table definitions.
-- No table is created anywhere else in the codebase.
-- Version: 2

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,     -- UUID or timestamp-based unique ID
    model_name TEXT NOT NULL,           -- e.g., "gpt2", "meta-llama/Llama-2-7b-hf"
    gpu_name TEXT NOT NULL,             -- From validate_environment.py
    gpu_memory_total_gb REAL NOT NULL,  -- Total GPU memory
    hostname TEXT NOT NULL,             -- Where experiment ran (WSL vs HPC node)
    cuda_version TEXT NOT NULL,         -- e.g., "12.1"
    driver_version TEXT NOT NULL,       -- e.g., "577.02"
    pytorch_version TEXT NOT NULL,      -- e.g., "2.5.1"
    transformers_version TEXT NOT NULL, -- e.g., "4.46.3"
    git_commit_hash TEXT NOT NULL,      -- Reproducibility: exact code version
    config_description TEXT,            -- Human-readable experiment goal
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trials (
    trial_id TEXT PRIMARY KEY,          -- UUID
    experiment_id TEXT NOT NULL,        -- Foreign key to experiments table
    trail_number INTEGER NOT NULL,      -- 1, 2, 3, ... (for ordering)
    config_hash TEXT NOT NULL,          -- MD5/SHA of config dict (detects drift)
    random_seed INTEGER NOT NULL,       -- RNG seed for this trial
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,             -- NULL until trial finishes
    status TEXT DEFAULT 'running',      -- 'running', 'completed', 'failed', 'oom'
    failure_reason TEXT,                -- Error message if status='failed'
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

CREATE TABLE IF NOT EXISTS inference_cycles (
    cycle_id INTEGER PRIMARY KEY AUTOINCREMENT,
    trial_id TEXT NOT NULL,             -- Foreign key to trials table
    cycle_number INTEGER NOT NULL,      -- Sequential: 1, 2, 3, ...
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Configuration for this cycle
    batch_size INTEGER NOT NULL,
    max_seq_length INTEGER NOT NULL,
    precision TEXT NOT NULL,            -- 'fp16', 'bf16', 'fp32'

    -- Performance metrics
    tokens_generated INTEGER NOT NULL,
    tokens_per_second REAL NOT NULL,
    latency_ms REAL NOT NULL,

    -- GPU State
    gpu_memory_allocated_mb REAL NOT NULL,
    gpu_memory_reserved_mb REAL NOT NULL,
    gpu_utilization_percent REAL,       -- May be NULL if polling fails

    -- Agent decision
    agent_action TEXT,                  -- 'increase_batch', 'decrease_batch', 'switch_precision', 'no_change'
    agent_reason TEXT,                  -- Why the agent made this decision

    -- Error Tracking
    oom_event BOOLEAN DEFAULT 0,        -- 1 if OOM occurred this cycle
    FOREIGN KEY (trial_id) REFERENCES trials(trial_id)
);


CREATE TABLE IF NOT EXISTS experiment_summary (
    experiment_id TEXT PRIMARY KEY,     -- Foreign key to experiments
    num_trials INTEGER NOT NULL,
    
    -- Latency statistics (milliseconds)
    mean_latency_ms REAL NOT NULL,
    std_latency_ms REAL NOT NULL,
    min_latency_ms REAL NOT NULL,
    max_latency_ms REAL NOT NULL,
    
    -- Throughput statistics (tokens/sec)
    mean_tokens_per_sec REAL NOT NULL,
    std_tokens_per_sec REAL NOT NULL,
    min_tokens_per_sec REAL NOT NULL,
    max_tokens_per_sec REAL NOT NULL,
    
    -- Memory statistics (MB)
    peak_memory_allocated_mb REAL NOT NULL,
    peak_memory_reserved_mb REAL NOT NULL,
    
    -- Error statistics
    total_oom_events INTEGER DEFAULT 0,
    oom_frequency REAL,                 -- OOMs per trial (avg)
    
    -- Trial status breakdown
    successful_trials INTEGER NOT NULL,
    failed_trials INTEGER NOT NULL,
    
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);


-- Performance indexes for common queries
CREATE INDEX IF NOT EXISTS idx_trials_experiment ON trials(experiment_id);
CREATE INDEX IF NOT EXISTS idx_cycles_trial ON inference_cycles(trial_id);
CREATE INDEX IF NOT EXISTS idx_cycles_timestamp ON inference_cycles(timestamp);
CREATE INDEX IF NOT EXISTS idx_trials_status ON trials(status);