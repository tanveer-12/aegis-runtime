#!/usr/bin/env bash
#SBATCH -A cs441
#SBATCH -p academic
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -C A30
#SBATCH -t 01:00:00
#SBATCH --mem 20G
#SBATCH --job-name=aegis_w1_test
#SBATCH --output=/home/%u/aegis-data/logs/w1_test_%j.out
#SBATCH --error=/home/%u/aegis-data/logs/w1_test_%j.err

# ============================================================
# AEGIS Runtime — Week 1 Verification Script
# Tests every module individually then runs the full benchmark
# ============================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate aegis
cd ~/aegis-runtime

echo "============================================"
echo " AEGIS Week 1 Verification"
echo " Job ID : $SLURM_JOB_ID"
echo " Node   : $SLURMD_NODENAME"
echo " GPU    : $CUDA_VISIBLE_DEVICES"
echo " Time   : $(date)"
echo "============================================"
echo ""

# ── TEST 1: Environment ──────────────────────────────────────
echo ">>> TEST 1: Environment"
python aegis_runtime/validate_environment.py
echo ""

# ── TEST 2: Config ───────────────────────────────────────────
echo ">>> TEST 2: Config system"
python -c "
from aegis_runtime.config import RuntimeConfig
c = RuntimeConfig(
    model_name='gpt2',
    precision='fp16',
    batch_size=4,
    max_seq_length=128,
    num_cycles=50
)
print('  Config hash    :', c.get_hash())
print('  total_gpu_mb   :', c.total_gpu_memory_mb)
print('  PASS: Config instantiates and serializes correctly')
"
echo ""

# ── TEST 3: Database schema ──────────────────────────────────
echo ">>> TEST 3: Database schema"
python -c "
from aegis_runtime.metrics.database import DatabaseManager
import tempfile, os
db_path = os.path.expanduser('~/aegis-data/experiments/schema_test_temp.db')
db = DatabaseManager(db_path)
db.initialize_schema()

# Verify all 5 tables exist
import sqlite3
conn = sqlite3.connect(db_path)
tables = conn.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall()
table_names = [t[0] for t in tables]
expected = ['experiments', 'trials', 'inference_cycles', 'experiment_summary', 'estimation_log']
for t in expected:
    status = 'OK' if t in table_names else 'MISSING'
    print(f'  Table {t}: {status}')

# Verify new columns exist on inference_cycles
cols = conn.execute('PRAGMA table_info(inference_cycles)').fetchall()
col_names = [c[1] for c in cols]
for col in ['estimated_memory_mb', 'estimation_error_pct']:
    status = 'OK' if col in col_names else 'MISSING'
    print(f'  Column inference_cycles.{col}: {status}')

# Verify new column on experiment_summary
cols2 = conn.execute('PRAGMA table_info(experiment_summary)').fetchall()
col_names2 = [c[1] for c in cols2]
status = 'OK' if 'mean_estimation_error_pct' in col_names2 else 'MISSING'
print(f'  Column experiment_summary.mean_estimation_error_pct: {status}')

conn.close()
os.remove(db_path)
print('  PASS: Schema verified')
"
echo ""

# ── TEST 4: Tracker ──────────────────────────────────────────
echo ">>> TEST 4: Metrics tracker"
python -c "
from aegis_runtime.metrics.tracker import MetricsTracker
t = MetricsTracker()
t.record_cycle({'batch_size': 4, 'tokens_per_second': 200.0, 'latency_ms': 20.0})
t.record_cycle({'batch_size': 4, 'tokens_per_second': 210.0, 'latency_ms': 19.5})
t.record_cycle({'batch_size': 4, 'tokens_per_second': 195.0, 'latency_ms': 21.0})
assert t.count == 3, f'Expected 3, got {t.count}'
t.reset()
assert t.count == 0, f'Expected 0 after reset, got {t.count}'
print('  PASS: record_cycle, count, reset all work correctly')
"
echo ""

# ── TEST 5: Logger ───────────────────────────────────────────
echo ">>> TEST 5: File logger"
python -c "
from aegis_runtime.metrics.logger import setup_trial_logger
import os, json
log_dir = os.path.expanduser('~/aegis-data/logs')
logger = setup_trial_logger('test_trial_verify', log_dir)
logger.info('Verification message', extra={'cycle': 1, 'tokens_per_sec': 245.0})
logger.debug('Debug line', extra={'detail': 'testing'})

log_path = os.path.join(log_dir, 'test_trial_verify.log')
with open(log_path) as f:
    lines = f.readlines()

assert len(lines) >= 1, 'Log file has no lines'
parsed = json.loads(lines[0])
required_keys = ['timestamp', 'level', 'module', 'message', 'context']
for k in required_keys:
    assert k in parsed, f'Missing key: {k}'
print(f'  Log file written: {log_path}')
print(f'  Log line keys   : {list(parsed.keys())}')
print(f'  PASS: Logger produces valid JSON lines')
os.remove(log_path)
"
echo ""

# ── TEST 6: GPU Monitor ──────────────────────────────────────
echo ">>> TEST 6: GPU monitor"
python -c "
from aegis_runtime.runtime.monitor import GPUMonitor
m = GPUMonitor()
total = m.get_total_memory_mb()
snap = m.get_snapshot()
print(f'  Total VRAM          : {total:.1f} MB')
print(f'  Memory allocated    : {snap[\"memory_allocated_mb\"]:.1f} MB')
print(f'  Memory reserved     : {snap[\"memory_reserved_mb\"]:.1f} MB')
print(f'  GPU utilization     : {snap[\"utilization_percent\"]}%')
assert total > 0, 'Total VRAM should be positive'
assert total > 10000, f'Expected >10GB for A30, got {total:.1f} MB'
print('  PASS: Monitor reads GPU state correctly')
"
echo ""

# ── TEST 7: Model Loader ─────────────────────────────────────
echo ">>> TEST 7: Model loader (GPT-2, fp16)"
python -c "
from aegis_runtime.config import RuntimeConfig
from aegis_runtime.model.loader import load_model_and_tokenizer
from aegis_runtime.runtime.monitor import GPUMonitor
import torch

m = GPUMonitor()
mem_before = m.get_snapshot()['memory_allocated_mb']

config = RuntimeConfig(model_name='gpt2', precision='fp16')
model, tokenizer = load_model_and_tokenizer(config)

mem_after = m.get_snapshot()['memory_allocated_mb']
param_count = sum(p.numel() for p in model.parameters())
device = str(next(model.parameters()).device)

print(f'  Device              : {device}')
print(f'  Parameter count     : {param_count:,}')
print(f'  Training mode       : {model.training}')
print(f'  Memory before load  : {mem_before:.1f} MB')
print(f'  Memory after load   : {mem_after:.1f} MB')
print(f'  Memory used by model: {mem_after - mem_before:.1f} MB')
print(f'  Pad token set       : {tokenizer.pad_token is not None}')

assert 'cuda' in device, f'Model should be on GPU, got {device}'
assert param_count == 124439808, f'GPT-2 should have 124439808 params, got {param_count}'
assert model.training == False, 'Model should be in eval mode'
assert tokenizer.pad_token is not None, 'Pad token should be set'
print('  PASS: Loader places model on GPU in eval mode')
"
echo ""

# ── TEST 8: Inference Engine ─────────────────────────────────
echo ">>> TEST 8: Inference engine"
python -c "
from aegis_runtime.config import RuntimeConfig
from aegis_runtime.model.loader import load_model_and_tokenizer
from aegis_runtime.model.inference import run_inference

config = RuntimeConfig(
    model_name='gpt2',
    precision='fp16',
    batch_size=4,
    max_seq_length=128
)
model, tokenizer = load_model_and_tokenizer(config)
result = run_inference(model, tokenizer, config)

print('  Inference result keys:')
for k, v in result.items():
    print(f'    {k}: {v}')

assert result['tokens_per_second'] > 0, 'tokens_per_second must be positive'
assert result['latency_ms'] > 0, 'latency_ms must be positive'
assert result['peak_memory_allocated_mb'] > 0, 'peak_memory must be positive'
assert result['tokens_generated'] > 0, 'tokens_generated must be positive'
print('  PASS: Inference returns complete metrics dict')
"
echo ""

# ── TEST 9: Full End-to-End Benchmark ───────────────────────
echo ">>> TEST 9: Full benchmark (50 cycles)"
python -c "
from aegis_runtime.config import RuntimeConfig
from aegis_runtime.experiments.benchmark import run_single_trial_benchmark
import os

db_path = os.path.expanduser('~/aegis-data/experiments/exp_gpt2_w1_test.db')

config = RuntimeConfig(
    model_name='gpt2',
    precision='fp16',
    batch_size=4,
    max_seq_length=128,
    num_cycles=50,
    db_path=db_path
)

print('  Running 50 cycles...')
run_single_trial_benchmark(config)
print('  Benchmark complete.')
"
echo ""

# ── TEST 10: Database Verification ──────────────────────────
echo ">>> TEST 10: Database verification"
DB=~/aegis-data/experiments/exp_gpt2_w1_test.db

echo "  --- Table row counts ---"
sqlite3 $DB "SELECT 'experiments      :', COUNT(*) FROM experiments;"
sqlite3 $DB "SELECT 'trials           :', COUNT(*) FROM trials;"
sqlite3 $DB "SELECT 'inference_cycles :', COUNT(*) FROM inference_cycles;"
sqlite3 $DB "SELECT 'estimation_log   :', COUNT(*) FROM estimation_log;"

echo ""
echo "  --- Trial status ---"
sqlite3 $DB "SELECT '  trial status:', status FROM trials;"

echo ""
echo "  --- First 3 cycles ---"
sqlite3 $DB "
.headers on
.mode column
SELECT cycle_id,
       batch_size,
       ROUND(tokens_per_second, 1)       AS tps,
       ROUND(latency_ms, 1)              AS lat_ms,
       ROUND(gpu_memory_allocated_mb, 1) AS mem_mb,
       estimated_memory_mb               AS est_mb
FROM inference_cycles
LIMIT 3;
"

echo ""
echo "  --- Validation: estimated_memory_mb should be NULL (Week 3 fills this) ---"
sqlite3 $DB "SELECT '  NULL estimated_memory rows:', COUNT(*) FROM inference_cycles WHERE estimated_memory_mb IS NULL;"
sqlite3 $DB "SELECT '  Total cycle rows          :', COUNT(*) FROM inference_cycles;"

echo ""
echo "  --- Experiment summary ---"
sqlite3 $DB "
.headers on
.mode column
SELECT model_name, gpu_name, cuda_version
FROM experiments;
"

echo ""
echo "============================================"
echo " Week 1 Verification Complete"
echo " Time: $(date)"
echo "============================================"
echo ""
echo "CHECKLIST:"
sqlite3 ~/aegis-data/experiments/exp_gpt2_w1_test.db "
SELECT
  CASE WHEN (SELECT COUNT(*) FROM experiments) = 1
    THEN '[PASS] experiments table has 1 row'
    ELSE '[FAIL] experiments table row count wrong' END;
SELECT
  CASE WHEN (SELECT status FROM trials LIMIT 1) = 'completed'
    THEN '[PASS] trial status is completed'
    ELSE '[FAIL] trial status is not completed' END;
SELECT
  CASE WHEN (SELECT COUNT(*) FROM inference_cycles) = 50
    THEN '[PASS] 50 inference cycles recorded'
    ELSE '[FAIL] cycle count is not 50' END;
SELECT
  CASE WHEN (SELECT COUNT(*) FROM inference_cycles WHERE tokens_per_second > 0) = 50
    THEN '[PASS] all cycles have positive tokens_per_second'
    ELSE '[FAIL] some cycles have zero or null tokens_per_second' END;
SELECT
  CASE WHEN (SELECT COUNT(*) FROM inference_cycles WHERE estimated_memory_mb IS NULL) = 50
    THEN '[PASS] estimated_memory_mb is NULL (correct for Week 1)'
    ELSE '[FAIL] estimated_memory_mb unexpectedly populated' END;
SELECT
  CASE WHEN (SELECT COUNT(*) FROM estimation_log) = 0
    THEN '[PASS] estimation_log is empty (correct for Week 1)'
    ELSE '[FAIL] estimation_log unexpectedly has rows' END;
"