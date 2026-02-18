"""
AEGIS Runtime — analyzer.py

Responsibility: Post-experiment statistical computation over trial results.

Owns:
- Mean and std deviation of latency and tokens/sec across trials
- OOM frequency calculation
- Peak memory aggregation
- Writing summary rows to experiment_summary table

Does NOT own:
- Running trials (runner.py)
- Writing raw cycle logs (metrics/database.py)
- Any visualization (dashboard/)
"""