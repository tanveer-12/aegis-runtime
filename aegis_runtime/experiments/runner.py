"""
AEGIS Runtime — runner.py

Responsibility: Executes the full trial sequence for a given experiment.

Owns:
- Trial loop iteration
- Enforcing trial isolation protocol (memory reset, seed reset, buffer clear)
- Passing trial results to metrics/tracker.py

Does NOT own:
- Experiment parameter definition (benchmark.py)
- Statistical analysis (analyzer.py)
- Any inference logic (model/inference.py)
"""