# AEGIS Runtime

AEGIS Runtime is a modular, GPU-aware adaptive inference system for large language models (LLMs).

It dynamically adjusts inference parameters such as batch size and numerical precision to maximize throughput (tokens/sec) while preventing out-of-memory (OOM) errors.

This project is designed as a systems + ML engineering experiment focused on observability, performance control, and adaptive runtime behavior.

---

## 🚀 Objective

Build a GPU-aware inference runtime that:

- Maximizes tokens per second
- Prevents CUDA OOM crashes
- Adapts dynamically to workload conditions
- Provides real-time logging and observability
- Demonstrates modular runtime system design

---

## 🧠 Core Idea

Transformer-based LLM inference scales non-linearly with:

- Batch size
- Sequence length
- KV cache growth
- Precision (fp16 vs bf16)

Instead of using static inference configurations, AEGIS:

1. Monitors GPU memory usage
2. Tracks throughput (tokens/sec)
3. Detects OOM risks
4. Dynamically adjusts:
   - Batch size
   - Precision
   - Maximum sequence length
5. Logs every decision transparently

---

## 🏗 Architecture Overview

```
```
Incoming Request
      │
      ▼
  Scheduler          ← decides when and in what order requests are processed
      │
      ▼
  Controller         ← applies current agent settings to the request
      │
      ▼
   Monitor           ← reads GPU state before and after inference
      │
      ▼
Agent Decision       ← evaluates monitor output, updates control variables
      │
      ▼
Model Inference      ← TinyLlama forward pass with current batch/precision/seqlen
      │
      ▼
Metrics Tracker      ← computes tokens/sec, latency, OOM flag
      │
      ▼
Logger + SQLite      ← persists every cycle's full record
      │
      ▼
Live Dashboard       ← reads SQLite, visualizes in real-time
```
All components are modular and observable.

---


## 📁 Project Structure


```
aegis-runtime/
│
├── main.py                # Entry point
├── config.py              # Global configuration
│
├── model/
│   ├── __init__.py
│   ├── loader.py          # Load TinyLlama / Mistral
│   └── inference.py       # Forward pass logic
│
├── runtime/
│   ├── __init__.py
│   ├── monitor.py         # GPU + system metrics
│   ├── scheduler.py       # Batch & microbatch logic
│   ├── agent.py           # Decision-making logic
│   └── controller.py      # Orchestrates runtime flow
│
├── metrics/
│   ├── __init__.py
│   ├── tracker.py         # Collect latency, throughput
│   ├── logger.py          # Structured logging
│   └── database.py        # PostgreSQL integration
│
├── dashboard/
│   ├── __init__.py
│   ├── api.py             # FastAPI backend (metrics endpoint)
│   └── app.py             # Streamlit dashboard
│
├── experiments/
│   ├── __init__.py
│   └── benchmark.py       # Controlled test scenarios
│
└── logs/
    └── runtime.log
```
---

## ⚙️ Model

Default model: TinyLlama/TinyLlama-1.1B-Chat-v1.0


- 1.1B parameters
- Transformer-based
- Supports fp16 / bf16
- Suitable for GPU memory experimentation

---

## 📊 Observability & Logging

Every inference cycle logs:

- Timestamp
- Batch size
- Sequence length
- Precision
- Tokens generated
- Tokens/sec
- Latency
- GPU memory allocated
- GPU memory peak
- GPU utilization %
- Agent decision reason
- OOM events

Logs are stored in:

- Console (real-time)
- SQLite database
- Live dashboard visualizations

---

## 🎯 Adaptive Agent Rules (Initial Version)

- If GPU memory < 60% → Increase batch size
- If GPU memory > 85% → Reduce batch size
- If OOM occurs → Halve batch size and retry
- If memory pressure high → Switch to fp16
- If long sequence degrades throughput → Cap max sequence length

All decisions are logged and observable.

---
## 🧪 Benchmark Goals

Evaluate:

- Tokens/sec vs batch size
- Memory usage vs sequence length
- Adaptive runtime vs static configuration
- OOM prevention effectiveness

---

## 🖥 Environment

- WSL2 + Ubuntu 20.04
- Python 3.10+
- CUDA-enabled PyTorch
- Conda environment
- SQLite for metrics storage
- Streamlit or FastAPI for dashboard

---

## 📈 Why This Project Matters

AEGIS demonstrates:

- Understanding of GPU memory behavior
- Transformer inference scaling
- Runtime system design
- Observability-driven engineering
- Adaptive decision logic in ML systems
- ## 🔮 Future Extensions

- Replace SQLite with PostgreSQL for distributed deployment
- Multi-GPU scheduling
- Reinforcement learning-based agent
- Integration with vLLM or TensorRT
- Cloud deployment (AWS/GCP)
- Distributed request queue

---

## 📌 Status

Work in progress.

Currently focused on building the adaptive GPU-aware inference control layer and observability system.

---
## 👤 Author

Tanveer  
MS Computer Science  
Focus: Computer Science + ML Engineering
