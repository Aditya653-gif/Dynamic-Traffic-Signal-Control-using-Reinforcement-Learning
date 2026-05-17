# Traffic Signal Control using Reinforcement Learning

> Comparing **Q-Learning · SARSA · DQN · Double DQN** for adaptive traffic signal optimisation  
> Live dashboard + offline comparison study across 7 Tamil Nadu intersections

---

## Overview

This project applies **Reinforcement Learning** to the urban traffic signal control problem. Four algorithms are trained and evaluated under identical simulation conditions (same dataset, state space, action space, reward function, and episode count) to produce a fair, reproducible comparison.

A **Flask-based real-time dashboard** separately demonstrates Double DQN controlling live traffic across seven Tamil Nadu city intersections.

### Algorithms Implemented

| Algorithm | Type | Key Property |
|-----------|------|-------------|
| **Q-Learning** | Tabular, off-policy | Uses max Q(s', a') for updates |
| **SARSA** | Tabular, on-policy | Updates with Q(s', a') where a' is the next action actually taken |
| **DQN** | Neural network, off-policy | Experience replay + target network; susceptible to overestimation bias |
| **Double DQN** | Neural network, off-policy | Decouples action selection from evaluation to reduce overestimation |

---

## Architecture

### Double DQN (Deep Q-Network)

The AI controller uses a **Double DQN** architecture to avoid the overestimation bias of standard DQN:

```
State (9-dim)                  Action (2)
┌──────────────────┐           ┌──────────┐
│ Queue N (norm)   │           │ 0: Hold  │
│ Queue S (norm)   │  ┌─────┐ │ 1: Switch│
│ Queue E (norm)   │─→│ 128 │ └──────────┘
│ Queue W (norm)   │  │ReLU │
│ Phase One-Hot(4) │  ├─────┤
│ Timer (norm)     │  │ 128 │─→ Q(s,a)
└──────────────────┘  │ReLU │
                      ├─────┤
                      │  2  │
                      └─────┘
```

**Double DQN Update Rule:**
- **Policy network** selects the best action: `a* = argmax Q_policy(s', a)`
- **Target network** evaluates that action: `Q_target(s', a*)`
- Prevents overestimation of Q-values

---

## Project Structure

```
traffic_dqn/
├── app.py                          # Flask server — live dashboard + Double DQN
├── run_rl_comparison.py            # Entry point: train all 4 algorithms & generate graphs
├── START_DASHBOARD.bat             # Windows quick-launcher for the dashboard
│
├── src/                            # Core modules
│   ├── traffic_environment.py      # Environment classes (continuous + traditional controllers)
│   ├── dataset_generator.py        # Synthetic traffic dataset generator
│   ├── q_learning.py               # Q-Learning agent + discrete environment
│   ├── sarsa.py                    # SARSA agent
│   ├── dqn_training.py             # Double DQN agent, network, trainer
│   ├── standard_dqn.py             # Standard DQN agent (for comparison)
│   ├── rl_comparison.py            # Orchestrator — trains all 4 under identical conditions
│   ├── metrics_collector.py        # Per-episode metric collection + convergence detection
│   ├── comparison_visualizer.py    # Unified graph generation (normalised comparison)
│   ├── final_bar_graph.py          # Final efficiency bar graph (waiting-time reduction %)
│   └── traffic_api.py              # Simulation API layer for the dashboard
│
├── data/
│   ├── traffic_dataset.csv         # Synthetic traffic data (7 cities × 30 days)
│   └── simulation_log.csv          # Live dashboard simulation log
│
├── models/
│   └── comparison/                 # Saved model weights (DQN, Double DQN)
│
├── results/
│   └── rl_comparison/              # All comparison outputs
│       ├── unified_comparison.png              # Single normalised comparison figure
│       ├── final_efficiency_bar_graph.png      # Waiting-time reduction bar chart
│       ├── final_comparison_table.csv          # Summary table (CSV)
│       ├── comparison_summary.json             # Summary table (JSON)
│       ├── all_algorithms_metrics.csv          # Combined per-episode data
│       └── <algorithm>_episode_metrics.csv     # Per-algorithm episode data
│
├── checkpoints/                    # Live dashboard model checkpoints
├── templates/
│   └── index.html                  # Dashboard UI
├── requirements.txt
└── README.md
```

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Episodes | 500 |
| Steps per episode | 100 |
| Epsilon schedule | 1.0 → 0.01 (decay 0.99065) |
| Discount factor (tabular) | 0.95 |
| Learning rate (tabular) | 0.1 |
| Discount factor (DQN / Double DQN) | 0.99 |
| Learning rate (DQN / Double DQN) | 0.001 |
| Network architecture | 128 → 128 (ReLU) |
| Replay buffer size | 10,000 |
| Batch size | 32 |
| Target network update | Every 10 episodes |
| Random seed | 42 |

All four algorithms share the same environment, state space (9-dim), action space (Hold / Switch), and reward function.

### Reward Function

```
R = −(α · wait_time  +  β · remaining_queue  +  γ · switched)
    α = 0.6,  β = 0.3,  γ = 0.1
```

---

## Evaluation Methodology

1. Each algorithm is trained independently for 500 episodes under identical conditions.
2. Per-episode metrics are recorded: **total reward, average waiting time, average queue length, throughput**, and **training loss** (DQN / Double DQN only).
3. **Convergence episode** is detected programmatically using sliding-window variance on the reward signal.
4. Final performance is averaged over the last 50 episodes.

### Efficiency Metric — Average Waiting Time Reduction (%)

The final bar graph computes reduction relative to the **worst-performing model**:

```
Reduction(%) = ((Worst_Avg_Wait − Model_Avg_Wait) / Worst_Avg_Wait) × 100
```

- No values are hardcoded — all derived from real training outputs.
- The worst model automatically receives 0%.
- The formula ensures fair, mathematically correct comparison.

---

## How to Run

### Prerequisites

- Python 3.9+
- pip

### Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Generate Dataset (if needed)

```bash
python -c "import sys; sys.path.insert(0,'src'); from dataset_generator import TrafficDatasetGenerator; TrafficDatasetGenerator().generate_dataset('data/traffic_dataset.csv')"
```

### Run RL Comparison Study

```bash
python run_rl_comparison.py
```

This will:
1. Train Q-Learning, SARSA, DQN, and Double DQN (500 episodes each)
2. Save per-episode metrics to `results/rl_comparison/`
3. Generate the unified comparison graph and final efficiency bar chart
4. Print the performance summary table

### Run Live Dashboard

```bash
python app.py
# Then open http://127.0.0.1:5000
```

Or double-click `START_DASHBOARD.bat` on Windows.

---

## Output Graphs

| File | Description |
|------|-------------|
| `unified_comparison.png` | 7-row figure: normalised reward, wait, queue, throughput, loss curves + bar charts + summary table |
| `final_efficiency_bar_graph.png` | Average Waiting Time Reduction (%) with summary table |

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, Flask, PyTorch |
| Frontend | HTML5, CSS3, JavaScript |
| Map | Leaflet.js + CartoDB Dark Matter tiles |
| Charts | Chart.js |
| RL Framework | Custom (PyTorch) — Q-Learning, SARSA, DQN, Double DQN |
| Streaming | Server-Sent Events (SSE) |

---

## License

Academic project — Tamil Nadu traffic signal optimisation using reinforcement learning.
