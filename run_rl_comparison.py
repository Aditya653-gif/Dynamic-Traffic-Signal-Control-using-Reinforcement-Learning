"""
Entry Point: Run RL Comparison Study
=====================================
Trains Q-Learning, SARSA, DQN, and Double DQN on the traffic signal
control environment under identical conditions (500 episodes each),
generates professional comparison graphs and performance tables.

Usage:
    python run_rl_comparison.py
"""

import os
import sys

# Ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from rl_comparison import run_comparison, Config


def main():
    cfg = Config()

    # Override if desired (defaults are already set)
    # cfg.NUM_EPISODES = 500
    # cfg.EPISODE_LENGTH = 100

    collector = run_comparison(cfg)

    print("\nDone. Check results/rl_comparison/ for all outputs:")
    print("  Graphs:")
    print("    - unified_comparison.png          (all metrics, normalised)")
    print("    - final_efficiency_bar_graph.png   (waiting-time reduction %)")
    print("  Data:")
    print("    - all_algorithms_metrics.csv")
    print("    - final_comparison_table.csv")
    print("    - comparison_summary.json")
    print("    - Per-algorithm episode CSVs")
    print("    - comparison_summary.json")
    print("    - Per-algorithm CSVs")


if __name__ == '__main__':
    main()
