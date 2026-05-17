"""
Final Efficiency Bar Graph Generator
=====================================
Reads real training results from results/rl_comparison/ and generates:
  1. A professional bar chart showing Average Waiting Time Reduction (%)
  2. A summary table printed to console and rendered inside the graph

No values are hardcoded — everything is derived from computed metrics.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Paths ───────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'rl_comparison')
OUTPUT_DIR  = os.path.join(PROJECT_DIR, 'results', 'rl_comparison')

# ── Load real metrics ───────────────────────────────────────────────────

def load_metrics() -> pd.DataFrame:
    """Load final comparison table produced by the training pipeline."""
    csv_path = os.path.join(RESULTS_DIR, 'final_comparison_table.csv')
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found. Run training first (run_rl_comparison.py).")
        sys.exit(1)
    return pd.read_csv(csv_path)


def compute_reduction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Waiting Time Reduction (%) for every algorithm.

    Formula:
      Reduction(%) = ((Worst_Wait − Model_Wait) / Worst_Wait) × 100

    The worst-performing model (highest avg waiting time) gets 0%.
    """
    worst_wait = df['final_avg_waiting_time'].max()
    df = df.copy()
    df['waiting_time_reduction_pct'] = (
        (worst_wait - df['final_avg_waiting_time']) / worst_wait * 100
    ).round(2)
    return df


# ── Plot ────────────────────────────────────────────────────────────────

ALGO_COLORS = {
    'Q-Learning':  '#E74C3C',
    'SARSA':       '#F39C12',
    'DQN':         '#3498DB',
    'Double DQN':  '#2ECC71',
}

ALGO_ORDER = ['Q-Learning', 'SARSA', 'DQN', 'Double DQN']


def generate_final_bar_graph(df: pd.DataFrame):
    """Generate publication-ready bar graph + summary table in one figure."""

    # Ensure consistent ordering
    df['_sort'] = df['algorithm'].map({a: i for i, a in enumerate(ALGO_ORDER)})
    df = df.sort_values('_sort').reset_index(drop=True)

    algos    = df['algorithm'].tolist()
    waits    = df['final_avg_waiting_time'].values
    reductions = df['waiting_time_reduction_pct'].values
    colors   = [ALGO_COLORS.get(a, '#888888') for a in algos]
    n        = len(algos)
    x        = np.arange(n)
    width    = 0.52

    # ── Figure layout: bar chart (top) + table (bottom) ────────────────
    fig = plt.figure(figsize=(14, 9), facecolor='#FAFAFA')
    gs  = GridSpec(2, 1, figure=fig, height_ratios=[3, 1.1], hspace=0.30)

    # ── Bar Chart ──────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0])
    bars = ax.bar(x, reductions, width, color=colors,
                  edgecolor='white', linewidth=1.0, zorder=3)

    # Value labels above bars
    for bar, val in zip(bars, reductions):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.35,
                f'{val:.2f}%', ha='center', va='bottom',
                fontsize=12, fontweight='bold', color='#222222')

    ax.set_xticks(x)
    ax.set_xticklabels(algos, fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Waiting Time Reduction (%)', fontsize=13, fontweight='bold')
    ax.set_title(
        'Average Waiting Time Reduction (%) – Reinforcement Learning Model Comparison',
        fontsize=15, fontweight='bold', pad=14
    )
    ax.set_ylim(0, max(reductions) * 1.18)
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Summary Table ──────────────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[1])
    ax_tbl.axis('off')

    col_labels = ['Model Name', 'Final Avg Waiting Time (s)',
                  'Waiting Time Reduction (%)']
    cell_data = []
    for _, row in df.iterrows():
        cell_data.append([
            row['algorithm'],
            f"{row['final_avg_waiting_time']:.2f}",
            f"{row['waiting_time_reduction_pct']:.2f}%"
        ])

    tbl = ax_tbl.table(cellText=cell_data, colLabels=col_labels,
                        loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.auto_set_column_width(col=list(range(len(col_labels))))
    tbl.scale(1.0, 1.8)

    # Style header
    for j in range(len(col_labels)):
        cell = tbl[0, j]
        cell.set_facecolor('#2C3E50')
        cell.set_text_props(color='white', fontweight='bold')

    # Style rows
    for i, (_, row) in enumerate(df.iterrows()):
        bg = ALGO_COLORS.get(row['algorithm'], '#888888')
        for j in range(len(col_labels)):
            cell = tbl[i + 1, j]
            cell.set_facecolor(bg + '18')
            cell.set_edgecolor('#CCCCCC')

    # ── Save ───────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, 'final_efficiency_bar_graph.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def print_summary_table(df: pd.DataFrame):
    """Print text summary to console."""
    worst = df['final_avg_waiting_time'].max()
    worst_algo = df.loc[df['final_avg_waiting_time'].idxmax(), 'algorithm']

    print("\n" + "=" * 78)
    print("  AVERAGE WAITING TIME REDUCTION — RL MODEL COMPARISON")
    print("=" * 78)
    print(f"\n  Baseline (worst model): {worst_algo}  ({worst:.2f}s avg wait)")
    print(f"  Formula: Reduction% = ((Worst_Wait - Model_Wait) / Worst_Wait) × 100\n")
    print(f"  {'Model':<16} {'Avg Wait (s)':>14} {'Reduction (%)':>16}")
    print("  " + "-" * 50)
    for _, row in df.iterrows():
        print(f"  {row['algorithm']:<16} {row['final_avg_waiting_time']:>14.2f} "
              f"{row['waiting_time_reduction_pct']:>15.2f}%")
    print("  " + "-" * 50)
    print()


# ── Entry point ─────────────────────────────────────────────────────────

def main():
    df = load_metrics()
    df = compute_reduction(df)
    print_summary_table(df)
    generate_final_bar_graph(df)


if __name__ == '__main__':
    main()
