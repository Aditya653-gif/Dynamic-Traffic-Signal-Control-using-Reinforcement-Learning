"""
Professional Comparison Visualizer for RL Algorithms
Generates publication-quality plots comparing Q-Learning, SARSA, DQN, and Double DQN
on traffic signal control performance metrics.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import os
from typing import Dict, List, Optional

# ── Professional style configuration ───────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#FAFAFA',
    'axes.facecolor': '#FFFFFF',
    'axes.edgecolor': '#CCCCCC',
    'axes.labelcolor': '#333333',
    'axes.titlecolor': '#222222',
    'xtick.color': '#555555',
    'ytick.color': '#555555',
    'grid.color': '#E0E0E0',
    'grid.alpha': 0.7,
    'grid.linestyle': '--',
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '#CCCCCC',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

# Color palette – distinct, colorblind-friendly
ALGO_COLORS = {
    'Q-Learning':  '#E74C3C',   # Red
    'SARSA':       '#F39C12',   # Orange
    'DQN':         '#3498DB',   # Blue
    'Double DQN':  '#2ECC71',   # Green
}

ALGO_MARKERS = {
    'Q-Learning':  'o',
    'SARSA':       's',
    'DQN':         '^',
    'Double DQN':  'D',
}

ALGO_LINESTYLES = {
    'Q-Learning':  '-',
    'SARSA':       '--',
    'DQN':         '-.',
    'Double DQN':  '-',
}


def _smooth(data: np.ndarray, window: int = 20) -> np.ndarray:
    """Apply moving average smoothing while preserving array length."""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    padded = np.pad(data, (window // 2, window - 1 - window // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(data)]


def _add_watermark(ax, text='RL Comparison Study'):
    """Add subtle watermark to plot."""
    ax.text(0.98, 0.02, text, transform=ax.transAxes,
            fontsize=7, color='#BBBBBB', ha='right', va='bottom',
            fontstyle='italic')


class ComparisonVisualizer:
    """Generates all comparison plots for the RL study."""

    def __init__(self, metrics_collector, output_dir: str = 'results/rl_comparison'):
        """
        Args:
            metrics_collector: ComparisonMetricsCollector instance
            output_dir: Directory to save all plots
        """
        self.collector = metrics_collector
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────
    #  1. Reward vs Episodes (all 4 algorithms)
    # ─────────────────────────────────────────────────────────────────────
    def plot_reward_comparison(self, smooth_window: int = 20):
        """Plot episode reward convergence for all algorithms."""
        fig, ax = plt.subplots(figsize=(14, 7))

        for name in self.collector.get_all_names():
            metrics = self.collector.get_algorithm(name)
            rewards = np.array(metrics.rewards)
            episodes = np.arange(1, len(rewards) + 1)
            smoothed = _smooth(rewards, smooth_window)

            color = ALGO_COLORS.get(name, '#888888')
            ax.plot(episodes, smoothed, label=name,
                    color=color, linewidth=2.2,
                    linestyle=ALGO_LINESTYLES.get(name, '-'))
            ax.fill_between(episodes, rewards, smoothed, alpha=0.08, color=color)

        ax.set_xlabel('Episode', fontweight='bold')
        ax.set_ylabel('Total Episode Reward', fontweight='bold')
        ax.set_title('Reward Convergence Comparison Across RL Algorithms',
                      fontweight='bold', pad=15)
        ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
        ax.grid(True)
        _add_watermark(ax)

        path = os.path.join(self.output_dir, 'reward_comparison.png')
        fig.savefig(path)
        plt.close(fig)
        print(f"    Saved: {path}")

    # ─────────────────────────────────────────────────────────────────────
    #  2. Average Waiting Time vs Episodes
    # ─────────────────────────────────────────────────────────────────────
    def plot_waiting_time_comparison(self, smooth_window: int = 20):
        """Plot average waiting time across training for all algorithms."""
        fig, ax = plt.subplots(figsize=(14, 7))

        for name in self.collector.get_all_names():
            metrics = self.collector.get_algorithm(name)
            waits = np.array(metrics.waiting_times)
            episodes = np.arange(1, len(waits) + 1)
            smoothed = _smooth(waits, smooth_window)

            color = ALGO_COLORS.get(name, '#888888')
            ax.plot(episodes, smoothed, label=name,
                    color=color, linewidth=2.2,
                    linestyle=ALGO_LINESTYLES.get(name, '-'))
            ax.fill_between(episodes, waits, smoothed, alpha=0.08, color=color)

        ax.set_xlabel('Episode', fontweight='bold')
        ax.set_ylabel('Average Waiting Time (seconds)', fontweight='bold')
        ax.set_title('Average Waiting Time Reduction During Training',
                      fontweight='bold', pad=15)
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax.grid(True)
        _add_watermark(ax)

        path = os.path.join(self.output_dir, 'waiting_time_comparison.png')
        fig.savefig(path)
        plt.close(fig)
        print(f"    Saved: {path}")

    # ─────────────────────────────────────────────────────────────────────
    #  3. Average Queue Length vs Episodes
    # ─────────────────────────────────────────────────────────────────────
    def plot_queue_length_comparison(self, smooth_window: int = 20):
        """Plot average queue length across training for all algorithms."""
        fig, ax = plt.subplots(figsize=(14, 7))

        for name in self.collector.get_all_names():
            metrics = self.collector.get_algorithm(name)
            queues = np.array(metrics.queue_lengths)
            episodes = np.arange(1, len(queues) + 1)
            smoothed = _smooth(queues, smooth_window)

            color = ALGO_COLORS.get(name, '#888888')
            ax.plot(episodes, smoothed, label=name,
                    color=color, linewidth=2.2,
                    linestyle=ALGO_LINESTYLES.get(name, '-'))
            ax.fill_between(episodes, queues, smoothed, alpha=0.08, color=color)

        ax.set_xlabel('Episode', fontweight='bold')
        ax.set_ylabel('Average Queue Length (vehicles)', fontweight='bold')
        ax.set_title('Queue Length Reduction During Training',
                      fontweight='bold', pad=15)
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax.grid(True)
        _add_watermark(ax)

        path = os.path.join(self.output_dir, 'queue_length_comparison.png')
        fig.savefig(path)
        plt.close(fig)
        print(f"    Saved: {path}")

    # ─────────────────────────────────────────────────────────────────────
    #  4. Training Loss vs Episodes (DQN & Double DQN only)
    # ─────────────────────────────────────────────────────────────────────
    def plot_loss_comparison(self, smooth_window: int = 20):
        """Plot training loss for DQN and Double DQN only."""
        fig, ax = plt.subplots(figsize=(14, 7))

        nn_algos = ['DQN', 'Double DQN']
        for name in nn_algos:
            if name not in self.collector.algorithms:
                continue
            metrics = self.collector.get_algorithm(name)
            losses = np.array([l if l is not None else 0.0 for l in metrics.losses])
            episodes = np.arange(1, len(losses) + 1)
            smoothed = _smooth(losses, smooth_window)

            color = ALGO_COLORS.get(name, '#888888')
            ax.plot(episodes, smoothed, label=name,
                    color=color, linewidth=2.2,
                    linestyle=ALGO_LINESTYLES.get(name, '-'))
            ax.fill_between(episodes, losses, smoothed, alpha=0.08, color=color)

        ax.set_xlabel('Episode', fontweight='bold')
        ax.set_ylabel('Training Loss (MSE)', fontweight='bold')
        ax.set_title('Training Loss Convergence: DQN vs Double DQN',
                      fontweight='bold', pad=15)
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax.grid(True)
        _add_watermark(ax)

        path = os.path.join(self.output_dir, 'training_loss_comparison.png')
        fig.savefig(path)
        plt.close(fig)
        print(f"    Saved: {path}")

    # ─────────────────────────────────────────────────────────────────────
    #  5. Final Performance Bar Chart
    # ─────────────────────────────────────────────────────────────────────
    def plot_final_bar_chart(self):
        """
        Bar chart comparing final performance metrics after training:
        - Final average reward
        - Final average waiting time
        - Final queue length
        - Convergence episode
        """
        table = self.collector.generate_comparison_table()
        algos = table['algorithm'].tolist()
        n = len(algos)
        x = np.arange(n)
        width = 0.55
        colors = [ALGO_COLORS.get(a, '#888888') for a in algos]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Final Performance Comparison After 500 Training Episodes',
                     fontsize=16, fontweight='bold', y=0.98)

        # ── 5a. Final Average Reward ──
        ax = axes[0, 0]
        vals = table['final_reward'].values
        bars = ax.bar(x, vals, width, color=colors, edgecolor='white', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(algos, fontweight='bold')
        ax.set_ylabel('Final Average Reward', fontweight='bold')
        ax.set_title('Final Average Reward', fontweight='bold')
        ax.grid(axis='y')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{v:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        _add_watermark(ax)

        # ── 5b. Final Average Waiting Time ──
        ax = axes[0, 1]
        vals = table['final_avg_waiting_time'].values
        bars = ax.bar(x, vals, width, color=colors, edgecolor='white', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(algos, fontweight='bold')
        ax.set_ylabel('Average Waiting Time (s)', fontweight='bold')
        ax.set_title('Final Average Waiting Time', fontweight='bold')
        ax.grid(axis='y')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{v:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        _add_watermark(ax)

        # ── 5c. Final Queue Length ──
        ax = axes[1, 0]
        vals = table['final_queue_length'].values
        bars = ax.bar(x, vals, width, color=colors, edgecolor='white', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(algos, fontweight='bold')
        ax.set_ylabel('Average Queue Length', fontweight='bold')
        ax.set_title('Final Average Queue Length', fontweight='bold')
        ax.grid(axis='y')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{v:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        _add_watermark(ax)

        # ── 5d. Convergence Episode ──
        ax = axes[1, 1]
        vals = table['convergence_episode'].values.astype(int)
        bars = ax.bar(x, vals, width, color=colors, edgecolor='white', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(algos, fontweight='bold')
        ax.set_ylabel('Convergence Episode', fontweight='bold')
        ax.set_title('Convergence Speed (Lower = Faster)', fontweight='bold')
        ax.grid(axis='y')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'Ep {v}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        _add_watermark(ax)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        path = os.path.join(self.output_dir, 'final_performance_bars.png')
        fig.savefig(path)
        plt.close(fig)
        print(f"    Saved: {path}")

    # ─────────────────────────────────────────────────────────────────────
    #  6. Throughput Comparison
    # ─────────────────────────────────────────────────────────────────────
    def plot_throughput_comparison(self, smooth_window: int = 20):
        """Plot throughput comparison across training."""
        fig, ax = plt.subplots(figsize=(14, 7))

        for name in self.collector.get_all_names():
            metrics = self.collector.get_algorithm(name)
            tp = np.array(metrics.throughputs)
            episodes = np.arange(1, len(tp) + 1)
            smoothed = _smooth(tp, smooth_window)

            color = ALGO_COLORS.get(name, '#888888')
            ax.plot(episodes, smoothed, label=name,
                    color=color, linewidth=2.2,
                    linestyle=ALGO_LINESTYLES.get(name, '-'))

        ax.set_xlabel('Episode', fontweight='bold')
        ax.set_ylabel('Episode Throughput (vehicles served)', fontweight='bold')
        ax.set_title('Throughput Improvement During Training',
                      fontweight='bold', pad=15)
        ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
        ax.grid(True)
        _add_watermark(ax)

        path = os.path.join(self.output_dir, 'throughput_comparison.png')
        fig.savefig(path)
        plt.close(fig)
        print(f"    Saved: {path}")

    # ─────────────────────────────────────────────────────────────────────
    #  7. Combined Dashboard (all key metrics in one figure)
    # ─────────────────────────────────────────────────────────────────────
    def plot_combined_dashboard(self, smooth_window: int = 20):
        """Generate a single combined figure with all key metric plots."""
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)
        fig.suptitle('Reinforcement Learning Comparison Study\n'
                     'Traffic Signal Control — 500 Episodes',
                     fontsize=18, fontweight='bold', y=0.99)

        algo_names = self.collector.get_all_names()

        # --- Panel 1: Reward ---
        ax1 = fig.add_subplot(gs[0, 0])
        for name in algo_names:
            m = self.collector.get_algorithm(name)
            r = _smooth(np.array(m.rewards), smooth_window)
            eps = np.arange(1, len(r) + 1)
            ax1.plot(eps, r, label=name, color=ALGO_COLORS.get(name),
                     linewidth=2, linestyle=ALGO_LINESTYLES.get(name, '-'))
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Reward Convergence')
        ax1.legend(fontsize=8)
        ax1.grid(True)

        # --- Panel 2: Waiting Time ---
        ax2 = fig.add_subplot(gs[0, 1])
        for name in algo_names:
            m = self.collector.get_algorithm(name)
            w = _smooth(np.array(m.waiting_times), smooth_window)
            eps = np.arange(1, len(w) + 1)
            ax2.plot(eps, w, label=name, color=ALGO_COLORS.get(name),
                     linewidth=2, linestyle=ALGO_LINESTYLES.get(name, '-'))
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Avg Wait Time (s)')
        ax2.set_title('Average Waiting Time')
        ax2.legend(fontsize=8)
        ax2.grid(True)

        # --- Panel 3: Queue Length ---
        ax3 = fig.add_subplot(gs[1, 0])
        for name in algo_names:
            m = self.collector.get_algorithm(name)
            q = _smooth(np.array(m.queue_lengths), smooth_window)
            eps = np.arange(1, len(q) + 1)
            ax3.plot(eps, q, label=name, color=ALGO_COLORS.get(name),
                     linewidth=2, linestyle=ALGO_LINESTYLES.get(name, '-'))
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Avg Queue Length')
        ax3.set_title('Queue Length Reduction')
        ax3.legend(fontsize=8)
        ax3.grid(True)

        # --- Panel 4: Throughput ---
        ax4 = fig.add_subplot(gs[1, 1])
        for name in algo_names:
            m = self.collector.get_algorithm(name)
            t = _smooth(np.array(m.throughputs), smooth_window)
            eps = np.arange(1, len(t) + 1)
            ax4.plot(eps, t, label=name, color=ALGO_COLORS.get(name),
                     linewidth=2, linestyle=ALGO_LINESTYLES.get(name, '-'))
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Throughput')
        ax4.set_title('Throughput Improvement')
        ax4.legend(fontsize=8)
        ax4.grid(True)

        # --- Panel 5: Training Loss (DQN & Double DQN) ---
        ax5 = fig.add_subplot(gs[2, 0])
        for name in ['DQN', 'Double DQN']:
            if name not in self.collector.algorithms:
                continue
            m = self.collector.get_algorithm(name)
            losses = np.array([l if l is not None else 0.0 for l in m.losses])
            smoothed = _smooth(losses, smooth_window)
            eps = np.arange(1, len(smoothed) + 1)
            ax5.plot(eps, smoothed, label=name, color=ALGO_COLORS.get(name),
                     linewidth=2, linestyle=ALGO_LINESTYLES.get(name, '-'))
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Training Loss')
        ax5.set_title('Training Loss (Neural Network Methods)')
        ax5.legend(fontsize=8)
        ax5.grid(True)

        # --- Panel 6: Final Bar Comparison ---
        ax6 = fig.add_subplot(gs[2, 1])
        table = self.collector.generate_comparison_table()
        algos = table['algorithm'].tolist()
        colors = [ALGO_COLORS.get(a, '#888888') for a in algos]
        vals = table['final_reward'].values
        x = np.arange(len(algos))
        bars = ax6.bar(x, vals, 0.55, color=colors, edgecolor='white')
        ax6.set_xticks(x)
        ax6.set_xticklabels(algos, fontsize=9, fontweight='bold')
        ax6.set_ylabel('Final Avg Reward')
        ax6.set_title('Final Reward Comparison')
        ax6.grid(axis='y')
        for bar, v in zip(bars, vals):
            ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{v:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        path = os.path.join(self.output_dir, 'combined_dashboard.png')
        fig.savefig(path)
        plt.close(fig)
        print(f"    Saved: {path}")

    # ─────────────────────────────────────────────────────────────────────
    #  8. Performance Table as Image
    # ─────────────────────────────────────────────────────────────────────
    def plot_comparison_table_image(self):
        """Render the comparison table as a professionally styled image."""
        table = self.collector.generate_comparison_table()

        fig, ax = plt.subplots(figsize=(14, 3 + 0.5 * len(table)))
        ax.axis('off')
        ax.set_title('Final Performance Comparison Table',
                      fontsize=15, fontweight='bold', pad=20)

        col_labels = ['Algorithm', 'Final Reward', 'Avg Wait Time (s)',
                       'Queue Length', 'Throughput', 'Convergence Ep.']
        cell_data = []
        row_colors = []

        for _, row in table.iterrows():
            cell_data.append([
                row['algorithm'],
                f"{row['final_reward']:.2f}",
                f"{row['final_avg_waiting_time']:.2f}",
                f"{row['final_queue_length']:.2f}",
                f"{row['final_throughput']:.2f}",
                str(int(row['convergence_episode']))
            ])
            c = ALGO_COLORS.get(row['algorithm'], '#888888')
            # Lighten color for cell background
            row_colors.append(c + '22')  # alpha hex

        tbl = ax.table(cellText=cell_data, colLabels=col_labels,
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
        for i in range(len(cell_data)):
            algo_name = cell_data[i][0]
            bg = ALGO_COLORS.get(algo_name, '#888888')
            for j in range(len(col_labels)):
                cell = tbl[i + 1, j]
                cell.set_facecolor(bg + '18')
                cell.set_edgecolor('#CCCCCC')

        path = os.path.join(self.output_dir, 'comparison_table.png')
        fig.savefig(path)
        plt.close(fig)
        print(f"    Saved: {path}")

    # ─────────────────────────────────────────────────────────────────────
    #  9. Unified Normalized Comparison (single consolidated view)
    # ─────────────────────────────────────────────────────────────────────
    def plot_unified_comparison(self, smooth_window: int = 20):
        """
        One comprehensive figure containing every comparison element:

        Row 1–5  : Normalised time-series panels (shared X-axis, Episode 1–500)
                   Reward · Waiting Time · Queue Length · Throughput · Training Loss
        Row 6    : Final-performance bar charts (Reward, Wait, Queue, Convergence)
        Row 7    : Summary performance table rendered as an image

        All earlier separate graphs are superseded by this single output.
        """
        algo_names = self.collector.get_all_names()
        nn_algos = [n for n in algo_names if n in ('DQN', 'Double DQN')]

        # ── Metric definitions for the time-series rows ─────────────────
        metric_defs = [
            ('Normalised Episode Reward',
             lambda m: np.array(m.rewards),
             '-', 2.2, algo_names),
            ('Normalised Average Waiting Time',
             lambda m: np.array(m.waiting_times),
             '--', 2.0, algo_names),
            ('Normalised Queue Length',
             lambda m: np.array(m.queue_lengths),
             ':', 2.0, algo_names),
            ('Normalised Throughput',
             lambda m: np.array(m.throughputs),
             '-.', 2.0, algo_names),
            ('Normalised Training Loss (DQN & Double DQN)',
             lambda m: np.array([l if l is not None else 0.0 for l in m.losses]),
             (0, (5, 1)), 1.6, nn_algos),
        ]

        n_ts = len(metric_defs)           # 5 time-series rows
        ts_h = 3.2                         # height per time-series panel
        bar_h = 4.0                        # height for bar-chart row
        tbl_h = 2.6                        # height for table row
        total_h = n_ts * ts_h + bar_h + tbl_h + 2.0  # +padding

        fig = plt.figure(figsize=(20, total_h))
        gs = GridSpec(
            n_ts + 2, 4, figure=fig,
            height_ratios=[ts_h] * n_ts + [bar_h, tbl_h],
            hspace=0.38, wspace=0.32
        )

        fig.suptitle(
            'Unified Reinforcement Learning Performance Comparison – 500 Episodes',
            fontsize=18, fontweight='bold', y=0.997
        )

        # ── Helper: min-max normalisation ───────────────────────────────
        def _minmax(arr: np.ndarray) -> np.ndarray:
            lo, hi = arr.min(), arr.max()
            if hi - lo < 1e-12:
                return np.zeros_like(arr)
            return (arr - lo) / (hi - lo)

        # ================================================================
        #  ROWS 0-4 : Normalised time-series panels (full-width, shared X)
        # ================================================================
        ts_axes = []
        for idx in range(n_ts):
            ax = fig.add_subplot(gs[idx, :], sharex=ts_axes[0] if ts_axes else None)
            ts_axes.append(ax)

        for idx, (title, getter, ls, lw, subset) in enumerate(metric_defs):
            ax = ts_axes[idx]
            for name in subset:
                m = self.collector.get_algorithm(name)
                raw = getter(m)
                smoothed = _smooth(raw, smooth_window)
                normed = _minmax(smoothed)
                episodes = np.arange(1, len(normed) + 1)

                color = ALGO_COLORS.get(name, '#888888')
                ax.plot(episodes, normed, label=name,
                        color=color, linewidth=lw, linestyle=ls)
                ax.fill_between(episodes, 0, normed, alpha=0.04, color=color)

            ax.set_ylabel('Normalised\n(0 – 1)', fontsize=9, fontweight='bold')
            ax.set_title(title, fontsize=11, fontweight='bold', loc='left', pad=4)
            ax.set_ylim(-0.05, 1.08)
            ax.grid(True)
            ax.legend(
                loc='upper left', fontsize=8, frameon=True,
                fancybox=True, shadow=False, ncol=len(subset),
                borderaxespad=0.3
            )
            # hide x-tick labels for all but the last time-series panel
            if idx < n_ts - 1:
                plt.setp(ax.get_xticklabels(), visible=False)

        ts_axes[-1].set_xlabel('Episode', fontsize=11, fontweight='bold')
        ts_axes[-1].set_xlim(1, max(
            len(self.collector.get_algorithm(algo_names[0]).rewards), 500
        ))

        # ================================================================
        #  ROW 5 : Final-performance bar charts  (4 side-by-side sub-panels)
        # ================================================================
        table = self.collector.generate_comparison_table()
        algos = table['algorithm'].tolist()
        n_algo = len(algos)
        x = np.arange(n_algo)
        width = 0.55
        colors = [ALGO_COLORS.get(a, '#888888') for a in algos]

        bar_specs = [
            ('Final Avg Reward',       'final_reward',           '{:+.1f}'),
            ('Final Avg Wait (s)',     'final_avg_waiting_time', '{:.1f}'),
            ('Final Queue Length',     'final_queue_length',     '{:.1f}'),
            ('Convergence Episode',    'convergence_episode',    'Ep {}'),
        ]

        for col_idx, (label, key, fmt) in enumerate(bar_specs):
            ax = fig.add_subplot(gs[n_ts, col_idx])
            vals = table[key].values
            if key == 'convergence_episode':
                vals = vals.astype(int)
            bars = ax.bar(x, vals, width, color=colors, edgecolor='white', linewidth=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(algos, fontsize=8, fontweight='bold', rotation=20, ha='right')
            ax.set_title(label, fontsize=10, fontweight='bold')
            ax.grid(axis='y')
            for bar, v in zip(bars, vals):
                txt = fmt.format(v) if '{}' in fmt else fmt.format(v)
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        txt, ha='center', va='bottom', fontsize=8, fontweight='bold')

        # ================================================================
        #  ROW 6 : Performance summary table
        # ================================================================
        ax_tbl = fig.add_subplot(gs[n_ts + 1, :])
        ax_tbl.axis('off')
        ax_tbl.set_title('Final Performance Comparison Table',
                          fontsize=11, fontweight='bold', pad=8)

        col_labels = ['Algorithm', 'Final Reward', 'Avg Wait (s)',
                       'Queue Length', 'Throughput', 'Conv. Episode']
        cell_data = []
        for _, row in table.iterrows():
            cell_data.append([
                row['algorithm'],
                f"{row['final_reward']:+.2f}",
                f"{row['final_avg_waiting_time']:.2f}",
                f"{row['final_queue_length']:.2f}",
                f"{row['final_throughput']:.2f}",
                str(int(row['convergence_episode']))
            ])

        tbl = ax_tbl.table(cellText=cell_data, colLabels=col_labels,
                            loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.auto_set_column_width(col=list(range(len(col_labels))))
        tbl.scale(1.0, 1.7)
        for j in range(len(col_labels)):
            cell = tbl[0, j]
            cell.set_facecolor('#2C3E50')
            cell.set_text_props(color='white', fontweight='bold')
        for i in range(len(cell_data)):
            bg = ALGO_COLORS.get(cell_data[i][0], '#888888')
            for j in range(len(col_labels)):
                cell = tbl[i + 1, j]
                cell.set_facecolor(bg + '18')
                cell.set_edgecolor('#CCCCCC')

        # ── Master legend below the figure ──────────────────────────────
        legend_handles = []
        legend_labels = []
        for name in algo_names:
            color = ALGO_COLORS.get(name, '#888888')
            for title, _, ls, lw, subset in metric_defs:
                if name not in subset:
                    continue
                short = title.replace('Normalised ', '').split('(')[0].strip()
                h, = ts_axes[0].plot([], [], color=color, linestyle=ls,
                                     linewidth=lw)
                legend_handles.append(h)
                legend_labels.append(f'{name} – {short}')

        fig.legend(
            legend_handles, legend_labels,
            loc='lower center', ncol=4, fontsize=8,
            frameon=True, fancybox=True, shadow=True,
            bbox_to_anchor=(0.5, -0.008)
        )

        path = os.path.join(self.output_dir, 'unified_comparison.png')
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {path}")

    # ─────────────────────────────────────────────────────────────────────
    #  Generate All
    # ─────────────────────────────────────────────────────────────────────
    def generate_all_plots(self):
        """Generate the unified comparison plot and the final efficiency bar graph."""
        print("\n  Generating unified comparison plot...")
        self.plot_unified_comparison()

        # Final efficiency bar graph (reads from saved CSV — real values)
        print("  Generating final efficiency bar graph...")
        from final_bar_graph import load_metrics, compute_reduction, \
            generate_final_bar_graph, print_summary_table
        df = load_metrics()
        df = compute_reduction(df)
        generate_final_bar_graph(df)
        print_summary_table(df)

        print(f"\n  All plots saved to: {self.output_dir}/")
