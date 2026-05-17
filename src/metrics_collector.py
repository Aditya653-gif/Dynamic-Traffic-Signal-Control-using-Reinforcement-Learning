"""
Metrics Collector for RL Comparison Study
Collects, stores, and manages per-episode metrics for all algorithms.
Provides convergence detection and final performance summary.
"""

import numpy as np
import pandas as pd
import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class EpisodeMetrics:
    """Metrics for a single training episode"""
    episode: int
    total_reward: float
    avg_waiting_time: float
    avg_queue_length: float
    throughput: float
    training_loss: Optional[float] = None  # Only for DQN/Double DQN


class AlgorithmMetrics:
    """Collects and manages metrics for a single algorithm"""

    def __init__(self, algorithm_name: str):
        self.algorithm_name = algorithm_name
        self.episodes: List[EpisodeMetrics] = []

    def add_episode(self, episode: int, total_reward: float,
                    avg_waiting_time: float, avg_queue_length: float,
                    throughput: float, training_loss: Optional[float] = None):
        """Record metrics for one episode"""
        self.episodes.append(EpisodeMetrics(
            episode=episode,
            total_reward=total_reward,
            avg_waiting_time=avg_waiting_time,
            avg_queue_length=avg_queue_length,
            throughput=throughput,
            training_loss=training_loss
        ))

    @property
    def rewards(self) -> List[float]:
        return [e.total_reward for e in self.episodes]

    @property
    def waiting_times(self) -> List[float]:
        return [e.avg_waiting_time for e in self.episodes]

    @property
    def queue_lengths(self) -> List[float]:
        return [e.avg_queue_length for e in self.episodes]

    @property
    def throughputs(self) -> List[float]:
        return [e.throughput for e in self.episodes]

    @property
    def losses(self) -> List[Optional[float]]:
        return [e.training_loss for e in self.episodes]

    def get_smoothed(self, data: List[float], window: int = 20) -> np.ndarray:
        """Apply moving average smoothing"""
        if len(data) < window:
            return np.array(data)
        kernel = np.ones(window) / window
        # Pad edges to maintain length
        padded = np.pad(data, (window // 2, window - 1 - window // 2), mode='edge')
        return np.convolve(padded, kernel, mode='valid')[:len(data)]

    def compute_convergence_episode(self, window: int = 50, 
                                     variance_threshold: float = None) -> int:
        """
        Detect convergence episode using sliding window variance.
        
        Convergence = the first episode where the reward variance over
        a sliding window drops below a threshold and stays stable for
        the remaining training episodes.
        
        Args:
            window: Size of sliding window
            variance_threshold: Variance threshold (auto-computed if None)
            
        Returns:
            Convergence episode number (1-indexed), or total episodes if not converged
        """
        rewards = np.array(self.rewards)
        n = len(rewards)

        if n < window * 2:
            return n  # Not enough data

        # Compute sliding window variance
        variances = []
        for i in range(n - window + 1):
            win = rewards[i:i + window]
            variances.append(np.var(win))

        variances = np.array(variances)

        # Auto-compute threshold: use a fraction of overall variance
        if variance_threshold is None:
            # Use the minimum variance in the last quarter as reference
            last_quarter = variances[len(variances) * 3 // 4:]
            if len(last_quarter) > 0:
                baseline_var = np.median(last_quarter)
                # Threshold = 1.5x the baseline variance 
                variance_threshold = baseline_var * 1.5
            else:
                return n

        # Find first episode where variance drops below threshold
        # and stays below for at least `window` more episodes
        for i in range(len(variances) - window):
            if variances[i] <= variance_threshold:
                # Check if it stays below for rest of this check window
                remaining = variances[i:i + window]
                if np.all(remaining <= variance_threshold * 1.2):
                    return i + 1  # 1-indexed episode number

        # If never fully stable, return the point of minimum variance
        return int(np.argmin(variances)) + 1

    def get_final_metrics(self, last_n: int = 50) -> Dict:
        """Get final averaged performance metrics (last N episodes)"""
        n = min(last_n, len(self.episodes))
        last_episodes = self.episodes[-n:]

        return {
            'algorithm': self.algorithm_name,
            'final_reward': np.mean([e.total_reward for e in last_episodes]),
            'final_avg_waiting_time': np.mean([e.avg_waiting_time for e in last_episodes]),
            'final_queue_length': np.mean([e.avg_queue_length for e in last_episodes]),
            'final_throughput': np.mean([e.throughput for e in last_episodes]),
            'convergence_episode': self.compute_convergence_episode()
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame"""
        records = []
        for e in self.episodes:
            records.append({
                'algorithm': self.algorithm_name,
                'episode': e.episode,
                'total_reward': e.total_reward,
                'avg_waiting_time': e.avg_waiting_time,
                'avg_queue_length': e.avg_queue_length,
                'throughput': e.throughput,
                'training_loss': e.training_loss
            })
        return pd.DataFrame(records)


class ComparisonMetricsCollector:
    """Collects metrics for all algorithms and generates comparison outputs"""

    def __init__(self):
        self.algorithms: Dict[str, AlgorithmMetrics] = {}

    def register_algorithm(self, name: str) -> AlgorithmMetrics:
        """Register a new algorithm for comparison"""
        self.algorithms[name] = AlgorithmMetrics(name)
        return self.algorithms[name]

    def get_algorithm(self, name: str) -> AlgorithmMetrics:
        """Get metrics collector for specific algorithm"""
        return self.algorithms[name]

    def get_all_names(self) -> List[str]:
        """Get all registered algorithm names"""
        return list(self.algorithms.keys())

    def generate_comparison_table(self, last_n: int = 50) -> pd.DataFrame:
        """
        Generate final performance comparison table.
        
        Returns DataFrame with columns:
        - Algorithm, Final Reward, Final Avg Waiting Time, 
          Final Queue Length, Convergence Episode
        """
        rows = []
        for name, metrics in self.algorithms.items():
            final = metrics.get_final_metrics(last_n)
            rows.append(final)

        df = pd.DataFrame(rows)
        # Sort by final reward (best first)
        df = df.sort_values('final_reward', ascending=False).reset_index(drop=True)
        return df

    def save_all_metrics(self, output_dir: str):
        """Save all per-episode metrics to CSV files"""
        os.makedirs(output_dir, exist_ok=True)

        # Per-algorithm CSVs
        for name, metrics in self.algorithms.items():
            df = metrics.to_dataframe()
            safe_name = name.lower().replace(' ', '_').replace('-', '_')
            filepath = os.path.join(output_dir, f'{safe_name}_episode_metrics.csv')
            df.to_csv(filepath, index=False)

        # Combined CSV
        all_dfs = []
        for name, metrics in self.algorithms.items():
            all_dfs.append(metrics.to_dataframe())
        
        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            combined.to_csv(os.path.join(output_dir, 'all_algorithms_metrics.csv'), index=False)

        # Comparison table
        comparison = self.generate_comparison_table()
        comparison.to_csv(os.path.join(output_dir, 'final_comparison_table.csv'), index=False)

        # JSON summary
        summary = {}
        for name, metrics in self.algorithms.items():
            final = metrics.get_final_metrics()
            final['convergence_episode'] = int(final['convergence_episode'])
            summary[name] = final
        
        with open(os.path.join(output_dir, 'comparison_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"  All metrics saved to {output_dir}/")

    def print_comparison_table(self):
        """Print formatted comparison table to console"""
        table = self.generate_comparison_table()

        print("\n" + "=" * 100)
        print("  REINFORCEMENT LEARNING ALGORITHM COMPARISON - FINAL PERFORMANCE")
        print("=" * 100)
        print(f"\n  {'Algorithm':<16} {'Final Reward':>14} {'Avg Wait Time':>15} "
              f"{'Queue Length':>14} {'Throughput':>12} {'Convergence':>14}")
        print("  " + "-" * 90)

        for _, row in table.iterrows():
            print(f"  {row['algorithm']:<16} {row['final_reward']:>+14.2f} "
                  f"{row['final_avg_waiting_time']:>15.2f} "
                  f"{row['final_queue_length']:>14.2f} "
                  f"{row['final_throughput']:>12.2f} "
                  f"{'Ep ' + str(int(row['convergence_episode'])):>14}")

        print("  " + "-" * 90)
        print()
