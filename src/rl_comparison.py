"""
RL Comparison Study — Orchestrator
===================================
Trains Q-Learning, SARSA, DQN, and Double DQN under identical conditions
on the traffic signal control environment, collects per-episode metrics,
generates professional comparison graphs and summary tables.

All algorithms share:
  - Same dataset (data/traffic_dataset.csv)
  - Same state/action space  (4-direction traffic, Hold/Switch actions)
  - Same reward function      R = -(0.6*wait + 0.3*queue + 0.1*switch)
  - Same episode length       100 steps
  - Same number of episodes   500
  - Same epsilon schedule     1.0 → 0.01
"""

import os
import sys
import time
import numpy as np
import random
import torch

# ── Path setup ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

# ── Imports from project modules ────────────────────────────────────────
from q_learning import DiscreteTrafficEnvironment, QLearningAgent
from sarsa import SARSAAgent
from dqn_training import TrafficEnvironment as DQNTrafficEnvironment, DoubleDQNAgent
from standard_dqn import StandardDQNAgent
from metrics_collector import ComparisonMetricsCollector
from comparison_visualizer import ComparisonVisualizer


# =========================================================================
#  Configuration — single source of truth for experimental parameters
# =========================================================================
class Config:
    DATASET_PATH = os.path.join(PROJECT_DIR, 'data', 'traffic_dataset.csv')
    NUM_EPISODES = 500
    EPISODE_LENGTH = 100
    OUTPUT_DIR = os.path.join(PROJECT_DIR, 'results', 'rl_comparison')
    MODELS_DIR = os.path.join(PROJECT_DIR, 'models', 'comparison')

    # Shared hyperparameters
    LEARNING_RATE_TABULAR = 0.1
    DISCOUNT_FACTOR = 0.95
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.99065  # 1.0 → ~0.01 over 500 episodes

    # DQN-specific
    DQN_LR = 0.001
    DQN_GAMMA = 0.99
    REPLAY_BUFFER = 10000
    BATCH_SIZE = 32
    TARGET_UPDATE_FREQ = 10  # episodes

    # Random seed for reproducibility
    SEED = 42


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================================
#  Training functions — one per algorithm
# =========================================================================

def train_q_learning(collector: ComparisonMetricsCollector, cfg: Config):
    """Train Q-Learning (tabular) and record metrics."""
    print("\n" + "=" * 70)
    print("  TRAINING: Q-Learning (Tabular, Off-Policy)")
    print("=" * 70)

    metrics = collector.register_algorithm('Q-Learning')
    env = DiscreteTrafficEnvironment(cfg.DATASET_PATH, episode_length=cfg.EPISODE_LENGTH)
    agent = QLearningAgent(
        learning_rate=cfg.LEARNING_RATE_TABULAR,
        discount_factor=cfg.DISCOUNT_FACTOR,
        epsilon=cfg.EPSILON_START,
        epsilon_decay=cfg.EPSILON_DECAY
    )

    for ep in range(cfg.NUM_EPISODES):
        state = env.reset()
        episode_reward = 0
        episode_loss = 0
        num_steps = 0

        while True:
            action = agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            td_error = agent.learn(state, action, reward, next_state, done)

            episode_reward += reward
            episode_loss += td_error
            num_steps += 1

            if done:
                break
            state = next_state

        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        avg_wait = env.episode_metrics['total_wait'] / max(num_steps, 1)
        avg_queue = env.episode_metrics['total_queue'] / max(num_steps, 1)
        throughput = env.episode_metrics['throughput']

        metrics.add_episode(
            episode=ep + 1,
            total_reward=episode_reward,
            avg_waiting_time=avg_wait,
            avg_queue_length=avg_queue,
            throughput=throughput,
            training_loss=None  # No neural-network loss for tabular
        )

        if (ep + 1) % 50 == 0:
            print(f"    Ep {ep+1:>4}/{cfg.NUM_EPISODES} | "
                  f"Reward: {episode_reward:>+8.2f} | "
                  f"Wait: {avg_wait:>6.2f}s | "
                  f"Queue: {avg_queue:>6.2f} | "
                  f"Eps: {agent.epsilon:.4f}")

    print("  Q-Learning training complete.")


def train_sarsa(collector: ComparisonMetricsCollector, cfg: Config):
    """Train SARSA (on-policy) and record metrics."""
    print("\n" + "=" * 70)
    print("  TRAINING: SARSA (On-Policy TD)")
    print("=" * 70)

    metrics = collector.register_algorithm('SARSA')
    env = DiscreteTrafficEnvironment(cfg.DATASET_PATH, episode_length=cfg.EPISODE_LENGTH)
    agent = SARSAAgent(
        learning_rate=cfg.LEARNING_RATE_TABULAR,
        discount_factor=cfg.DISCOUNT_FACTOR,
        epsilon=cfg.EPSILON_START,
        epsilon_decay=cfg.EPSILON_DECAY
    )

    for ep in range(cfg.NUM_EPISODES):
        state = env.reset()
        action = agent.select_action(state, training=True)
        episode_reward = 0
        episode_loss = 0
        num_steps = 0

        while True:
            next_state, reward, done, info = env.step(action)
            next_action = agent.select_action(next_state, training=True)
            td_error = agent.learn(state, action, reward, next_state, next_action, done)

            episode_reward += reward
            episode_loss += td_error
            num_steps += 1

            if done:
                break
            state = next_state
            action = next_action

        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        avg_wait = env.episode_metrics['total_wait'] / max(num_steps, 1)
        avg_queue = env.episode_metrics['total_queue'] / max(num_steps, 1)
        throughput = env.episode_metrics['throughput']

        metrics.add_episode(
            episode=ep + 1,
            total_reward=episode_reward,
            avg_waiting_time=avg_wait,
            avg_queue_length=avg_queue,
            throughput=throughput,
            training_loss=None
        )

        if (ep + 1) % 50 == 0:
            print(f"    Ep {ep+1:>4}/{cfg.NUM_EPISODES} | "
                  f"Reward: {episode_reward:>+8.2f} | "
                  f"Wait: {avg_wait:>6.2f}s | "
                  f"Queue: {avg_queue:>6.2f} | "
                  f"Eps: {agent.epsilon:.4f}")

    print("  SARSA training complete.")


def train_standard_dqn(collector: ComparisonMetricsCollector, cfg: Config):
    """Train Standard DQN and record metrics."""
    print("\n" + "=" * 70)
    print("  TRAINING: Standard DQN (with overestimation bias)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metrics = collector.register_algorithm('DQN')
    env = DQNTrafficEnvironment(cfg.DATASET_PATH, episode_length=cfg.EPISODE_LENGTH)
    agent = StandardDQNAgent(
        state_size=env.state_size,
        action_size=len(env.action_space),
        learning_rate=cfg.DQN_LR,
        gamma=cfg.DQN_GAMMA,
        epsilon_start=cfg.EPSILON_START,
        epsilon_end=cfg.EPSILON_END,
        epsilon_decay=cfg.EPSILON_DECAY,
        device=device
    )

    for ep in range(cfg.NUM_EPISODES):
        state = env.reset()
        episode_loss_sum = 0
        episode_loss_count = 0

        while True:
            action = agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)

            loss = agent.train_step()
            if loss is not None:
                episode_loss_sum += loss
                episode_loss_count += 1

            state = next_state
            if done:
                break

        if (ep + 1) % cfg.TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()
        agent.decay_epsilon()

        avg_loss = episode_loss_sum / max(episode_loss_count, 1)
        avg_wait = env.episode_metrics['total_wait'] / max(cfg.EPISODE_LENGTH, 1)
        avg_queue = env.episode_metrics['total_queue'] / max(cfg.EPISODE_LENGTH, 1)
        throughput = env.episode_metrics['throughput']

        metrics.add_episode(
            episode=ep + 1,
            total_reward=env.episode_reward,
            avg_waiting_time=avg_wait,
            avg_queue_length=avg_queue,
            throughput=throughput,
            training_loss=avg_loss
        )

        if (ep + 1) % 50 == 0:
            print(f"    Ep {ep+1:>4}/{cfg.NUM_EPISODES} | "
                  f"Reward: {env.episode_reward:>+8.2f} | "
                  f"Loss: {avg_loss:>8.4f} | "
                  f"Wait: {avg_wait:>6.2f}s | "
                  f"Eps: {agent.epsilon:.4f}")

    # Save model
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
    agent.save(os.path.join(cfg.MODELS_DIR, 'standard_dqn.pt'))
    print("  Standard DQN training complete.")


def train_double_dqn(collector: ComparisonMetricsCollector, cfg: Config):
    """Train Double DQN and record metrics (uses existing DoubleDQNAgent)."""
    print("\n" + "=" * 70)
    print("  TRAINING: Double DQN (reduced overestimation bias)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metrics = collector.register_algorithm('Double DQN')
    env = DQNTrafficEnvironment(cfg.DATASET_PATH, episode_length=cfg.EPISODE_LENGTH)
    agent = DoubleDQNAgent(
        state_size=env.state_size,
        action_size=len(env.action_space),
        learning_rate=cfg.DQN_LR,
        gamma=cfg.DQN_GAMMA,
        epsilon_start=cfg.EPSILON_START,
        epsilon_end=cfg.EPSILON_END,
        epsilon_decay=cfg.EPSILON_DECAY,
        device=device
    )

    for ep in range(cfg.NUM_EPISODES):
        state = env.reset()
        episode_loss_sum = 0
        episode_loss_count = 0

        while True:
            action = agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)

            loss = agent.train_step()
            if loss is not None:
                episode_loss_sum += loss
                episode_loss_count += 1

            state = next_state
            if done:
                break

        if (ep + 1) % cfg.TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()
        agent.decay_epsilon()

        avg_loss = episode_loss_sum / max(episode_loss_count, 1)
        avg_wait = env.episode_metrics['total_wait'] / max(cfg.EPISODE_LENGTH, 1)
        avg_queue = env.episode_metrics['total_queue'] / max(cfg.EPISODE_LENGTH, 1)
        throughput = env.episode_metrics['throughput']

        metrics.add_episode(
            episode=ep + 1,
            total_reward=env.episode_reward,
            avg_waiting_time=avg_wait,
            avg_queue_length=avg_queue,
            throughput=throughput,
            training_loss=avg_loss
        )

        if (ep + 1) % 50 == 0:
            print(f"    Ep {ep+1:>4}/{cfg.NUM_EPISODES} | "
                  f"Reward: {env.episode_reward:>+8.2f} | "
                  f"Loss: {avg_loss:>8.4f} | "
                  f"Wait: {avg_wait:>6.2f}s | "
                  f"Eps: {agent.epsilon:.4f}")

    # Save model
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
    agent.save(os.path.join(cfg.MODELS_DIR, 'double_dqn.pt'))
    print("  Double DQN training complete.")


# =========================================================================
#  Main Comparison Pipeline
# =========================================================================

def run_comparison(cfg: Config = None) -> ComparisonMetricsCollector:
    """
    Run the full RL comparison study.
    
    Returns:
        ComparisonMetricsCollector with all training results
    """
    if cfg is None:
        cfg = Config()

    # Validate dataset
    if not os.path.exists(cfg.DATASET_PATH):
        print(f"ERROR: Dataset not found at {cfg.DATASET_PATH}")
        print("       Run dataset_generator.py first.")
        sys.exit(1)

    set_seed(cfg.SEED)

    print("\n" + "#" * 70)
    print("#  REINFORCEMENT LEARNING COMPARISON STUDY")
    print("#  Traffic Signal Control — Q-Learning | SARSA | DQN | Double DQN")
    print("#" * 70)
    print(f"\n  Dataset        : {cfg.DATASET_PATH}")
    print(f"  Episodes       : {cfg.NUM_EPISODES}")
    print(f"  Episode Length  : {cfg.EPISODE_LENGTH} steps")
    print(f"  Epsilon        : {cfg.EPSILON_START} -> {cfg.EPSILON_END}")
    print(f"  Output Dir     : {cfg.OUTPUT_DIR}")
    print(f"  Device         : {'cuda' if torch.cuda.is_available() else 'cpu'}")

    collector = ComparisonMetricsCollector()
    start_time = time.time()

    # ── Train all algorithms ────────────────────────────────────────────
    train_q_learning(collector, cfg)
    train_sarsa(collector, cfg)
    train_standard_dqn(collector, cfg)
    train_double_dqn(collector, cfg)

    elapsed = time.time() - start_time
    print(f"\n  Total training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ── Save metrics ────────────────────────────────────────────────────
    print("\n  Saving per-episode metrics...")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    collector.save_all_metrics(cfg.OUTPUT_DIR)

    # ── Generate plots ──────────────────────────────────────────────────
    viz = ComparisonVisualizer(collector, cfg.OUTPUT_DIR)
    viz.generate_all_plots()

    # ── Print comparison table ──────────────────────────────────────────
    collector.print_comparison_table()

    print("\n" + "#" * 70)
    print("#  COMPARISON STUDY COMPLETE")
    print(f"#  Results saved to: {cfg.OUTPUT_DIR}")
    print("#" * 70 + "\n")

    return collector


if __name__ == '__main__':
    run_comparison()
