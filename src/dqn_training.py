"""
Phase 3: Double Deep Q-Network (DQN) Adaptive Traffic Signal Control System
Implements state-of-the-art reinforcement learning for optimal signal timing
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from datetime import datetime
import os
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import random


class TrafficEnvironment:
    """Traffic simulation environment (Gym-like interface)"""
    
    def __init__(self, dataset_path: str, start_idx: int = 0, episode_length: int = 100):
        """
        Initialize traffic environment
        
        Args:
            dataset_path: Path to traffic dataset CSV
            start_idx: Starting index in dataset
            episode_length: Number of steps per episode
        """
        self.df = pd.read_csv(dataset_path)
        self.df.columns = self.df.columns.str.strip()
        
        self.start_idx = start_idx
        self.episode_length = episode_length
        self.current_step = 0
        
        # Current state
        self.current_idx = start_idx
        self.current_phase = 0  # 0=North, 1=East, 2=South, 3=West
        self.phase_duration = 0
        
        # Action space: 0=Hold, 1=Switch
        self.action_space = [0, 1]
        
        # State dimensions: 4 vehicle counts + 4 phase one-hot + 1 phase duration
        self.state_size = 9  # N,S,E,W vehicles (4) + phase one-hot (4) + phase duration (1)
        
        # Reward tracking
        self.episode_reward = 0
        self.episode_metrics = {
            'total_wait': 0,
            'total_queue': 0,
            'throughput': 0,
            'phase_switches': 0
        }
    
    def get_state(self) -> np.ndarray:
        """
        Get current state as normalized vector
        
        State: [N_veh, S_veh, E_veh, W_veh, phase_onehot(4)]
        Returns normalized state in [0, 1]
        """
        row = self.df.iloc[self.current_idx]
        
        vehicles = np.array([
            row['north_vehicle_count'],
            row['south_vehicle_count'],
            row['east_vehicle_count'],
            row['west_vehicle_count']
        ], dtype=np.float32)
        
        # Normalize to [0, 1]
        vehicles_norm = vehicles / max(vehicles.max(), 1.0)
        
        # One-hot encode current phase
        phase_onehot = np.zeros(4, dtype=np.float32)
        phase_onehot[self.current_phase] = 1.0
        
        # Phase duration normalized (0-2, represent 0-120 seconds)
        phase_duration_norm = np.array([self.phase_duration / 120.0], dtype=np.float32)
        
        # Combine state
        state = np.concatenate([vehicles_norm, phase_onehot, phase_duration_norm])
        
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: 0=Hold current phase, 1=Switch to next phase
            
        Returns:
            (next_state, reward, done, info)
        """
        row = self.df.iloc[self.current_idx]
        
        # Get current metrics
        vehicles = {
            'North': row['north_vehicle_count'],
            'South': row['south_vehicle_count'],
            'East': row['east_vehicle_count'],
            'West': row['west_vehicle_count']
        }
        
        current_phase_name = ['North', 'East', 'South', 'West'][self.current_phase]
        active_vehicles = vehicles[current_phase_name]
        
        # Calculate throughput (vehicles passing through)
        max_throughput = int(0.75 * 30)  # 0.75 veh/sec * 30 sec phase
        throughput = min(int(active_vehicles), max_throughput)
        
        # Calculate wait time and queue
        remaining_queue = max(0, active_vehicles - throughput)
        wait_time = 30 if remaining_queue > 0 else 15  # Simplified
        
        # Handle action
        switched = False
        if action == 1:  # Switch phase
            self.current_phase = (self.current_phase + 1) % 4
            self.phase_duration = 0
            switched = True
        else:  # Hold phase
            self.phase_duration = min(self.phase_duration + 30, 120)
        
        # Calculate reward (improved version)
        alpha, beta, gamma = 0.6, 0.3, 0.1
        reward = -(alpha * wait_time + beta * remaining_queue + gamma * int(switched))
        
        # Update episode metrics
        self.episode_reward += reward
        self.episode_metrics['total_wait'] += wait_time
        self.episode_metrics['total_queue'] += remaining_queue
        self.episode_metrics['throughput'] += throughput
        self.episode_metrics['phase_switches'] += int(switched)
        
        # Move to next step
        self.current_step += 1
        self.current_idx += 1
        
        # Handle episode end
        done = (self.current_step >= self.episode_length) or (self.current_idx >= len(self.df))
        
        # Get next state
        if done:
            next_state = self.get_state()
        else:
            next_state = self.get_state()
        
        info = {
            'phase_switched': switched,
            'throughput': throughput,
            'wait_time': wait_time,
            'queue': remaining_queue
        }
        
        return next_state, reward, done, info
    
    def reset(self, start_idx: Optional[int] = None) -> np.ndarray:
        """Reset environment to start of new episode"""
        if start_idx is not None:
            self.start_idx = start_idx
            self.current_idx = start_idx
        else:
            # Random start index
            max_idx = max(0, len(self.df) - self.episode_length)
            self.current_idx = np.random.randint(0, max_idx + 1) if max_idx > 0 else 0
            self.start_idx = self.current_idx
        
        self.current_step = 0
        self.current_phase = 0
        self.phase_duration = 0
        self.episode_reward = 0
        self.episode_metrics = {
            'total_wait': 0,
            'total_queue': 0,
            'throughput': 0,
            'phase_switches': 0
        }
        
        return self.get_state()


class DQNNetwork(nn.Module):
    """Neural network for Q-value estimation"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        """
        Initialize DQN network
        
        Args:
            state_size: Dimension of state vector
            action_size: Number of possible actions
            hidden_size: Size of hidden layers
        """
        super(DQNNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
        self.relu = nn.ReLU()
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through network"""
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class DoubleDQNAgent:
    """Double DQN Agent for traffic signal control"""
    
    def __init__(self, state_size: int, action_size: int = 2, 
                 learning_rate: float = 0.001, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.99065, device: str = 'cpu'):
        """
        Initialize Double DQN Agent
        
        Args:
            state_size: Dimension of state
            action_size: Number of actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Starting exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Epsilon decay rate (0.99065 for 500 episodes: 1.0 -> 0.01)
            device: 'cpu' or 'cuda'
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = device
        
        # Networks
        self.q_network = DQNNetwork(state_size, action_size).to(device)
        self.target_network = DQNNetwork(state_size, action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 32
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Training metrics
        self.loss_history = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: Whether in training mode (affects epsilon)
            
        Returns:
            Action index
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return random.choice(range(self.action_size))
        else:
            # Exploit: choose best action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool):
        """Add experience to replay buffer"""
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def train_step(self) -> Optional[float]:
        """
        Train on a batch from replay buffer
        
        Returns:
            Loss value or None if buffer too small
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(np.array(actions)).to(self.device)
        rewards_t = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.FloatTensor(np.array(dones)).to(self.device)
        
        # Double DQN update rule
        # 1. Use main network to select best action in next state
        with torch.no_grad():
            next_q_main = self.q_network(next_states_t)
            best_actions = next_q_main.argmax(dim=1)
        
        # 2. Use target network to evaluate selected actions
        with torch.no_grad():
            next_q_target = self.target_network(next_states_t)
            max_next_q_values = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
        
        # 3. Compute target Q-values
        target_q_values = rewards_t + (1 - dones_t) * self.gamma * max_next_q_values
        
        # 4. Compute current Q-values
        current_q_values = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        
        # 5. Compute loss and backprop
        loss = self.loss_fn(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.loss_history.append(loss.item())
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network weights from main network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay epsilon for exploration"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, path: str):
        """Save trained model"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path: str):
        """Load trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.epsilon = checkpoint['epsilon']


class DQNTrainer:
    """Trainer for Double DQN Agent"""
    
    def __init__(self, dataset_path: str, num_trials: int = 5, 
                 episodes_per_trial: int = 500, episode_length: int = 100):
        """
        Initialize trainer
        
        Args:
            dataset_path: Path to traffic dataset
            num_trials: Number of independent training runs
            episodes_per_trial: Episodes per trial (500 for 500-episode training)
            episode_length: Steps per episode
        """
        self.dataset_path = dataset_path
        self.num_trials = num_trials
        self.episodes_per_trial = episodes_per_trial
        self.episode_length = episode_length
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Training metrics storage
        self.trials_history = []
    
    def train_single_trial(self, trial_idx: int) -> Dict:
        """
        Run single training trial
        
        Args:
            trial_idx: Trial number
            
        Returns:
            Trial history dictionary
        """
        print(f"\n{'='*80}")
        print(f"TRIAL {trial_idx + 1}/{self.num_trials}")
        print(f"{'='*80}")
        
        # Initialize environment and agent
        env = TrafficEnvironment(self.dataset_path, episode_length=self.episode_length)
        agent = DoubleDQNAgent(
            state_size=env.state_size,
            action_size=len(env.action_space),
            device=self.device
        )
        
        # Training history
        trial_history = {
            'episode_rewards': [],
            'episode_losses': [],
            'average_waits': [],
            'average_queues': [],
            'total_switches': [],
            'loss_per_episode': []
        }
        
        # Training loop
        for episode in range(self.episodes_per_trial):
            state = env.reset()
            episode_loss_sum = 0
            episode_loss_count = 0
            
            # Episode loop
            while True:
                # Select and execute action
                action = agent.select_action(state, training=True)
                next_state, reward, done, info = env.step(action)
                
                # Store in replay buffer
                agent.remember(state, action, reward, next_state, done)
                
                # Train on batch
                loss = agent.train_step()
                if loss is not None:
                    episode_loss_sum += loss
                    episode_loss_count += 1
                
                state = next_state
                
                if done:
                    break
            
            # Update target network every 10 episodes
            if (episode + 1) % 10 == 0:
                agent.update_target_network()
            
            # Decay epsilon
            agent.decay_epsilon()
            
            # Record metrics
            avg_loss = episode_loss_sum / max(episode_loss_count, 1)
            avg_wait = env.episode_metrics['total_wait'] / max(self.episode_length, 1)
            avg_queue = env.episode_metrics['total_queue'] / max(self.episode_length, 1)
            
            trial_history['episode_rewards'].append(env.episode_reward)
            trial_history['episode_losses'].append(avg_loss)
            trial_history['average_waits'].append(avg_wait)
            trial_history['average_queues'].append(avg_queue)
            trial_history['total_switches'].append(env.episode_metrics['phase_switches'])
            trial_history['loss_per_episode'].append(avg_loss)
            
            if (episode + 1) % 10 == 0:
                print(f"  Episode {episode + 1}/{self.episodes_per_trial} | "
                      f"Reward: {env.episode_reward:+.2f} | "
                      f"Avg Loss: {avg_loss:.4f} | "
                      f"Epsilon: {agent.epsilon:.4f}")
        
        # Save trained model
        model_path = os.path.join(
            os.path.dirname(self.dataset_path),
            '..',
            'models',
            f'dqn_model_trial_{trial_idx + 1}.pt'
        )
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        agent.save(model_path)
        print(f"  ✓ Model saved to {model_path}")
        
        return trial_history
    
    def train(self) -> Dict:
        """
        Run multiple training trials and average results
        
        Returns:
            Aggregated training history
        """
        print("\n" + "="*80)
        print("[TRAFFIC] PHASE 3: DOUBLE DQN TRAINING (500 EPISODES)")
        print("="*80)
        print(f"\nTraining Configuration:")
        print(f"  Trials: {self.num_trials}")
        print(f"  Episodes per trial: {self.episodes_per_trial}")
        print(f"  Steps per episode: {self.episode_length}")
        print(f"  Replay buffer size: 10,000")
        print(f"  Network: 2 hidden layers (128 units each, ReLU)")
        print(f"  Epsilon decay: 1.0 → 0.01 over 500 episodes")
        print(f"  Device: {self.device}")
        
        # Run all trials
        for trial in range(self.num_trials):
            trial_history = self.train_single_trial(trial)
            self.trials_history.append(trial_history)
        
        # Aggregate results
        aggregated_history = self._aggregate_trials()
        
        return aggregated_history
    
    def _aggregate_trials(self) -> Dict:
        """Aggregate results from multiple trials"""
        aggregated = {
            'mean_episode_rewards': [],
            'std_episode_rewards': [],
            'mean_episode_losses': [],
            'std_episode_losses': [],
            'mean_average_waits': [],
            'mean_average_queues': [],
            'mean_total_switches': []
        }
        
        # Average across trials for each episode
        for episode in range(self.episodes_per_trial):
            rewards = [trial['episode_rewards'][episode] for trial in self.trials_history]
            losses = [trial['episode_losses'][episode] for trial in self.trials_history]
            waits = [trial['average_waits'][episode] for trial in self.trials_history]
            queues = [trial['average_queues'][episode] for trial in self.trials_history]
            switches = [trial['total_switches'][episode] for trial in self.trials_history]
            
            aggregated['mean_episode_rewards'].append(np.mean(rewards))
            aggregated['std_episode_rewards'].append(np.std(rewards))
            aggregated['mean_episode_losses'].append(np.mean(losses))
            aggregated['std_episode_losses'].append(np.std(losses))
            aggregated['mean_average_waits'].append(np.mean(waits))
            aggregated['mean_average_queues'].append(np.mean(queues))
            aggregated['mean_total_switches'].append(np.mean(switches))
        
        return aggregated
    
    def save_results(self, aggregated_history: Dict, output_dir: str = None) -> str:
        """Save training results"""
        if output_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_dir = os.path.dirname(script_dir)
            output_dir = os.path.join(project_dir, 'results')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save aggregated metrics
        results_file = os.path.join(output_dir, 'dqn_training_results.json')
        with open(results_file, 'w') as f:
            json.dump(aggregated_history, f, indent=2)
        
        print(f"\n✓ Training results saved to {results_file}")
        
        return results_file
    
    def plot_training_metrics(self, aggregated_history: Dict, output_dir: str = None):
        """Generate training visualization plots"""
        if output_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_dir = os.path.dirname(script_dir)
            output_dir = os.path.join(project_dir, 'results')
        
        os.makedirs(output_dir, exist_ok=True)
        
        episodes = range(1, len(aggregated_history['mean_episode_rewards']) + 1)
        
        # Figure 1: Reward and Loss curves
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Double DQN Training Metrics (Averaged over 5 Trials)', 
                     fontsize=16, fontweight='bold')
        
        # Reward convergence
        ax = axes[0, 0]
        mean_rewards = aggregated_history['mean_episode_rewards']
        std_rewards = aggregated_history['std_episode_rewards']
        ax.plot(episodes, mean_rewards, 'b-', linewidth=2, label='Mean Reward')
        ax.fill_between(episodes, 
                       np.array(mean_rewards) - np.array(std_rewards),
                       np.array(mean_rewards) + np.array(std_rewards),
                       alpha=0.3, label='±1 Std Dev')
        ax.set_xlabel('Episode', fontsize=11, fontweight='bold')
        ax.set_ylabel('Episode Reward', fontsize=11, fontweight='bold')
        ax.set_title('Reward Convergence', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Loss convergence
        ax = axes[0, 1]
        mean_losses = aggregated_history['mean_episode_losses']
        std_losses = aggregated_history['std_episode_losses']
        ax.plot(episodes, mean_losses, 'r-', linewidth=2, label='Mean Loss')
        ax.fill_between(episodes,
                       np.array(mean_losses) - np.array(std_losses),
                       np.array(mean_losses) + np.array(std_losses),
                       alpha=0.3, label='±1 Std Dev')
        ax.set_xlabel('Episode', fontsize=11, fontweight='bold')
        ax.set_ylabel('Average Loss', fontsize=11, fontweight='bold')
        ax.set_title('Training Loss Convergence', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Average wait time
        ax = axes[1, 0]
        mean_waits = aggregated_history['mean_average_waits']
        ax.plot(episodes, mean_waits, 'g-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Episode', fontsize=11, fontweight='bold')
        ax.set_ylabel('Average Wait Time (seconds)', fontsize=11, fontweight='bold')
        ax.set_title('Wait Time Improvement During Training', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Average queue length
        ax = axes[1, 1]
        mean_queues = aggregated_history['mean_average_queues']
        ax.plot(episodes, mean_queues, 'orange', linewidth=2, marker='s', markersize=4)
        ax.set_xlabel('Episode', fontsize=11, fontweight='bold')
        ax.set_ylabel('Average Queue Length (vehicles)', fontsize=11, fontweight='bold')
        ax.set_title('Queue Length Reduction During Training', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, 'dqn_training_curves.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Training curves saved to {plot_file}")
        plt.close()
        
        # Figure 2: Phase switches (stability)
        fig, ax = plt.subplots(figsize=(12, 6))
        mean_switches = aggregated_history['mean_total_switches']
        ax.bar(episodes, mean_switches, color='steelblue', alpha=0.7)
        ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Phase Switches', fontsize=12, fontweight='bold')
        ax.set_title('Phase Switching Frequency (Lower = More Stable)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        switches_file = os.path.join(output_dir, 'dqn_phase_switches.png')
        plt.savefig(switches_file, dpi=300, bbox_inches='tight')
        print(f"✓ Phase switches plot saved to {switches_file}")
        plt.close()
    
    def print_summary(self, aggregated_history: Dict):
        """Print training summary"""
        print("\n" + "="*80)
        print("📊 DOUBLE DQN TRAINING SUMMARY")
        print("="*80)
        
        rewards = aggregated_history['mean_episode_rewards']
        losses = aggregated_history['mean_episode_losses']
        waits = aggregated_history['mean_average_waits']
        
        print(f"\n🎯 CONVERGENCE METRICS:")
        print(f"  Initial Reward: {rewards[0]:+.2f}")
        print(f"  Final Reward: {rewards[-1]:+.2f}")
        print(f"  Reward Improvement: {rewards[-1] - rewards[0]:+.2f} ({(rewards[-1] - rewards[0])/abs(rewards[0])*100:+.1f}%)")
        
        print(f"\n📉 LOSS CONVERGENCE:")
        print(f"  Initial Loss: {losses[0]:.4f}")
        print(f"  Final Loss: {losses[-1]:.4f}")
        print(f"  Loss Reduction: {losses[0] - losses[-1]:.4f} ({(losses[0] - losses[-1])/losses[0]*100:.1f}%)")
        
        print(f"\n⏱️  WAIT TIME REDUCTION:")
        print(f"  Initial Avg Wait: {waits[0]:.2f}s")
        print(f"  Final Avg Wait: {waits[-1]:.2f}s")
        print(f"  Improvement: {waits[0] - waits[-1]:.2f}s ({(waits[0] - waits[-1])/waits[0]*100:.1f}%)")
        
        print("\n" + "="*80)


def main():
    """Main training execution"""
    print("\n[TRAFFIC] PHASE 3: DOUBLE DQN FOR ADAPTIVE TRAFFIC CONTROL\n")
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    dataset_path = os.path.join(project_dir, 'data', 'traffic_dataset.csv')
    
    # Check dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at {dataset_path}")
        print("   Please run Phase 1 (dataset_generator.py) first")
        return
    
    # Initialize trainer
    trainer = DQNTrainer(
        dataset_path=dataset_path,
        num_trials=5,
        episodes_per_trial=500,
        episode_length=100
    )
    
    # Run training
    aggregated_history = trainer.train()
    
    # Save results
    trainer.save_results(aggregated_history)
    
    # Generate plots
    trainer.plot_training_metrics(aggregated_history)
    
    # Print summary
    trainer.print_summary(aggregated_history)
    
    print("\n✅ Phase 3 Training Complete!")
    print("   Check 'results/' folder for model files and visualizations\n")
    
    return trainer, aggregated_history


if __name__ == "__main__":
    trainer, history = main()
