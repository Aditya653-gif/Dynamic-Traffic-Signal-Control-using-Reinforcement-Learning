"""
Phase 5: Q-Learning (Tabular) Implementation
Tabular Q-Learning for traffic signal control
Discrete state and action spaces
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime


class DiscreteTrafficEnvironment:
    """Discrete traffic environment for Q-Learning"""
    
    def __init__(self, dataset_path: str, start_idx: int = 0, episode_length: int = 100):
        """Initialize discrete traffic environment"""
        self.df = pd.read_csv(dataset_path)
        self.df.columns = self.df.columns.str.strip()
        
        self.start_idx = start_idx
        self.episode_length = episode_length
        self.current_step = 0
        self.current_idx = start_idx
        self.current_phase = 0
        self.phase_duration = 0
        
        # Discrete action space
        self.action_space = [0, 1]  # 0=Hold, 1=Switch
        
        # State discretization
        self.vehicle_bins = [0, 10, 20, 50, 100, 150]  # Vehicle count bins
        self.phase_bins = [0, 30, 60, 90, 120]  # Phase duration bins
        
        # Episode metrics
        self.episode_reward = 0
        self.episode_metrics = {
            'total_wait': 0,
            'total_queue': 0,
            'throughput': 0,
            'phase_switches': 0
        }
    
    def _discretize_vehicles(self, count):
        """Convert vehicle count to discrete bin"""
        for i, threshold in enumerate(self.vehicle_bins[1:]):
            if count < threshold:
                return i
        return len(self.vehicle_bins) - 1
    
    def _discretize_phase_duration(self, duration):
        """Convert phase duration to discrete bin"""
        for i, threshold in enumerate(self.phase_bins[1:]):
            if duration < threshold:
                return i
        return len(self.phase_bins) - 1
    
    def get_state(self) -> Tuple:
        """
        Get current state as discrete tuple
        Returns: (N_discrete, S_discrete, E_discrete, W_discrete, phase, phase_duration_discrete)
        """
        if self.current_idx >= len(self.df):
            self.current_idx = self.start_idx
        
        row = self.df.iloc[self.current_idx]
        
        # Discretize vehicle counts
        n_veh = self._discretize_vehicles(row['north_vehicle_count'])
        s_veh = self._discretize_vehicles(row['south_vehicle_count'])
        e_veh = self._discretize_vehicles(row['east_vehicle_count'])
        w_veh = self._discretize_vehicles(row['west_vehicle_count'])
        
        # Discretize phase duration
        phase_dur = self._discretize_phase_duration(self.phase_duration)
        
        # State is tuple of discrete values
        state = (n_veh, s_veh, e_veh, w_veh, self.current_phase, phase_dur)
        
        return state
    
    def step(self, action: int) -> Tuple:
        """
        Execute one step
        Returns: (next_state, reward, done, info)
        """
        row = self.df.iloc[self.current_idx]
        
        vehicles = {
            0: row['north_vehicle_count'],
            1: row['east_vehicle_count'],
            2: row['south_vehicle_count'],
            3: row['west_vehicle_count']
        }
        
        active_vehicles = vehicles[self.current_phase]
        max_throughput = int(0.75 * 30)
        throughput = min(int(active_vehicles), max_throughput)
        remaining_queue = max(0, active_vehicles - throughput)
        wait_time = 30 if remaining_queue > 0 else 15
        
        # Handle action
        switched = False
        if action == 1:  # Switch
            self.current_phase = (self.current_phase + 1) % 4
            self.phase_duration = 0
            switched = True
        else:  # Hold
            self.phase_duration = min(self.phase_duration + 30, 120)
        
        # Reward
        alpha, beta, gamma = 0.6, 0.3, 0.1
        reward = -(alpha * wait_time + beta * remaining_queue + gamma * int(switched))
        
        # Update metrics
        self.episode_reward += reward
        self.episode_metrics['total_wait'] += wait_time
        self.episode_metrics['total_queue'] += remaining_queue
        self.episode_metrics['throughput'] += throughput
        self.episode_metrics['phase_switches'] += int(switched)
        
        self.current_step += 1
        self.current_idx += 1
        
        done = (self.current_step >= self.episode_length) or (self.current_idx >= len(self.df))
        next_state = self.get_state()
        
        info = {
            'wait_time': wait_time,
            'queue_length': remaining_queue,
            'throughput': throughput,
            'phase_switched': switched
        }
        
        return next_state, reward, done, info
    
    def reset(self, start_idx: Optional[int] = None):
        """Reset environment"""
        if start_idx is not None:
            self.start_idx = start_idx
        
        self.current_idx = self.start_idx
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


class QLearningAgent:
    """Q-Learning agent for traffic signal control"""
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.95, 
                 epsilon: float = 1.0, epsilon_decay: float = 0.99065):
        """
        Initialize Q-Learning agent
        
        Args:
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial epsilon for exploration
            epsilon_decay: Epsilon decay rate (0.99065 for 500 episodes: 1.0 -> 0.01)
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        
        # Q-table: state -> action -> Q-value
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Training history
        self.episode_rewards = []
        self.episode_losses = []
        self.episode_avg_waiting = []
        self.episode_queue_lengths = []
        self.episode_convergence = []
    
    def select_action(self, state: Tuple, training: bool = True) -> int:
        """Select action using epsilon-greedy strategy"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, 2)  # Random action
        else:
            # Greedy action
            q_values = [self.q_table[state][a] for a in [0, 1]]
            return np.argmax(q_values)
    
    def learn(self, state: Tuple, action: int, reward: float, next_state: Tuple, done: bool):
        """Update Q-value using Q-Learning rule"""
        next_q_values = [self.q_table[next_state][a] for a in [0, 1]]
        max_next_q = max(next_q_values) if next_q_values else 0
        
        current_q = self.q_table[state][action]
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        
        self.q_table[state][action] = new_q
        
        return abs(new_q - current_q)  # TD error
    
    def train_episode(self, env: DiscreteTrafficEnvironment) -> Tuple[float, float, float]:
        """Train for one episode"""
        state = env.reset()
        episode_reward = 0
        episode_loss = 0
        num_steps = 0
        
        while True:
            action = self.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            
            td_error = self.learn(state, action, reward, next_state, done)
            
            episode_reward += reward
            episode_loss += td_error
            num_steps += 1
            
            if done:
                break
            
            state = next_state
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        avg_loss = episode_loss / num_steps if num_steps > 0 else 0
        avg_waiting_time = env.episode_metrics['total_wait'] / num_steps if num_steps > 0 else 0
        avg_queue_length = env.episode_metrics['total_queue'] / num_steps if num_steps > 0 else 0
        
        self.episode_rewards.append(episode_reward)
        self.episode_losses.append(avg_loss)
        self.episode_avg_waiting.append(avg_waiting_time)
        self.episode_queue_lengths.append(avg_queue_length)
        
        # Convergence metric (Q-value std)
        all_q_values = []
        for state_q_vals in self.q_table.values():
            all_q_values.extend(state_q_vals.values())
        
        convergence = np.std(all_q_values) if all_q_values else 0
        self.episode_convergence.append(convergence)
        
        return episode_reward, avg_loss, avg_waiting_time
    
    def evaluate(self, env: DiscreteTrafficEnvironment, num_episodes: int = 5) -> Dict:
        """Evaluate trained agent"""
        total_reward = 0
        total_wait = 0
        total_queue = 0
        total_throughput = 0
        total_switches = 0
        
        for _ in range(num_episodes):
            state = env.reset()
            while True:
                action = self.select_action(state, training=False)
                next_state, reward, done, info = env.step(action)
                
                total_reward += reward
                total_wait += info['wait_time']
                total_queue += info['queue_length']
                total_throughput += info['throughput']
                total_switches += int(info['phase_switched'])
                
                if done:
                    break
                state = next_state
        
        steps = num_episodes * env.episode_length
        
        return {
            'avg_reward': total_reward / num_episodes,
            'avg_waiting_time': total_wait / steps,
            'avg_queue_length': total_queue / steps,
            'avg_throughput': total_throughput / steps,
            'phase_switches': total_switches
        }


def train_q_learning(dataset_path: str, num_episodes: int = 500) -> QLearningAgent:
    """Train Q-Learning agent"""
    env = DiscreteTrafficEnvironment(dataset_path, episode_length=100)
    agent = QLearningAgent()
    
    print("Training Q-Learning Agent...")
    for episode in range(num_episodes):
        reward, loss, wait_time = agent.train_episode(env)
        
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}/{num_episodes} - Reward: {reward:.2f}, Loss: {loss:.4f}, Wait: {wait_time:.2f}s, Epsilon: {agent.epsilon:.4f}")
    
    return agent


if __name__ == '__main__':
    # Training example
    agent = train_q_learning('data/traffic_dataset.csv', num_episodes=500)
    
    # Evaluation
    env = DiscreteTrafficEnvironment('data/traffic_dataset.csv', episode_length=100)
    results = agent.evaluate(env, num_episodes=5)
    
    print("\nQ-Learning Evaluation Results:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")
