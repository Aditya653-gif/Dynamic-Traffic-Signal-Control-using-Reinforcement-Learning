"""
Phase 5: SARSA (State-Action-Reward-State-Action) Implementation
On-policy temporal difference learning for traffic signal control
Follows epsilon-greedy policy both for action selection and updates
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import json


class SARSAAgent:
    """SARSA agent for traffic signal control (On-policy TD learning)"""
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.95,
                 epsilon: float = 1.0, epsilon_decay: float = 0.99065):
        """
        Initialize SARSA agent
        
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
    
    def learn(self, state: Tuple, action: int, reward: float, 
              next_state: Tuple, next_action: int, done: bool) -> float:
        """
        Update Q-value using SARSA rule (On-Policy TD)
        
        SARSA: Q(s,a) = Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        
        Note: Uses Q-value of next_action (not max like Q-Learning)
        This makes it on-policy since we use the action we'll actually take
        """
        next_q = self.q_table[next_state][next_action] if not done else 0
        
        current_q = self.q_table[state][action]
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_q - current_q)
        
        self.q_table[state][action] = new_q
        
        return abs(new_q - current_q)  # TD error
    
    def train_episode(self, env) -> Tuple[float, float, float]:
        """
        Train for one episode using SARSA
        Note: env should have reset() and step() methods with same interface
        """
        state = env.reset()
        action = self.select_action(state, training=True)
        
        episode_reward = 0
        episode_loss = 0
        num_steps = 0
        
        while True:
            next_state, reward, done, info = env.step(action)
            next_action = self.select_action(next_state, training=True)
            
            td_error = self.learn(state, action, reward, next_state, next_action, done)
            
            episode_reward += reward
            episode_loss += td_error
            num_steps += 1
            
            if done:
                break
            
            state = next_state
            action = next_action
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        avg_loss = episode_loss / num_steps if num_steps > 0 else 0
        avg_waiting_time = env.episode_metrics['total_wait'] / num_steps if num_steps > 0 else 0
        avg_queue_length = env.episode_metrics['total_queue'] / num_steps if num_steps > 0 else 0
        
        self.episode_rewards.append(episode_reward)
        self.episode_losses.append(avg_loss)
        self.episode_avg_waiting.append(avg_waiting_time)
        self.episode_queue_lengths.append(avg_queue_length)
        
        # Convergence metric
        all_q_values = []
        for state_q_vals in self.q_table.values():
            all_q_values.extend(state_q_vals.values())
        
        convergence = np.std(all_q_values) if all_q_values else 0
        self.episode_convergence.append(convergence)
        
        return episode_reward, avg_loss, avg_waiting_time
    
    def evaluate(self, env, num_episodes: int = 5) -> Dict:
        """Evaluate trained SARSA agent"""
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
def train_sarsa(dataset_path: str, num_episodes: int = 500) -> SARSAAgent:
    """Train SARSA agent"""
    # Import here to avoid circular imports
    from q_learning import DiscreteTrafficEnvironment
    
    env = DiscreteTrafficEnvironment(dataset_path, episode_length=100)
    agent = SARSAAgent()
    
    print("Training SARSA Agent...")
    for episode in range(num_episodes):
        reward, loss, wait_time = agent.train_episode(env)
        
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}/{num_episodes} - Reward: {reward:.2f}, Loss: {loss:.4f}, Wait: {wait_time:.2f}s, Epsilon: {agent.epsilon:.4f}")
    
    return agent


if __name__ == '__main__':
    agent = train_sarsa('data/traffic_dataset.csv', num_episodes=500)
    
    # Evaluation
    from q_learning import DiscreteTrafficEnvironment
    env = DiscreteTrafficEnvironment('data/traffic_dataset.csv', episode_length=100)
    results = agent.evaluate(env, num_episodes=5)
    
    print("\nSARSA Evaluation Results:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")
