"""
Standard DQN Implementation for Traffic Signal Control
Uses single network for both action selection and Q-value evaluation.
Unlike Double DQN, this is susceptible to overestimation bias since
the same network selects and evaluates actions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Dict, List, Tuple, Optional
import random


class StandardDQNNetwork(nn.Module):
    """Neural network for Q-value estimation (same architecture as Double DQN)"""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(StandardDQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class StandardDQNAgent:
    """
    Standard DQN Agent for traffic signal control.

    Key difference from Double DQN:
    - Uses target network's max Q-value directly for computing targets
    - Q_target = r + gamma * max_a' Q_target(s', a')
    - This causes overestimation bias because the same max operation
      both selects and evaluates the action.
    """

    def __init__(self, state_size: int, action_size: int = 2,
                 learning_rate: float = 0.001, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.99065, device: str = 'cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = device

        # Single Q-network + target network (standard DQN uses target for stability)
        self.q_network = StandardDQNNetwork(state_size, action_size).to(device)
        self.target_network = StandardDQNNetwork(state_size, action_size).to(device)
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
        """Select action using epsilon-greedy policy"""
        if training and np.random.random() < self.epsilon:
            return random.choice(range(self.action_size))
        else:
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
        Train on a batch from replay buffer using STANDARD DQN update.

        Standard DQN: Q_target = r + gamma * max_a' Q_target(s', a')
        This uses the target network's own max to compute targets,
        which leads to overestimation bias.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(np.array(actions)).to(self.device)
        rewards_t = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.FloatTensor(np.array(dones)).to(self.device)

        # STANDARD DQN: Use target network's max Q-value directly
        # This is the key difference from Double DQN
        with torch.no_grad():
            next_q_target = self.target_network(next_states_t)
            max_next_q_values = next_q_target.max(dim=1)[0]

        # Compute target Q-values
        target_q_values = rewards_t + (1 - dones_t) * self.gamma * max_next_q_values

        # Compute current Q-values
        current_q_values = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Loss and backprop
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
