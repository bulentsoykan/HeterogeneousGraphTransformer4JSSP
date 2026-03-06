"""
Proximal Policy Optimization (PPO) for JSSP with Heterogeneous Graph States.

Implements the PPO algorithm (Schulman et al., 2017) adapted for:
- Graph-structured states (HeteroData)
- Variable-length episodes
- Action masking for invalid actions

Key Features:
- Clipped surrogate objective for stable policy updates
- Generalized Advantage Estimation (GAE)
- Value function clipping
- Entropy regularization
- Gradient clipping
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from torch_geometric.data import HeteroData, Batch
from collections import deque

from src.Policy import HGTPolicy


@dataclass
class PPOConfig:
    """Configuration for PPO training."""

    # PPO hyperparameters
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE parameter
    clip_epsilon: float = 0.2  # PPO clipping parameter
    value_loss_coef: float = 0.5  # Value loss coefficient
    entropy_coef: float = 0.01  # Entropy bonus coefficient
    max_grad_norm: float = 0.5  # Gradient clipping threshold

    # Optimization
    learning_rate: float = 3e-4
    num_epochs: int = 4  # Number of PPO epochs per update
    batch_size: int = 64  # Mini-batch size for PPO updates
    normalize_advantage: bool = True

    # Value function
    value_clip: bool = True  # Whether to clip value function updates


class RolloutBuffer:
    """
    Storage for trajectories collected during environment interaction.

    Stores: states, actions, rewards, values, log_probs, dones
    """

    def __init__(self):
        self.states: List[HeteroData] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.dones: List[bool] = []

        self.returns: List[float] = []
        self.advantages: List[float] = []

    def add(
        self,
        state: HeteroData,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Add a transition to the buffer."""
        self.states.append(state.cpu())
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def compute_returns_and_advantages(self, last_value: float, gamma: float, gae_lambda: float):
        """
        Compute returns and advantages using GAE.

        Args:
            last_value: Value estimate for the final state
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        # Convert to numpy
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones)

        # Compute advantages using GAE
        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
                last_gae = 0
            else:
                next_value = values[t + 1]

            # TD error
            delta = rewards[t] + gamma * next_value - values[t]

            # GAE
            last_gae = delta + gamma * gae_lambda * last_gae
            advantages[t] = last_gae

        # Returns = advantages + values
        returns = advantages + values[:-1]

        self.returns = returns.tolist()
        self.advantages = advantages.tolist()

    def get(self) -> Dict[str, List]:
        """Get all stored data."""
        return {
            'states': self.states,
            'actions': self.actions,
            'returns': self.returns,
            'advantages': self.advantages,
            'old_log_probs': self.log_probs,
            'old_values': self.values,
        }

    def clear(self):
        """Clear the buffer."""
        self.__init__()

    def __len__(self):
        return len(self.states)


class PPOTrainer:
    """
    PPO Trainer for HGT Policy on JSSP.
    """

    def __init__(
        self,
        policy: HGTPolicy,
        config: PPOConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize PPO Trainer.

        Args:
            policy: HGTPolicy instance
            config: PPO configuration
            device: Device for training
        """
        self.policy = policy.to(device)
        self.config = config
        self.device = device

        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate)

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # Training statistics
        self.train_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': [],
            'approx_kl': [],
            'clip_fraction': [],
        }

    def collect_rollout(
        self,
        env,
        num_steps: Optional[int] = None,
        num_episodes: int = 1,
        deterministic: bool = False
    ) -> Dict[str, float]:
        """
        Collect rollout(s) from the environment.

        Args:
            env: JSSP environment
            num_steps: Maximum steps per episode (None = until done)
            num_episodes: Number of episodes to collect
            deterministic: Whether to use deterministic policy

        Returns:
            Episode statistics
        """
        self.policy.eval()

        episode_rewards = []
        episode_lengths = []
        episode_makespans = []

        for ep in range(num_episodes):
            state, info = env.reset()
            done = False
            episode_reward = 0
            steps = 0

            while not done:
                # Convert state to tensor
                state_tensor = state.to(self.device)

                # Select action
                with torch.no_grad():
                    action, log_prob, _, value = self.policy.get_action_and_value(
                        state_tensor,
                        deterministic=deterministic
                    )

                action_int = action.item()
                log_prob_val = log_prob.item()
                value_val = value.item()

                # Step environment
                next_state, reward, terminated, truncated, info = env.step(action_int)
                done = terminated or truncated

                # Store transition
                self.buffer.add(
                    state=state,
                    action=action_int,
                    reward=reward,
                    value=value_val,
                    log_prob=log_prob_val,
                    done=done
                )

                state = next_state
                episode_reward += reward
                steps += 1

                if num_steps is not None and steps >= num_steps:
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            episode_makespans.append(info.get('makespan', 0))

        # Compute returns and advantages
        if len(self.buffer) > 0:
            # Get value for last state
            with torch.no_grad():
                last_state = state.to(self.device)
                last_value = self.policy.get_value(last_state).item()

            self.buffer.compute_returns_and_advantages(
                last_value=last_value,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda
            )

        return {
            'mean_reward': np.mean(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'mean_makespan': np.mean(episode_makespans),
            'min_makespan': np.min(episode_makespans),
        }

    def update(self) -> Dict[str, float]:
        """
        Perform PPO update using collected rollouts.

        Returns:
            Training statistics
        """
        if len(self.buffer) == 0:
            return {}

        self.policy.train()

        # Get data from buffer
        data = self.buffer.get()
        states = data['states']
        actions = torch.tensor(data['actions'], dtype=torch.long, device=self.device)
        returns = torch.tensor(data['returns'], dtype=torch.float32, device=self.device)
        advantages = torch.tensor(data['advantages'], dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(data['old_log_probs'], dtype=torch.float32, device=self.device)
        old_values = torch.tensor(data['old_values'], dtype=torch.float32, device=self.device)

        # Normalize advantages
        if self.config.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Training metrics
        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_entropies = []
        epoch_total_losses = []
        epoch_approx_kls = []
        epoch_clip_fractions = []

        # PPO epochs
        for epoch in range(self.config.num_epochs):
            # Mini-batch training
            indices = np.arange(len(states))
            np.random.shuffle(indices)

            for start in range(0, len(states), self.config.batch_size):
                end = min(start + self.config.batch_size, len(states))
                batch_indices = indices[start:end]

                # Batch states (using PyG batching)
                batch_states = [states[i] for i in batch_indices]
                batch_states_tensor = Batch.from_data_list(batch_states).to(self.device)

                batch_actions = actions[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_old_values = old_values[batch_indices]

                # Evaluate actions
                log_probs, values, entropy = self.policy.evaluate_actions(
                    batch_states_tensor,
                    batch_actions
                )

                values = values.squeeze()

                # Policy loss (PPO clipped objective)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                if self.config.value_clip:
                    # Clipped value loss
                    value_pred_clipped = batch_old_values + torch.clamp(
                        values - batch_old_values,
                        -self.config.clip_epsilon,
                        self.config.clip_epsilon
                    )
                    value_loss1 = (values - batch_returns).pow(2)
                    value_loss2 = (value_pred_clipped - batch_returns).pow(2)
                    value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
                else:
                    value_loss = 0.5 * (values - batch_returns).pow(2).mean()

                # Entropy loss (encourage exploration)
                entropy_loss = -entropy.mean()

                # Total loss
                total_loss = (
                    policy_loss +
                    self.config.value_loss_coef * value_loss +
                    self.config.entropy_coef * entropy_loss
                )

                # Optimization step
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                # Logging
                epoch_policy_losses.append(policy_loss.item())
                epoch_value_losses.append(value_loss.item())
                epoch_entropies.append(-entropy_loss.item())
                epoch_total_losses.append(total_loss.item())

                # Approximate KL divergence
                with torch.no_grad():
                    approx_kl = (batch_old_log_probs - log_probs).mean().item()
                    clip_fraction = ((ratio - 1.0).abs() > self.config.clip_epsilon).float().mean().item()

                epoch_approx_kls.append(approx_kl)
                epoch_clip_fractions.append(clip_fraction)

        # Clear buffer
        self.buffer.clear()

        # Return statistics
        stats = {
            'policy_loss': np.mean(epoch_policy_losses),
            'value_loss': np.mean(epoch_value_losses),
            'entropy': np.mean(epoch_entropies),
            'total_loss': np.mean(epoch_total_losses),
            'approx_kl': np.mean(epoch_approx_kls),
            'clip_fraction': np.mean(epoch_clip_fractions),
        }

        # Store for history
        for key, val in stats.items():
            self.train_stats[key].append(val)

        return stats

    def save(self, path: str):
        """Save trainer state."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_stats': self.train_stats,
            'config': self.config,
        }, path)

    def load(self, path: str):
        """Load trainer state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_stats = checkpoint['train_stats']
        self.config = checkpoint['config']
