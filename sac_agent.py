import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import random
from collections import deque, namedtuple
import copy

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Create experience replay buffer
class ReplayBuffer:
    """Experience replay buffer"""

    def __init__(self, buffer_size, batch_size):
        """Initialize parameters"""
        self.memory = deque(maxlen=buffer_size)  # Experience pool
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add new experience"""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences"""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the number of experiences in the buffer"""
        return len(self.memory)


# Create actor network (policy network)
class Actor(nn.Module):
    """Actor (policy) network"""

    def __init__(self, state_size, action_size, hidden_size=256, log_std_min=-20, log_std_max=2):
        """Initialize parameters and build model"""
        super(Actor, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Shared layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Mean layer
        self.mu = nn.Linear(hidden_size, action_size)

        # Standard deviation layer
        self.log_std = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        """Forward propagation"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        # Calculate mean
        mu = self.mu(x)

        # Calculate log standard deviation and limit range
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mu, log_std

    def sample(self, state):
        """Sample actions based on state and calculate log probability"""
        mu, log_std = self.forward(state)
        std = log_std.exp()

        # Normal distribution
        normal = Normal(mu, std)

        # Reparameterization sampling
        x_t = normal.rsample()

        # Use tanh to compress action space to [-1, 1]
        action = torch.tanh(x_t)

        # Calculate log probability
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob


# Create critic network (Q network)
class Critic(nn.Module):
    """Critic (Q value) network"""

    def __init__(self, state_size, action_size, hidden_size=256):
        """Initialize parameters and build model"""
        super(Critic, self).__init__()

        # Q1
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.q1 = nn.Linear(hidden_size, 1)

        # Q2 (double Q network)
        self.fc3 = nn.Linear(state_size + action_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.q2 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        """Forward propagation"""
        x = torch.cat([state, action], dim=1)

        # Q1
        x1 = F.relu(self.fc1(x))
        x1 = F.relu(self.fc2(x1))
        q1 = self.q1(x1)

        # Q2
        x2 = F.relu(self.fc3(x))
        x2 = F.relu(self.fc4(x2))
        q2 = self.q2(x2)

        return q1, q2


# SAC agent
class SACAgent:
    """Agent implementing the Soft Actor-Critic algorithm"""

    def __init__(self, state_size, action_size, hidden_size=256, buffer_size=1000000, batch_size=256,
                 gamma=0.99, tau=0.005, lr_actor=3e-4, lr_critic=3e-4, lr_alpha=3e-4,
                 initial_alpha=0.2, target_entropy=None):
        """Initialize agent parameters"""
        self.state_size = state_size
        self.action_size = action_size

        # Hyperparameters
        self.gamma = gamma  # Discount factor
        self.tau = tau  # Soft update coefficient
        self.batch_size = batch_size

        # Initialize experience replay buffer
        self.memory = ReplayBuffer(buffer_size, batch_size)

        # Create actor network (policy network)
        self.actor = Actor(state_size, action_size, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Create critic network (Q network)
        self.critic1 = Critic(state_size, action_size, hidden_size).to(device)
        self.critic1_target = Critic(state_size, action_size, hidden_size).to(device)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr_critic)

        # Copy weights
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        # Temperature parameter alpha, controlling the balance between exploration and exploitation
        self.log_alpha = torch.tensor(np.log(initial_alpha)).to(device)
        self.log_alpha.requires_grad = True
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

        # Target entropy
        if target_entropy is None:
            self.target_entropy = -action_size  # Default to negative of action space dimension
        else:
            self.target_entropy = target_entropy

    def act(self, state, evaluate=False):
        """Select action based on current policy"""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        with torch.no_grad():
            if evaluate:
                # Deterministic policy (for testing)
                action, _ = self.actor.sample(state)
            else:
                # Stochastic policy (for training)
                mu, log_std = self.actor(state)
                std = log_std.exp()
                normal = Normal(mu, std)
                x_t = normal.rsample()
                action = torch.tanh(x_t)

        return action.cpu().numpy()[0]

    def step(self, state, action, reward, next_state, done):
        """Save experience and learn"""
        # Save experience
        self.memory.add(state, action, reward, next_state, done)

        # Start learning when enough samples are accumulated
        if len(self.memory) > self.batch_size:
            self.learn()

    def learn(self):
        """Learn from experiences, update networks"""
        # Randomly sample a batch of experiences from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample()

        # Calculate current alpha value
        alpha = self.log_alpha.exp().detach()

        # ---------- Update critic network ----------
        # Get next actions and log probabilities
        next_actions, next_log_probs = self.actor.sample(next_states)

        # Calculate target Q values
        with torch.no_grad():
            # Calculate Q values for next states
            q1_next_target, q2_next_target = self.critic1_target(next_states, next_actions)
            q_next_target = torch.min(q1_next_target, q2_next_target)

            # Calculate entropy-regularized target Q values
            q_target = rewards + self.gamma * (1 - dones) * (q_next_target - alpha * next_log_probs)

        # Calculate current Q values
        q1, q2 = self.critic1(states, actions)

        # Calculate critic loss
        critic1_loss = F.mse_loss(q1, q_target)
        critic2_loss = F.mse_loss(q2, q_target)
        critic_loss = critic1_loss + critic2_loss

        # Update critic network
        self.critic1_optimizer.zero_grad()
        critic_loss.backward()
        self.critic1_optimizer.step()

        # ---------- Update actor network ----------
        # Get actions and log probabilities
        actions_pi, log_probs_pi = self.actor.sample(states)

        # Calculate Q values
        q1_pi, q2_pi = self.critic1(states, actions_pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Calculate actor loss
        actor_loss = (alpha * log_probs_pi - q_pi).mean()

        # Update actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---------- Update temperature parameter alpha ----------
        # Calculate alpha loss
        alpha_loss = -(self.log_alpha * (log_probs_pi + self.target_entropy).detach()).mean()

        # Update alpha
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ---------- Soft update target network ----------
        self.soft_update(self.critic1, self.critic1_target)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters: θ_target = τ*θ_local + (1-τ)*θ_target"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, filepath):
        """Save model"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic1_target_state_dict': self.critic1_target.state_dict(),
            'log_alpha': self.log_alpha,
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """Load model"""
        if torch.cuda.is_available():
            checkpoint = torch.load(filepath)
        else:
            checkpoint = torch.load(filepath, map_location=torch.device('cpu'))

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
        self.log_alpha = checkpoint['log_alpha']
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        print(f"Model loaded from {filepath}")