import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from collections import deque
from sac_agent import SACAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


LOG_DIR = "./sac_lr_comparison"
os.makedirs(LOG_DIR, exist_ok=True)
TOTAL_EPISODES = 500
MAX_STEPS = 200
LR_LIST = [3e-4, 1e-4, 5e-5]

# compatibility wrapper
class SACWrapper:
    def __init__(self, agent):
        self.agent = agent

    def select_action(self, obs, evaluate=True):
        return self.agent.act(obs, evaluate=evaluate)

    def step(self, *args, **kwargs):
        return self.agent.step(*args, **kwargs)

    def learn(self):
        return self.agent.learn()

    def __getattr__(self, name):
        return getattr(self.agent, name)


# train
def train_sac_on_pendulum(lr_actor):
    env = gym.make("Pendulum-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    agent = SACWrapper(SACAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_size=256,
        buffer_size=100000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        lr_actor=lr_actor,
        lr_critic=lr_actor,
        lr_alpha=lr_actor,
        initial_alpha=0.2
    ))

    scores = []

    for episode in range(TOTAL_EPISODES):
        state = env.reset()
        episode_reward = 0

        for t in range(MAX_STEPS):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, float(done))
            state = next_state
            episode_reward += reward
            if done:
                break

        scores.append(episode_reward)

        if episode % 10 == 0:
            avg = np.mean(scores[-10:])
            print(f"[LR={lr_actor:.0e}] Episode {episode}/{TOTAL_EPISODES} | Last 10 avg: {avg:.2f}")

    env.close()

    torch.save(scores, os.path.join(LOG_DIR, f"scores_lr_{lr_actor:.0e}.pt"))


# Plotting
def plot_results():
    plt.figure(figsize=(10, 6))
    for lr in LR_LIST:
        scores = torch.load(os.path.join(LOG_DIR, f"scores_lr_{lr:.0e}.pt"))
        moving_avg = np.convolve(scores, np.ones(10)/10, mode='valid')
        plt.plot(moving_avg, label=f"lr={lr:.0e}")

    plt.title("SAC Performance on Pendulum-v1 with Different LR")
    plt.xlabel("Episode")
    plt.ylabel("Reward (10-episode MA)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, "lr_comparison_plot.png"))
    plt.show()


if __name__ == "__main__":
    for lr in LR_LIST:
        train_sac_on_pendulum(lr_actor=lr)

    plot_results()
