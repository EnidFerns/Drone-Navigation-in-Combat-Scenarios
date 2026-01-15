# using stablebaselines algorithms and gyms Pendulum-v1 environment


import os
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from stable_baselines3 import DDPG, TD3, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback


TOTAL_TIMESTEPS = 100000
EVAL_INTERVAL = 1000
LOG_DIR = "./pendulum_comparison"
os.makedirs(LOG_DIR, exist_ok=True)

ALGOS = {
    "DDPG": DDPG,
    "TD3": TD3,
    "SAC": SAC
}

# Callback - Logging
class ScoreLoggerCallback(BaseCallback):
    def __init__(self, algo_name, eval_env, log_path, eval_interval=1000):
        super().__init__()
        self.eval_env = eval_env
        self.eval_interval = eval_interval
        self.episode_rewards = []
        self.eval_rewards = []
        self.log_path = log_path
        self.algo_name = algo_name

    def _on_step(self):
        if self.n_calls % self.eval_interval == 0:
            reward_sum = 0
            for _ in range(5): 
                # evaluating 5 episodes
                done = False
                obs = self.eval_env.reset()
                episode_reward = 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, _ = self.eval_env.step(action)
                    episode_reward += reward
                reward_sum += episode_reward
            avg_reward = reward_sum / 5.0
            self.eval_rewards.append(avg_reward)
        return True

    def _on_training_end(self):
        torch.save({
            "eval_rewards": self.eval_rewards
        }, os.path.join(self.log_path, f"{self.algo_name}_scores.pth"))

# Training
def train_agent(algo_name, model_class):
    print(f"\n Training {algo_name}...")
    model_path = os.path.join(LOG_DIR, algo_name)
    os.makedirs(model_path, exist_ok=True)

    train_env = DummyVecEnv([lambda: Monitor(gym.make("Pendulum-v1"))])
    eval_env = gym.make("Pendulum-v1")

    model = model_class(
        "MlpPolicy",
        train_env,
        verbose=0,
        learning_rate=1e-3,
        buffer_size=100000,
        batch_size=64,
        gamma=0.99,
        tau=0.005,
        train_freq=(1, "step"),
    )

    callback = ScoreLoggerCallback(algo_name, eval_env, LOG_DIR, eval_interval=EVAL_INTERVAL)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    model.save(os.path.join(model_path, f"{algo_name}_pendulum"))

# Plotting
def plot_results():
    plt.figure(figsize=(12, 8))
    for algo in ALGOS:
        path = os.path.join(LOG_DIR, f"{algo}_scores.pth")
        if os.path.exists(path):
            data = torch.load(path)
            eval_rewards = data["eval_rewards"]
            plt.plot(
                np.arange(len(eval_rewards)) * EVAL_INTERVAL,
                eval_rewards,
                label=algo
            )
    plt.title("Learning Curve Comparison (Pendulum-v1)")
    plt.xlabel("Timestep")
    plt.ylabel("Average Evaluation Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(LOG_DIR, "comparison_plot.png"))
    plt.show()

def analyze_results():
    results = []

    print("\nAgent Performance Summary:\n")
    print(f"{'Algo':<8} {'Final Score':>12} {'Best Score':>12} {'@50k':>10} {'@100k':>10}")

    for algo in ALGOS:
        path = os.path.join(LOG_DIR, f"{algo}_scores.pth")
        if os.path.exists(path):
            data = torch.load(path)
            scores = data["eval_rewards"]

            final_score = scores[-1]
            best_score = max(scores)
            at_50k = scores[len(scores) // 2] if len(scores) > 1 else float('nan')
            at_100k = scores[-1] if len(scores) > 1 else float('nan')

            print(f"{algo:<8} {final_score:12.2f} {best_score:12.2f} {at_50k:10.2f} {at_100k:10.2f}")
            results.append({
                "Algorithm": algo,
                "Final Score": final_score,
                "Best Score": best_score,
                "Score @50k": at_50k,
                "Score @100k": at_100k
            })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(LOG_DIR, "performance_summary.csv"), index=False)
    print(f"\nSummary saved to: {LOG_DIR}/performance_summary.csv")

# Main Script
if __name__ == "__main__":
    for algo_name, model_class in ALGOS.items():
        train_agent(algo_name, model_class)

    plot_results()
    analyze_results()
