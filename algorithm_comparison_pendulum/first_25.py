import torch
import matplotlib.pyplot as plt
import numpy as np
import os

LOG_DIR = "./pendulum_comparison"
EVAL_INTERVAL = 1000
NUM_EVALS = 20  
# first 20 evaluation points

algos = ["DDPG", "TD3", "SAC"]
colors = ["blue", "orange", "green"]

plt.figure(figsize=(10, 6))
for algo, color in zip(algos, colors):
    path = os.path.join(LOG_DIR, f"{algo}_scores.pth")
    if os.path.exists(path):
        data = torch.load(path)
        eval_rewards = data["eval_rewards"]
        timesteps = np.arange(len(eval_rewards)) * EVAL_INTERVAL

        # Only take the first 20 evaluations
        plt.plot(
            timesteps[:NUM_EVALS],
            np.array(eval_rewards)[:NUM_EVALS],
            label=algo,
            color=color
        )

plt.title("Early Learning Curve (First 20 Evaluation Points)")
plt.xlabel("Timestep")
plt.ylabel("Average Evaluation Reward")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, "early_phase_plot.png"))
plt.show()
