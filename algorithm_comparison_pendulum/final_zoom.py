import torch
import matplotlib.pyplot as plt
import numpy as np
import os

LOG_DIR = "./pendulum_comparison"
EVAL_INTERVAL = 1000
ZOOM_START = 40000  
# Zoom in from timestep 40k onward

algos = ["DDPG", "TD3", "SAC"]
colors = ["blue", "orange", "green"]

plt.figure(figsize=(10, 6))
for algo, color in zip(algos, colors):
    path = os.path.join(LOG_DIR, f"{algo}_scores.pth")
    if os.path.exists(path):
        data = torch.load(path)
        eval_rewards = data["eval_rewards"]
        timesteps = np.arange(len(eval_rewards)) * EVAL_INTERVAL

        # Zoom-in
        mask = timesteps >= ZOOM_START
        plt.plot(
            timesteps[mask],
            np.array(eval_rewards)[mask],
            label=algo,
            color=color
        )

plt.title("Zoomed Learning Curve (Final Phase)")
plt.xlabel("Timestep")
plt.ylabel("Average Evaluation Reward")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, "zoomed_final_phase_plot.png"))
plt.show()
