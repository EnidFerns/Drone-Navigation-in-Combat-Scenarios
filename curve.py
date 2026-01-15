import numpy as np
import torch
import matplotlib.pyplot as plt
import os

# Load training data (if saved)
model_dir = "./models/20250330_112843/"  # Replace with your model directory

# load training data
try:
    data = torch.load(os.path.join(model_dir, "training_data.pth"))
    scores = data["scores"]
    avg_scores = data["avg_scores"]
    eval_scores = data["eval_scores"]
except:
    print("Training data file not found, trying to reconstruct from checkpoints...")
    # Try to infer training progress from model filenames
    model_files = [f for f in os.listdir(model_dir) if f.startswith("sac_agent_episode_")]
    episodes = [int(f.split("_")[-1].split(".")[0]) for f in model_files]
    episodes.sort()
    print(f"Found checkpoints: {episodes}")

    # In this case, you may need to manually construct some data points
    # This is just an example, cannot recover the real training curve
    episodes = np.array(episodes)
    dummy_scores = np.random.randn(len(episodes)) * 50 + episodes / 20

    plt.figure(figsize=(12, 8))
    plt.plot(episodes, dummy_scores, 'o-', label='Checkpoint evaluation scores (estimated)')
    plt.title('Training Checkpoints (Real data unavailable)')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{model_dir}/reconstructed_learning_curve.png')
    print(f"Generated estimate chart saved to {model_dir}/reconstructed_learning_curve.png")
    exit()

plt.figure(figsize=(12, 8))

# scores for each episode
plt.plot(np.arange(1, len(scores) + 1), scores, label='Scores')

# moving average scores
plt.plot(np.arange(1, len(avg_scores) + 1), avg_scores, label='Average scores (last 100 episodes)', color='red')

# evaluation scores
if len(eval_scores) > 0:
    eval_episodes = np.arange(1, len(scores) + 1, len(scores) // len(eval_scores))[:len(eval_scores)]
    plt.plot(eval_episodes, eval_scores, 'o-', label='Evaluation scores', color='green')

plt.title('Training Learning Curve')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.legend()
plt.grid(True)

plt.savefig(f'{model_dir}/regenerated_learning_curve.png')
print(f"Chart saved to {model_dir}/regenerated_learning_curve.png")