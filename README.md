# 3D Drone Combat Reinforcement Learning

This project implements a 3D drone simulation environment with obstacle avoidance and combat capabilities, trained using Soft Actor-Critic (SAC) reinforcement learning.

## Project Overview

The simulation features:
- A 3D physics-based drone environment using PyBullet
- Obstacle avoidance navigation
- Enemy drone combat with weapons systems
- Training using the Soft Actor-Critic (SAC) algorithm
- Performance comparison between trained and untrained agents

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- PyBullet
- Matplotlib
- Gym
- Stablebaselines (for comparing algorithms - algorithm_comparison_pendulum/compare.py)

## Installation


## Folder Structure

- `Env.py` - The drone environment implementation using PyBullet
- `sac_agent.py` - Soft Actor-Critic agent implementation
- `train.py` - Training script for the drone agent
- `comparison_test.py` - Script to compare trained vs untrained agents
- `curve.py` - Utility to generate learning curves from training data
- `lr_comparisson.py` - To plot and compare the SAC agents training over different learning rates
- `algorithm_comparisson_pendulum\compare.py` - To plot the training of agents - DDPG, TD3 and SAC - using stable baselines tool

## Running the Code

### Training a New Agent

To train a new agent from scratch:

```bash
python train.py
```

This will:
- Create a new model directory with timestamp (e.g., `models/20250330_112843/`)
- Save model checkpoints during training
- Generate a learning curve graph
- The final model will be saved as `sac_agent_final.pth`

### Testing Agent Performance

To test and compare the performance of trained vs untrained agents:

```bash
python comparison_test.py --model models/20250330_112843/sac_agent_episode_9000.pth
```

Optional arguments:
- `--mode`: Choose between `untrained`, `trained`, or `compare` (default: `compare`)
- `--episodes`: Number of test episodes (default: 5)
- `--steps`: Maximum steps per episode (default: 1000)

### Visualizing Learning Curves

To generate or regenerate the learning curve from saved training data:

```bash
python curve.py
```

Make sure to update the model directory path in the script if necessary.

## Environment Configuration

The drone environment can be configured with different parameters:

- `render`: Enable visual rendering (set to `False` for faster training)
- `num_obstacles`: Number of obstacles in the environment
- `enemy_drones`: Number of enemy drones

Example:
```python
env = DroneEnv(render=True, num_obstacles=30, enemy_drones=2)
```

## Training Parameters

The SAC agent's hyperparameters can be customized in `train.py`:

- `hidden_size`: Neural network hidden layer size
- `buffer_size`: Replay buffer capacity
- `batch_size`: Training batch size
- `gamma`: Discount factor
- `tau`: Soft update coefficient
- `lr_actor`, `lr_critic`, `lr_alpha`: Learning rates

## Model Performance

The trained agent demonstrates:
- Effective obstacle avoidance
- Stable flight control
- Strategic engagement with enemy drones
- Significantly higher rewards compared to the untrained agent

## Pre-trained Model

A pre-trained model is included in the repository at `models/20250330_112843/sac_agent_episode_9000.pth`. This model was trained for 9000 episodes and shows good performance in navigation and combat scenarios.

## Algorithm Comparison

To train the 3 agents (DDPG, TD3, SAC) and generate learning curves,
- Navigate to the folder `algorithm_comparison_pendulum`
- Run the compare.py file
The generated graph will be saved in `algorithm_comparison_pendulum\pendulum_comparison\comparison_plot.png`

There's also a way to generate a graph of the early 25 episodes(Run the first_25.py file in the same directory) and a zoomed in version of the final phases(Run the python final_zoom.py file in the same directory) of the training for better comparison
The algorithms used were referenced from stable baselines documentation on OpenAI Gym pendulum-v1 environment

## Learning Rate Comparison

To train our custom SAC agent on the pendulum-v1 environment run the `lr_comparison.py` file
The graph will be saved in the folder `sac_lr_comparison\lr_comparison_plot.png`

## Troubleshooting

- **PyBullet GUI Issues**: If you encounter rendering problems, try setting `render=False` or restart your Python session
- **CUDA Out of Memory**: Reduce batch size or model size if you encounter GPU memory issues
- **Training Instability**: Try adjusting the learning rates or entropy coefficient if training is unstable

## Extending the Project

Possible extensions:
- Training the agent controlled drone to shoot
- Adding more complex environments with different obstacle patterns
- Implementing more sophisticated enemy AI
- Adding different drone types with varying capabilities
- Implementing multi-agent training