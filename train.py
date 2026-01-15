import numpy as np
import torch
import matplotlib.pyplot as plt
from Env import DroneEnv
from sac_agent import SACAgent
import time
import os
from datetime import datetime


def train(env, agent, num_episodes=1000, max_steps=1000, render=False,
          print_every=10, save_every=100, eval_every=50, render_eval=True):
    """
    Train the agent

    Parameters:
        env: Reinforcement learning environment
        agent: SAC agent
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        render: Whether to render the training process
        print_every: Frequency of printing information
        save_every: Frequency of saving the model
        eval_every: Frequency of evaluating the agent
        render_eval: Whether to render the evaluation process
    """
    # Create directory for saving models
    save_dir = f"models/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)

    # Record training information
    scores = []
    avg_scores = []
    eval_scores = []

    # Start training
    print("Starting training...")
    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        score = 0
        start_time = time.time()

        for t in range(max_steps):
            # Select action
            action = agent.act(state)

            # Execute action, get next state and reward
            next_state, reward, done, _ = env.step(action)

            # Render environment (if needed)
            if render:
                env.render()

            # Save experience and learn
            agent.step(state, action, reward, next_state, done)

            # Update state and cumulative reward
            state = next_state
            score += reward

            # If episode ends, break the loop
            if done:
                break

        # Record score
        scores.append(score)
        avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        avg_scores.append(avg_score)

        # Print training information
        if i_episode % print_every == 0:
            episode_time = time.time() - start_time
            print(
                f"Episode {i_episode}/{num_episodes} | Score: {score:.2f} | Average Score: {avg_score:.2f} | Time: {episode_time:.2f} seconds")

        # Save model
        if i_episode % save_every == 0:
            agent.save(f"{save_dir}/sac_agent_episode_{i_episode}.pth")

        # Evaluate agent
        if i_episode % eval_every == 0:
            eval_score = evaluate(env, agent, n_episodes=5, max_steps=max_steps, render=render_eval)
            eval_scores.append(eval_score)
            print(f"Evaluation Score: {eval_score:.2f}")

    # Training complete
    print("Training complete!")

    # Save final model
    agent.save(f"{save_dir}/sac_agent_final.pth")

    # Plot learning curve
    plot_scores(scores, avg_scores, eval_scores, save_dir)

    return scores, avg_scores, eval_scores


def evaluate(env, agent, n_episodes=10, max_steps=1000, render=True):
    """
    Evaluate agent performance

    Parameters:
        env: Reinforcement learning environment
        agent: SAC agent
        n_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        render: Whether to render the evaluation process

    Returns:
        Average evaluation score
    """
    scores = []

    for i in range(n_episodes):
        state = env.reset()
        score = 0

        for t in range(max_steps):
            # Use deterministic policy
            action = agent.act(state, evaluate=True)

            # Execute action, get next state and reward
            next_state, reward, done, _ = env.step(action)

            # Render environment (if needed)
            if render:
                env.render()

            # Update state and cumulative reward
            state = next_state
            score += reward

            # If episode ends, break the loop
            if done:
                break

        scores.append(score)

    return np.mean(scores)


def plot_scores(scores, avg_scores, eval_scores, save_dir):
    """
    Plot score charts

    Parameters:
        scores: List of scores for each episode
        avg_scores: List of average scores
        eval_scores: List of evaluation scores
        save_dir: Directory to save the chart
    """
    plt.figure(figsize=(12, 8))

    # Plot scores for each episode
    plt.plot(np.arange(1, len(scores) + 1), scores, label='Scores')

    # Plot moving average scores
    plt.plot(np.arange(1, len(avg_scores) + 1), avg_scores, label='Average Scores (last 100 episodes)', color='red')

    # Plot evaluation scores
    if len(eval_scores) > 0:
        eval_episodes = np.arange(1, len(scores) + 1, len(scores) // len(eval_scores))[:len(eval_scores)]
        plt.plot(eval_episodes, eval_scores, 'o-', label='Evaluation Scores', color='green')

    plt.title('Training Learning Curve')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

    # Save chart
    plt.savefig(f'{save_dir}/learning_curve.png')
    plt.close()


def record_video(env, agent, filepath, max_steps=1000):
    """
    Record agent performance video

    Parameters:
        env: Reinforcement learning environment
        agent: SAC agent
        filepath: Video save path
        max_steps: Maximum steps
    """
    state = env.reset()

    # Initialize video recorder
    env.render()  # Ensure GUI is initialized

    for t in range(max_steps):
        # Use deterministic policy
        action = agent.act(state, evaluate=True)

        # Execute action, get next state and reward
        next_state, reward, done, _ = env.step(action)

        # Render environment
        env.render()

        # Update state
        state = next_state

        # If episode ends, break the loop
        if done:
            break


if __name__ == "__main__":
    # Set random seed
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create environment
    env = DroneEnv(render=True, num_obstacles=30, enemy_drones=2)



    # Get dimensions of state and action spaces
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    print(f"State space dimensions: {state_size}")
    print(f"Action space dimensions: {action_size}")

    # Create SAC agent
    agent = SACAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_size=256,
        buffer_size=1000000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_alpha=3e-4,
        initial_alpha=0.2
    )


    # Train the agent
    scores, avg_scores, eval_scores = train(
        env=env,
        agent=agent,
        num_episodes=10000,
        max_steps=1000,
        render=False,  # Don't render during training to speed up
        print_every=10,
        save_every=100,
        eval_every=50,
        render_eval=True  # Render during evaluation
    )

    # Close environment
    env.close()

    # Create new environment for recording video
    env = DroneEnv(render=True, num_obstacles=30, enemy_drones=2)

    # Record final performance video
    print("Recording video...")
    record_video(env, agent, "final_performance.mp4")

    # Close environment
    env.close()