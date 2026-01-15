import torch
import numpy as np
from Env import DroneEnv
from sac_agent import SACAgent
import time
import pybullet as p
import os
import argparse


def test_drone(agent=None, random_actions=False, num_episodes=3, max_steps=1000, render=True, render_delay=0.01):
    """
    Test drone performance

    Parameters:
        agent: Trained agent, if None will use random actions
        random_actions: Whether to use pure random actions (for untrained state)
        num_episodes: Number of test episodes
        max_steps: Maximum steps per episode
        render: Whether to render the environment
        render_delay: Rendering delay time

    Returns:
        Statistics dictionary
    """
    # Create environment
    env = DroneEnv(render=render, num_obstacles=30, enemy_drones=2)

    # Statistics
    stats = {
        "episode_rewards": [],
        "episode_lengths": [],
        "termination_reasons": {
            "max_steps": 0,
            "out_of_bounds": 0,
            "zero_lifepoints": 0,
            "enemies_defeated": 0,
            "other": 0
        },
        "lifepoints_remaining": [],
        "obstacles_hit": 0,
        "enemy_hits": 0,
        "got_hit": 0
    }

    for i in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        step = 0
        done = False
        obstacles_hit = 0
        enemy_hits = 0
        got_hit = 0

        print(f"\nStarting test episode {i + 1}/{num_episodes}")

        while not done and step < max_steps:
            # Select action
            if random_actions:
                # Completely random action
                action = env.action_space.sample()
            elif agent is None:
                # Simple heuristic strategy: maintain flight height, move randomly
                action = np.zeros(env.action_space.shape[0])
                action[0] = 0.5  # Throttle maintained at medium
                action[1:4] = np.random.uniform(-0.2, 0.2, 3)  # Small attitude changes
                action[4] = 0 if np.random.random() < 0.8 else 1  # Occasionally fire
            else:
                # Use trained agent
                action = agent.act(state, evaluate=True)

            # Execute action
            next_state, reward, done, info = env.step(action)

            # Get drone position
            drone_pos, _ = p.getBasePositionAndOrientation(env.drone_id)

            # Render environment
            if render:
                env.render()
                time.sleep(render_delay)

            # Update statistics
            episode_reward += reward
            step += 1

            # Check collisions
            for contact in p.getContactPoints():
                if contact[1] == env.drone_id or contact[2] == env.drone_id:
                    # Collision with obstacles
                    if contact[1] in env.obstacle_ids or contact[2] in env.obstacle_ids:
                        obstacles_hit += 1
                    # Hit enemy
                    if contact[1] in env.enemy_drone_ids or contact[2] in env.enemy_drone_ids:
                        enemy_hits += 1

            # Check if got hit
            if info.get("drone_lifepoints", 100) < stats.get("last_lifepoints", 100):
                got_hit += 1
                stats["last_lifepoints"] = info.get("drone_lifepoints", 100)

            # Output some status information
            if step % 100 == 0:
                print(f"  Steps: {step}, Current cumulative reward: {episode_reward:.2f}")
                print(f"  Drone position: x={drone_pos[0]:.2f}, y={drone_pos[1]:.2f}, z={drone_pos[2]:.2f}")
                if "drone_lifepoints" in info:
                    print(f"  Drone life points: {info['drone_lifepoints']}")

            # Update state
            state = next_state

        # Determine termination reason
        termination_reason = "Other reasons"
        if step >= max_steps:
            termination_reason = "Maximum steps reached"
            stats["termination_reasons"]["max_steps"] += 1
        elif abs(drone_pos[0]) > 25 or abs(drone_pos[1]) > 25 or drone_pos[2] < -5 or drone_pos[2] > 15:
            termination_reason = f"Out of bounds (x={drone_pos[0]:.2f}, y={drone_pos[1]:.2f}, z={drone_pos[2]:.2f})"
            stats["termination_reasons"]["out_of_bounds"] += 1
        elif info.get("drone_lifepoints", 0) <= 0:
            termination_reason = "Drone life points reduced to zero"
            stats["termination_reasons"]["zero_lifepoints"] += 1
        elif len(env.enemy_drone_ids) == 0:
            termination_reason = "All enemies have been eliminated"
            stats["termination_reasons"]["enemies_defeated"] += 1
        else:
            stats["termination_reasons"]["other"] += 1

        # Update statistics
        stats["episode_rewards"].append(episode_reward)
        stats["episode_lengths"].append(step)
        stats["lifepoints_remaining"].append(info.get("drone_lifepoints", 0))
        stats["obstacles_hit"] += obstacles_hit
        stats["enemy_hits"] += enemy_hits
        stats["got_hit"] += got_hit

        print(f"\nEpisode termination reason: {termination_reason}")
        print(f"Episode {i + 1} completed: Total reward = {episode_reward:.2f}, Total steps = {step}")
        print(f"Number of obstacle collisions: {obstacles_hit}, Number of enemy hits: {enemy_hits}, Number of times got hit: {got_hit}")
        print(f"Remaining life points: {info.get('drone_lifepoints', 0)}")

    # Close environment
    env.close()

    # Calculate average statistics
    stats["avg_reward"] = np.mean(stats["episode_rewards"])
    stats["avg_length"] = np.mean(stats["episode_lengths"])
    stats["avg_lifepoints"] = np.mean(stats["lifepoints_remaining"])

    # Output overall statistics
    print("\n===== Test Statistics =====")
    print(f"Average reward: {stats['avg_reward']:.2f}")
    print(f"Average episode length: {stats['avg_length']:.2f}")
    print(f"Average remaining life points: {stats['avg_lifepoints']:.2f}")
    print(f"Total number of obstacle collisions: {stats['obstacles_hit']}")
    print(f"Total number of enemy hits: {stats['enemy_hits']}")
    print(f"Total number of times got hit: {stats['got_hit']}")
    print("Termination reason statistics:")
    for reason, count in stats["termination_reasons"].items():
        print(f"  {reason}: {count}")

    return stats


def compare_trained_vs_untrained(trained_model_path, num_episodes=3, max_steps=1000):
    """Compare performance before and after training"""
    print("\n========== Testing Untrained Drone ==========")
    untrained_stats = test_drone(
        agent=None,
        random_actions=True,
        num_episodes=num_episodes,
        max_steps=max_steps
    )

    print("\n========== Testing Trained Drone ==========")
    # Create and load trained agent
    env = DroneEnv(render=False, num_obstacles=30, enemy_drones=2)  # Temporarily create environment to get state and action space dimensions
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    env.close()

    agent = SACAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_size=256
    )

    agent.load(trained_model_path)
    print(f"Loaded trained model: {trained_model_path}")

    trained_stats = test_drone(
        agent=agent,
        random_actions=False,
        num_episodes=num_episodes,
        max_steps=max_steps
    )

    # performance comparison
    print("\n========== Performance Comparison ==========")
    print(f"Average reward: Untrained = {untrained_stats['avg_reward']:.2f}, Trained = {trained_stats['avg_reward']:.2f}")
    print(f"Average episode length: Untrained = {untrained_stats['avg_length']:.2f}, Trained = {trained_stats['avg_length']:.2f}")
    print(f"Average remaining life points: Untrained = {untrained_stats['avg_lifepoints']:.2f}, Trained = {trained_stats['avg_lifepoints']:.2f}")
    print(f"Number of obstacle collisions: Untrained = {untrained_stats['obstacles_hit']}, Trained = {trained_stats['obstacles_hit']}")
    print(f"Number of enemy hits: Untrained = {untrained_stats['enemy_hits']}, Trained = {trained_stats['enemy_hits']}")
    print(f"Number of times got hit: Untrained = {untrained_stats['got_hit']}, Trained = {trained_stats['got_hit']}")

    return untrained_stats, trained_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test drone performance')
    parser.add_argument('--model', type=str, default="models/20250330_112843/sac_agent_episode_9000.pth",
                        help='Path to trained model')
    parser.add_argument('--mode', type=str, choices=['untrained', 'trained', 'compare'], default='compare',
                        help='Test mode: untrained=test untrained only, trained=test trained only, compare=compare both')
    parser.add_argument('--episodes', type=int, default=5, help='Number of test episodes for each mode')
    parser.add_argument('--steps', type=int, default=1000, help='Maximum steps per episode')
    args = parser.parse_args()

    if args.mode == 'untrained':
        print("\n========== Testing Untrained Drone ==========")
        test_drone(agent=None, random_actions=True, num_episodes=args.episodes, max_steps=args.steps)

    elif args.mode == 'trained':
        print("\n========== Testing Trained Drone ==========")
        # Create and load trained agent
        env = DroneEnv(render=False)  # Temporarily create environment to get state and action space dimensions
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        env.close()

        agent = SACAgent(
            state_size=state_size,
            action_size=action_size,
            hidden_size=256
        )

        agent.load(args.model)
        print(f"Loaded trained model: {args.model}")

        test_drone(agent=agent, random_actions=False, num_episodes=args.episodes, max_steps=args.steps)

    elif args.mode == 'compare':
        compare_trained_vs_untrained(args.model, num_episodes=args.episodes, max_steps=args.steps)