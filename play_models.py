import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pygame
import numpy as np
import csv
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
from environment.custom_env import B2BNewsSelectionEnv
from environment.rendering import render_dashboard, close_rendering

def load_reinforce_model(env, path):
    policy = ActorCriticPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: 0.001
    )
    policy.load_state_dict(torch.load(path))
    return policy

def play_model(env, model, model_name, num_episodes=2, max_steps_per_episode=10):
    pygame.init()
    clock = pygame.time.Clock()
    episode_rewards = []
    all_steps = []
    
    print(f"Playing {model_name} for {num_episodes} episodes...")
    for episode in range(num_episodes):
        obs, _ = env.reset()
        rewards = []
        steps = 0
        done = False
        
        while not done and steps < max_steps_per_episode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    close_rendering(env)
                    print(f"{model_name} Final Episode Mean Rewards: {np.mean(episode_rewards):.2f}")
                    with open(f"{model_name}_results.csv", "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["Episode", "Step", "Action", "Reward", "Mean Reward"])
                        writer.writerows(all_steps)
                    return
            
            if model_name == "REINFORCE":
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                action, value, log_prob = model.forward(obs_tensor)
                action = action.item()
            else:
                action, _ = model.predict(obs)
            
            obs, reward, done, _, _ = env.step(action)
            rewards.append(reward)
            steps += 1
            mean_reward = np.mean(rewards) if rewards else 0.0
            
            # Use the sophisticated rendering GUI
            episode_info = f"{episode + 1}/{num_episodes}, Step {steps}/{max_steps_per_episode}"
            render_dashboard(env, action, model_name, episode_info)
            
            print(f"{model_name} Episode {episode + 1}, Step {steps}: Action={['Skip', 'Select', 'Prioritize'][action]}, Reward={reward:.2f}, Episode Mean Reward={mean_reward:.2f}, Done={done}")
            all_steps.append([episode + 1, steps, ['Skip', 'Select', 'Prioritize'][action], reward, mean_reward])
            clock.tick(1)  # 1 FPS for clearer visualization
            
            if done:
                print(f"{model_name} Episode {episode + 1} finished.")
                break
        
        episode_rewards.append(np.sum(rewards))
    
    close_rendering(env)
    print(f"{model_name} Final Episode Mean Rewards: {np.mean(episode_rewards):.2f}")
    with open(f"{model_name}_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Step", "Action", "Reward", "Mean Reward"])
        writer.writerows(all_steps)

def main():
    env = B2BNewsSelectionEnv()
    models = {
        "DQN": ("models/dqn/dqn_b2b.zip", lambda path: DQN.load(path)),
        "PPO": ("models/pg/ppo_b2b.zip", lambda path: PPO.load(path)),
        "A2C": ("models/pg/a2c_b2b.zip", lambda path: A2C.load(path)),
        "REINFORCE": ("models/pg/reinforce_b2b.pt", lambda path: load_reinforce_model(env, path))
    }
    
    print("Available models:", ", ".join(models.keys()))
    mode = input("Enter mode (single/all): ").strip().lower()
    num_episodes = int(input("Enter number of episodes (default 2): ") or 2)
    
    if mode == "single":
        model_name = input("Enter model to play (DQN, PPO, A2C, REINFORCE): ").strip().upper()
        if model_name in models:
            model_path, load_fn = models[model_name]
            try:
                model = load_fn(model_path)
                play_model(env, model, model_name, num_episodes)
            except FileNotFoundError:
                print(f"Model file {model_path} not found. Please train the model first.")
        else:
            print("Invalid model name.")
    elif mode == "all":
        for model_name, (model_path, load_fn) in models.items():
            try:
                model = load_fn(model_path)
                play_model(env, model, model_name, num_episodes)
            except FileNotFoundError:
                print(f"Model file {model_path} not found. Skipping {model_name}.")
    else:
        print("Invalid mode. Use 'single' or 'all'.")

if __name__ == "__main__":
    main()