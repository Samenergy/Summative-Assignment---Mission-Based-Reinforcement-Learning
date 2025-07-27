import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pygame
import numpy as np
import csv
import torch
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.policies import ActorCriticPolicy
from environment.custom_env import B2BNewsSelectionEnv
from environment.fallback_rendering import render_enhanced_dashboard, close_enhanced_rendering

def load_reinforce_model(env, path):
    policy = ActorCriticPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: 0.001
    )
    policy.load_state_dict(torch.load(path))
    return policy

def get_model_predictions(model, obs, model_name):
    """Get model predictions and action probabilities"""
    if model_name == "REINFORCE":
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action, value, log_prob = model.forward(obs_tensor)
        action = action.item()
        
        # Get action probabilities for REINFORCE
        with torch.no_grad():
            action_probs = torch.softmax(log_prob, dim=-1).squeeze().numpy()
            # Ensure it's a list/array with 3 values
            if action_probs.ndim == 0:  # If it's a scalar
                action_probs = np.array([0.33, 0.33, 0.34])
            elif len(action_probs) != 3:  # If it doesn't have 3 values
                action_probs = np.array([0.33, 0.33, 0.34])
    else:
        action, _ = model.predict(obs)
        
        # Get action probabilities for other models
        if hasattr(model, 'policy'):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                if hasattr(model.policy, 'get_distribution'):
                    dist = model.policy.get_distribution(obs_tensor)
                    action_probs = dist.distribution.probs.squeeze().numpy()
                    # Ensure it's a list/array with 3 values
                    if action_probs.ndim == 0 or len(action_probs) != 3:
                        action_probs = np.array([0.33, 0.33, 0.34])
                else:
                    # Fallback for models without direct probability access
                    action_probs = np.array([0.33, 0.33, 0.34])  # Equal probabilities
        else:
            action_probs = np.array([0.33, 0.33, 0.34])
    
    return action, action_probs

def play_model_enhanced(env, model, model_name, num_episodes=2, max_steps_per_episode=10):
    pygame.init()
    clock = pygame.time.Clock()
    episode_rewards = []
    all_steps = []
    
    print(f"🎮 Playing {model_name} with Enhanced 3D Visualization")
    print(f"📊 Episodes: {num_episodes}, Max Steps: {max_steps_per_episode}")
    print("=" * 60)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        rewards = []
        steps = 0
        done = False
        
        print(f"\n🚀 Starting Episode {episode + 1}/{num_episodes}")
        
        while not done and steps < max_steps_per_episode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    close_enhanced_rendering(env)
                    print(f"\n📈 {model_name} Final Results:")
                    print(f"   Average Episode Reward: {np.mean(episode_rewards):.2f}")
                    print(f"   Total Episodes: {len(episode_rewards)}")
                    
                    # Save results
                    with open(f"{model_name}_enhanced_results.csv", "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["Episode", "Step", "Action", "Reward", "Mean Reward", "State Values", "Action Probs"])
                        writer.writerows(all_steps)
                    return
            
            # Get model predictions with probabilities
            action, action_probs = get_model_predictions(model, obs, model_name)
            
            # Take action
            obs, reward, done, _, _ = env.step(action)
            rewards.append(reward)
            steps += 1
            mean_reward = np.mean(rewards) if rewards else 0.0
            
            # Prepare episode info
            episode_info = f"{episode + 1}/{num_episodes}, Step {steps}/{max_steps_per_episode}"
            
            # Use enhanced 3D rendering with neural network visualization
            render_enhanced_dashboard(
                env, 
                action, 
                model_name, 
                episode_info, 
                obs.tolist(),  # State values
                action_probs.tolist()  # Action probabilities
            )
            
            # Console output with enhanced formatting
            action_names = ['Skip', 'Select', 'Prioritize']
            reward_emoji = "✅" if reward > 0 else "❌" if reward < 0 else "➖"
            
            print(f"   Step {steps:2d}: {action_names[action]:10s} | "
                  f"Reward: {reward:6.2f} {reward_emoji} | "
                  f"Mean: {mean_reward:6.2f} | "
                  f"Probs: [{action_probs[0]:.2f}, {action_probs[1]:.2f}, {action_probs[2]:.2f}]")
            
            # Store detailed step information
            all_steps.append([
                episode + 1, steps, action_names[action], reward, mean_reward,
                str(obs.tolist()), str(action_probs.tolist())
            ])
            
            clock.tick(1)  # 1 FPS for clearer visualization
            
            if done:
                episode_total = np.sum(rewards)
                print(f"   🎯 Episode {episode + 1} finished. Total Reward: {episode_total:.2f}")
                break
        
        episode_rewards.append(np.sum(rewards))
    
    close_enhanced_rendering(env)
    
    # Final statistics
    print(f"\n🏆 {model_name} Training Complete!")
    print("=" * 60)
    print(f"📊 Final Statistics:")
    print(f"   Total Episodes: {len(episode_rewards)}")
    print(f"   Average Episode Reward: {np.mean(episode_rewards):.2f}")
    print(f"   Best Episode Reward: {np.max(episode_rewards):.2f}")
    print(f"   Worst Episode Reward: {np.min(episode_rewards):.2f}")
    print(f"   Standard Deviation: {np.std(episode_rewards):.2f}")
    
    # Save detailed results
    with open(f"{model_name}_enhanced_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Step", "Action", "Reward", "Mean Reward", "State Values", "Action Probs"])
        writer.writerows(all_steps)
    
    print(f"💾 Results saved to {model_name}_enhanced_results.csv")

def main():
    print("🎮 Enhanced B2B News Selection - Model Playground")
    print("=" * 60)
    print("🚀 Features:")
    print("   • Enhanced 3D Visualization with shadows and gradients")
    print("   • Real-time Neural Network State Visualization")
    print("   • Particle System Effects")
    print("   • Enhanced UI with 3D Cards and Shadows")
    print("   • Detailed Action Probability Analysis")
    print("   • Compatible with all systems (no OpenGL required)")
    print("=" * 60)
    
    env = B2BNewsSelectionEnv()
    models = {
        "DQN": ("models/dqn/best_dqn", lambda path: DQN.load(path)),
        "PPO": ("models/ppo/best_ppo", lambda path: PPO.load(path)),
        "A2C": ("models/a2c/best_a2c", lambda path: A2C.load(path)),
        "REINFORCE": ("models/reinforce/best_reinforce.pt", lambda path: load_reinforce_model(env, path))
    }
    
    print("\n📋 Available Models:", ", ".join(models.keys()))
    print("\n🎯 Select Mode:")
    print("   1. Single Model (detailed analysis)")
    print("   2. All Models (comparison)")
    print("   3. Quick Demo (random actions)")
    
    mode = input("\nEnter mode (1/2/3): ").strip()
    
    if mode == "1":
        print("\n🤖 Select Model:")
        for i, model_name in enumerate(models.keys(), 1):
            print(f"   {i}. {model_name}")
        
        model_choice = input("Enter model number: ").strip()
        try:
            model_index = int(model_choice) - 1
            model_name = list(models.keys())[model_index]
            num_episodes = int(input("Enter number of episodes (default 2): ") or 2)
            
            model_path, load_fn = models[model_name]
            try:
                print(f"\n🔄 Loading {model_name} model...")
                model = load_fn(model_path)
                print(f"✅ {model_name} loaded successfully!")
                play_model_enhanced(env, model, model_name, num_episodes)
            except FileNotFoundError:
                print(f"❌ Model file {model_path} not found. Please train the model first.")
        except (ValueError, IndexError):
            print("❌ Invalid model selection.")
    
    elif mode == "2":
        num_episodes = int(input("Enter number of episodes per model (default 1): ") or 1)
        results = {}
        
        for model_name, (model_path, load_fn) in models.items():
            try:
                print(f"\n🔄 Loading {model_name}...")
                model = load_fn(model_path)
                print(f"✅ {model_name} loaded successfully!")
                play_model_enhanced(env, model, model_name, num_episodes)
                results[model_name] = "Success"
            except FileNotFoundError:
                print(f"❌ Model file {model_path} not found. Skipping {model_name}.")
                results[model_name] = "Not Found"
        
        print(f"\n📊 Summary:")
        for model_name, status in results.items():
            print(f"   {model_name}: {status}")
    
    elif mode == "3":
        print("\n🎲 Running Quick Demo with Random Actions...")
        from environment.fallback_rendering import render_enhanced_dashboard, close_enhanced_rendering
        
        pygame.init()
        clock = pygame.time.Clock()
        
        for step in range(10):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    close_enhanced_rendering(env)
                    return
            
            action = env.action_space.sample()
            obs, reward, done, _, _ = env.step(action)
            
            # Random action probabilities for demo
            action_probs = np.random.dirichlet([1, 1, 1])
            
            render_enhanced_dashboard(
                env, 
                action, 
                "Random Agent", 
                f"Demo Step {step + 1}/10",
                obs.tolist(),
                action_probs.tolist()
            )
            
            print(f"Step {step + 1}: Action={['Skip', 'Select', 'Prioritize'][action]}, "
                  f"Reward={reward:.2f}, Probs={action_probs}")
            
            clock.tick(1)
            
            if done:
                obs, _ = env.reset()
        
        close_enhanced_rendering(env)
        print("🎉 Demo complete!")
    
    else:
        print("❌ Invalid mode selected.")

if __name__ == "__main__":
    main() 