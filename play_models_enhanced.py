import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pygame
import numpy as np
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.policies import ActorCriticPolicy
from environment.custom_env import B2BNewsSelectionEnv
from environment.fallback_rendering import render_enhanced_dashboard, close_enhanced_rendering

class CustomReinforcePolicy(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        super(CustomReinforcePolicy, self).__init__()
        # Main network (matches the saved model structure)
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )
        
        # Value head (matches the saved model structure)
        self.value_head = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        # Get action logits
        action_logits = self.net(x)
        # Get value
        value = self.value_head(x)
        # Get log probabilities
        log_probs = F.log_softmax(action_logits, dim=-1)
        
        # Sample action
        probs = F.softmax(action_logits, dim=-1)
        action = torch.multinomial(probs, 1).item()
        
        return action, value, log_probs

def load_reinforce_model(env, path):
    # Calculate input size based on observation space
    if hasattr(env.observation_space, 'spaces'):
        # Dict observation space
        input_size = sum(space.shape[0] for space in env.observation_space.spaces.values())
    else:
        # Box observation space
        input_size = env.observation_space.shape[0]
    
    # Create custom policy with matching architecture
    policy = CustomReinforcePolicy(
        input_size=input_size,
        hidden_size=64,  # Match the saved model's hidden size
        num_actions=env.action_space.n
    )
    
    # Load the saved state dict
    policy.load_state_dict(torch.load(path))
    policy.eval()  # Set to evaluation mode
    return policy

def get_model_predictions(model, obs, model_name):
    """Get model predictions and action probabilities"""
    # Prepare observation for model input
    if model_name in ["DQN", "PPO", "A2C"]:
        # Pass the dict as-is (SB3 expects dict of arrays)
        obs_input = {k: np.array(v, dtype=np.float32) for k, v in obs.items()}
    elif model_name == "REINFORCE":
        # Flatten for custom REINFORCE
        obs_input = np.concatenate([np.array(v, dtype=np.float32).ravel() for v in obs.values()])
    else:
        obs_input = obs
    
    if model_name == "REINFORCE":
        obs_tensor = torch.tensor(obs_input, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            # Get action logits from the network
            action_logits = model.net(obs_tensor)
            action_probs = torch.softmax(action_logits, dim=-1).squeeze().numpy()
            
            # Sample action based on probabilities
            action = torch.multinomial(torch.softmax(action_logits, dim=-1), 1).item()
            
            # Ensure proper shape
            if action_probs.ndim == 0:
                action_probs = np.array([0.33, 0.33, 0.34])
            elif len(action_probs) != 3:
                action_probs = np.array([0.33, 0.33, 0.34])
    
    elif model_name == "DQN":
        # For DQN, get Q-values and convert to action probabilities
        action, _ = model.predict(obs_input, deterministic=False)
        
        # Get Q-values for probability estimation
        with torch.no_grad():
            # Convert obs to tensor format expected by DQN
            obs_tensor = {}
            for key, value in obs_input.items():
                obs_tensor[key] = torch.tensor(value, dtype=torch.float32).unsqueeze(0)
            
            # Get Q-values from the model
            q_values = model.q_net(obs_tensor)
            
            # Convert Q-values to probabilities using softmax with temperature
            temperature = 1.0
            action_probs = torch.softmax(q_values / temperature, dim=-1).squeeze().numpy()
            
            if action_probs.ndim == 0 or len(action_probs) != 3:
                action_probs = np.array([0.33, 0.33, 0.34])
    
    else:  # PPO, A2C
        action, _ = model.predict(obs_input, deterministic=False)
        
        # Get action probabilities from the policy
        with torch.no_grad():
            try:
                # Convert observation to tensor format
                obs_tensor = {}
                for key, value in obs_input.items():
                    obs_tensor[key] = torch.tensor(value, dtype=torch.float32).unsqueeze(0)
                
                # Get features from the policy network
                features = model.policy.extract_features(obs_tensor)
                
                # Get action distribution
                if hasattr(model.policy, 'action_net'):
                    action_logits = model.policy.action_net(features)
                elif hasattr(model.policy, '_get_action_dist_from_latent'):
                    action_dist = model.policy._get_action_dist_from_latent(features)
                    action_logits = action_dist.distribution.logits
                else:
                    # Fallback: try to get distribution directly
                    action_dist = model.policy.get_distribution(obs_tensor)
                    action_logits = action_dist.distribution.logits
                
                # Convert logits to probabilities
                action_probs = torch.softmax(action_logits, dim=-1).squeeze().numpy()
                
                if action_probs.ndim == 0 or len(action_probs) != 3:
                    action_probs = np.array([0.33, 0.33, 0.34])
                    
            except Exception as e:
                print(f"Warning: Could not get action probabilities for {model_name}: {e}")
                # Fallback: estimate probabilities based on action selection
                action_probs = np.array([0.1, 0.1, 0.8]) if action == 2 else np.array([0.33, 0.33, 0.34])
    
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
            # Flatten obs dict for logging/rendering
            if isinstance(obs, dict):
                obs_flat = [obs['topic_relevance'], obs['sentiment'], obs['recency'], obs['company_match'], obs['time_pressure'], obs['quality_score']]
            else:
                obs_flat = obs
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
                obs_flat,  # State values
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
                str(obs_flat), str(action_probs.tolist())
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