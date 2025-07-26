import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
import torch.nn as nn
from environment.custom_env import B2BNewsSelectionEnv
from environment.rendering import render_dashboard, close_rendering

class MeanRewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []
        self.mean_rewards = []
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.locals["dones"][0]:
            episode_rewards = self.locals["rewards"]
            self.rewards.append(sum(episode_rewards))
            current_mean = np.mean(self.rewards[-100:])
            self.mean_rewards.append(current_mean)
            
            # Track best performance
            if current_mean > self.best_mean_reward:
                self.best_mean_reward = current_mean
            
            print(f"Episode {len(self.rewards)}: Mean Reward (last 100) = {current_mean:.3f}, Best = {self.best_mean_reward:.3f}")
        return True

class REINFORCE:
    def __init__(self, env, learning_rate=0.0003):
        self.env = env
        self.policy = ActorCriticPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=lambda _: learning_rate
        )
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.rewards = []
        self.mean_rewards = []
        self.best_mean_reward = -np.inf

    def train(self, num_episodes=15000):
        print(f"🚀 Training REINFORCE for {num_episodes} episodes...")
        print(f"📊 Learning Rate: {self.optimizer.param_groups[0]['lr']}")
        print("=" * 60)
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            log_probs = []
            episode_rewards = []
            
            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                action, value, log_prob = self.policy.forward(obs_tensor)
                log_probs.append(log_prob)
                obs, reward, done, _, _ = self.env.step(action.item())
                episode_rewards.append(reward)
            
            self.rewards.append(sum(episode_rewards))
            current_mean = np.mean(self.rewards[-100:])
            self.mean_rewards.append(current_mean)
            
            # Track best performance
            if current_mean > self.best_mean_reward:
                self.best_mean_reward = current_mean
            
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}: Mean Reward (last 100) = {current_mean:.3f}, Best = {self.best_mean_reward:.3f}")
            
            # Calculate returns with discounting
            returns = []
            R = 0
            gamma = 0.99  # Discount factor
            for r in episode_rewards[::-1]:
                R = r + gamma * R
                returns.insert(0, R)
            
            returns = torch.tensor(returns, dtype=torch.float32)
            # Normalize returns for stability
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # Calculate policy loss
            policy_loss = []
            for log_prob, R in zip(log_probs, returns):
                policy_loss.append(-log_prob * R)
            policy_loss = torch.stack(policy_loss).sum()
            
            # Add entropy regularization for exploration
            entropy = -torch.mean(torch.stack(log_probs))
            total_loss = policy_loss - 0.01 * entropy
            
            self.optimizer.zero_grad()
            total_loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()
    
    def save(self, path):
        torch.save(self.policy.state_dict(), path)

# Train PPO with improved hyperparameters
print("🚀 Training PPO with optimized hyperparameters...")
env = B2BNewsSelectionEnv()
ppo = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.0003,          # Optimized learning rate
    n_steps=2048,                  # Smaller batch for more frequent updates
    batch_size=64,                 # Smaller batch size
    n_epochs=10,                   # Multiple epochs per update
    gamma=0.99,                    # Discount factor
    gae_lambda=0.95,               # GAE lambda parameter
    clip_range=0.2,                # PPO clip range
    clip_range_vf=None,            # No value function clipping
    ent_coef=0.01,                 # Entropy coefficient for exploration
    vf_coef=0.5,                   # Value function coefficient
    max_grad_norm=0.5,             # Gradient clipping
    target_kl=0.01,                # Target KL divergence
    verbose=1
)

print("📊 PPO Hyperparameters:")
print(f"   Learning Rate: {ppo.learning_rate}")
print(f"   N Steps: {ppo.n_steps}")
print(f"   Batch Size: {ppo.batch_size}")
print(f"   N Epochs: {ppo.n_epochs}")
print(f"   Clip Range: {ppo.clip_range}")
print("=" * 60)

ppo.learn(total_timesteps=150000, callback=MeanRewardCallback())
ppo.save("models/pg/ppo_b2b")

# Train A2C with improved hyperparameters
print("\n🚀 Training A2C with optimized hyperparameters...")
a2c = A2C(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.0007,          # Optimized learning rate
    n_steps=5,                     # Smaller steps for more frequent updates
    gamma=0.99,                    # Discount factor
    gae_lambda=0.95,               # GAE lambda parameter
    ent_coef=0.01,                 # Entropy coefficient
    vf_coef=0.25,                  # Value function coefficient
    max_grad_norm=0.5,             # Gradient clipping
    rms_prop_eps=1e-5,             # RMSprop epsilon
    use_rms_prop=True,             # Use RMSprop optimizer
    use_sde=False,                 # No state-dependent exploration
    sde_sample_freq=-1,            # Not used
    verbose=1
)

print("📊 A2C Hyperparameters:")
print(f"   Learning Rate: {a2c.learning_rate}")
print(f"   N Steps: {a2c.n_steps}")
print(f"   Gamma: {a2c.gamma}")
print(f"   Entropy Coef: {a2c.ent_coef}")
print("=" * 60)

a2c.learn(total_timesteps=150000, callback=MeanRewardCallback())
a2c.save("models/pg/a2c_b2b")

# Train REINFORCE
print("\n🚀 Training REINFORCE...")
reinforce = REINFORCE(env, learning_rate=0.0003)
reinforce.train(num_episodes=15000)
reinforce.save("models/pg/reinforce_b2b.pt")

print("\n✅ All models trained! Testing PPO...")

# Test PPO (example)
obs, _ = env.reset()
rewards = []
episode = 1
step = 0
max_steps = env.max_articles
for _ in range(20):
    action, _ = ppo.predict(obs, deterministic=True)  # Use deterministic actions for testing
    obs, reward, done, _, _ = env.step(action)
    rewards.append(reward)
    step += 1
    # Only render if current_article_idx is valid
    if env.current_article_idx < len(env.articles):
        render_dashboard(env, action)
    print(f"Test Step: Action={['Skip', 'Select', 'Prioritize'][action]}, Reward={reward:.2f}, Mean Reward={np.mean(rewards):.2f}")
    if done:
        obs, _ = env.reset()
        episode += 1
        step = 0

close_rendering(env)
print(f"🎯 Final Test Mean Reward: {np.mean(rewards):.3f}")
print(f"🏆 Best Training Mean Reward: {MeanRewardCallback().best_mean_reward:.3f}")