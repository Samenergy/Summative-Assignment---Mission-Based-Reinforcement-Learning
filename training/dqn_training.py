import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from environment.custom_env import B2BNewsSelectionEnv
from environment.rendering import render_dashboard, close_rendering
import numpy as np

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
            current_mean = np.mean(self.rewards[-100:])  # Last 100 episodes
            self.mean_rewards.append(current_mean)
            
            # Track best performance
            if current_mean > self.best_mean_reward:
                self.best_mean_reward = current_mean
            
            print(f"Episode {len(self.rewards)}: Mean Reward (last 100) = {current_mean:.3f}, Best = {self.best_mean_reward:.3f}")
        return True

env = B2BNewsSelectionEnv()

# Improved DQN hyperparameters
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.0001,          # Reduced for more stable learning
    buffer_size=50000,             # Smaller buffer for faster updates
    exploration_fraction=0.2,      # Faster exploration decay
    exploration_final_eps=0.01,    # Lower final epsilon for better exploitation
    batch_size=64,                 # Smaller batch for more frequent updates
    train_freq=4,                  # Train every 4 steps
    gradient_steps=1,              # Single gradient step per update
    target_update_interval=1000,   # Update target network every 1000 steps
    learning_starts=1000,          # Start learning after 1000 steps
    gamma=0.99,                    # Discount factor
    tau=1.0,                       # Target network update rate
    verbose=1
)

print("🚀 Training DQN with optimized hyperparameters...")
print("📊 Hyperparameters:")
print(f"   Learning Rate: {model.learning_rate}")
print(f"   Buffer Size: {model.buffer_size}")
print(f"   Batch Size: {model.batch_size}")
print(f"   Exploration: {model.exploration_fraction} -> {model.exploration_final_eps}")
print(f"   Train Frequency: Every {model.train_freq} steps")
print("=" * 60)

model.learn(total_timesteps=150000, callback=MeanRewardCallback())
model.save("models/dqn/dqn_b2b")

print("✅ DQN training complete! Testing model...")

# Test trained model
obs, _ = env.reset()
rewards = []
episode = 1
step = 0
max_steps = env.max_articles

for _ in range(20):
    action, _ = model.predict(obs, deterministic=True)  # Use deterministic actions for testing
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