import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
import torch.nn as nn
from environment.custom_env import B2BNewsSelectionEnv
import json
from datetime import datetime

class EnhancedCallback(BaseCallback):
    """Enhanced callback with detailed monitoring and adaptive learning"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []
        self.mean_rewards = []
        self.best_mean_reward = -np.inf
        self.episode_lengths = []
        self.efficiency_scores = []
        self.high_value_selections = []
        self.training_stats = {
            'learning_rate': [],
            'exploration_rate': [],
            'policy_loss': [],
            'value_loss': []
        }
        
    def _on_step(self) -> bool:
        if self.locals["dones"][0]:
            episode_rewards = self.locals["rewards"]
            episode_length = len(episode_rewards)
            
            self.rewards.append(sum(episode_rewards))
            self.episode_lengths.append(episode_length)
            
            # Calculate rolling statistics
            current_mean = np.mean(self.rewards[-100:])
            self.mean_rewards.append(current_mean)
            
            # Track best performance
            if current_mean > self.best_mean_reward:
                self.best_mean_reward = current_mean
            
            # Extract additional metrics from info
            if 'infos' in self.locals and self.locals['infos']:
                info = self.locals['infos'][0]
                self.efficiency_scores.append(info.get('efficiency_score', 0))
                self.high_value_selections.append(info.get('high_value_selections', 0))
            
            # Enhanced logging
            if len(self.rewards) % 50 == 0:
                print(f"Episode {len(self.rewards)}:")
                print(f"  Mean Reward (last 100): {current_mean:.3f}")
                print(f"  Best Mean Reward: {self.best_mean_reward:.3f}")
                print(f"  Avg Episode Length: {np.mean(self.episode_lengths[-100:]):.1f}")
                if self.efficiency_scores:
                    print(f"  Avg Efficiency: {np.mean(self.efficiency_scores[-100:]):.3f}")
                print("=" * 50)
        
        return True

class REINFORCE:
    """Enhanced REINFORCE implementation with curriculum learning support"""
    
    def __init__(self, env, learning_rate=0.0003, gamma=0.99, entropy_coef=0.01, num_episodes=15000):
        self.env = env
        self.policy = ActorCriticPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=lambda _: learning_rate
        )
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.num_episodes = num_episodes
        self.rewards = []
        self.mean_rewards = []
        self.best_mean_reward = -np.inf
        self.episode_lengths = []
        self.efficiency_scores = []
        self.high_value_selections = []

    def train(self, num_episodes=None):
        if num_episodes is None:
            num_episodes = self.num_episodes
            
        print(f"🚀 Training REINFORCE for {num_episodes} episodes...")
        print(f"📊 Learning Rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"📊 Gamma: {self.gamma}")
        print(f"📊 Entropy Coefficient: {self.entropy_coef}")
        print("=" * 60)
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            done = False
            log_probs = []
            episode_rewards = []
            
            while not done:
                # Handle observation format from DummyVecEnv
                if isinstance(obs, (list, tuple)) and len(obs) > 0:
                    obs_array = obs[0] if isinstance(obs[0], np.ndarray) else np.array(obs[0])
                else:
                    obs_array = np.array(obs) if obs is not None else np.zeros(6)
                
                obs_tensor = torch.tensor(obs_array, dtype=torch.float32).unsqueeze(0)
                action, value, log_prob = self.policy.forward(obs_tensor)
                log_probs.append(log_prob)
                # DummyVecEnv expects a list of actions
                result = self.env.step([action.item()])
                if len(result) == 5:
                    obs, reward, terminated, truncated, info = result
                    done = terminated[0] or truncated[0]
                    episode_rewards.append(reward[0])
                else:
                    # Handle older gym format
                    obs, reward, done, info = result
                    episode_rewards.append(reward[0])
            
            # Track episode statistics
            episode_reward = sum(episode_rewards)
            episode_length = len(episode_rewards)
            
            self.rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Calculate rolling statistics
            current_mean = np.mean(self.rewards[-100:])
            self.mean_rewards.append(current_mean)
            
            # Track best performance
            if current_mean > self.best_mean_reward:
                self.best_mean_reward = current_mean
            
            # Extract additional metrics from info
            if hasattr(self.env, 'episode_stats'):
                self.efficiency_scores.append(self.env.episode_stats.get('efficiency_score', 0))
                self.high_value_selections.append(self.env.episode_stats.get('high_value_selections', 0))
            
            # Calculate returns with discounting
            returns = []
            R = 0
            for r in episode_rewards[::-1]:
                R = r + self.gamma * R
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
            total_loss = policy_loss - self.entropy_coef * entropy
            
            self.optimizer.zero_grad()
            total_loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            # Enhanced logging
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}:")
                print(f"  Mean Reward (last 100): {current_mean:.3f}")
                print(f"  Best Mean Reward: {self.best_mean_reward:.3f}")
                print(f"  Avg Episode Length: {np.mean(self.episode_lengths[-100:]):.1f}")
                if self.efficiency_scores:
                    print(f"  Avg Efficiency: {np.mean(self.efficiency_scores[-100:]):.3f}")
                print("=" * 50)
    
    def save(self, path):
        torch.save(self.policy.state_dict(), path)
    
    def load(self, path):
        self.policy.load_state_dict(torch.load(path))
    
    def predict(self, obs, deterministic=True):
        """Predict action for given observation"""
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action, _, _ = self.policy.forward(obs_tensor)
        return action.item(), None

class CurriculumLearning:
    """Curriculum learning to gradually increase difficulty"""
    
    def __init__(self, base_env_class, max_articles=20):
        self.base_env_class = base_env_class
        self.max_articles = max_articles
        self.current_difficulty = 0.3
        self.difficulty_increase = 0.1
        
    def create_env(self, difficulty=None):
        """Create environment with current difficulty"""
        if difficulty is None:
            difficulty = self.current_difficulty
        
        # Adjust environment parameters based on difficulty
        articles = int(self.max_articles * (0.5 + difficulty * 0.5))  # 10-20 articles
        
        def make_env():
            env = self.base_env_class(max_articles=articles)
            env = Monitor(env)
            return env
        
        return DummyVecEnv([make_env])
    
    def increase_difficulty(self):
        """Increase difficulty for next training phase"""
        self.current_difficulty = min(1.0, self.current_difficulty + self.difficulty_increase)
        print(f"🎯 Increasing difficulty to {self.current_difficulty:.2f}")
        return self.current_difficulty

class AdaptiveHyperparameters:
    """Adaptive hyperparameter adjustment based on performance"""
    
    @staticmethod
    def get_ppo_params(performance_ratio):
        """Get PPO parameters based on performance"""
        base_lr = 0.0003
        
        if performance_ratio < 0.3:
            lr = base_lr * 1.5
            n_steps = 1024  # Smaller batches for faster updates
        elif performance_ratio > 0.8:
            lr = base_lr * 0.7
            n_steps = 4096  # Larger batches for stability
        else:
            lr = base_lr
            n_steps = 2048
        
        return {
            'learning_rate': lr,
            'n_steps': n_steps,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'target_kl': 0.01
        }
    
    @staticmethod
    def get_a2c_params(performance_ratio):
        """Get A2C parameters based on performance"""
        base_lr = 0.0007
        
        if performance_ratio < 0.3:
            lr = base_lr * 1.3
            n_steps = 3  # More frequent updates
        elif performance_ratio > 0.8:
            lr = base_lr * 0.8
            n_steps = 8  # Less frequent updates
        else:
            lr = base_lr
            n_steps = 5
        
        return {
            'learning_rate': lr,
            'n_steps': n_steps,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'ent_coef': 0.01,
            'vf_coef': 0.25,
            'max_grad_norm': 0.5,
            'rms_prop_eps': 1e-5,
            'use_rms_prop': True
        }
    
    @staticmethod
    def get_reinforce_params(performance_ratio):
        """Get REINFORCE parameters based on performance"""
        base_lr = 0.0003
        
        if performance_ratio < 0.3:
            lr = base_lr * 1.5  # Increase learning rate for poor performance
            num_episodes = 20000  # More episodes for learning
        elif performance_ratio > 0.8:
            lr = base_lr * 0.7  # Decrease learning rate for good performance
            num_episodes = 10000  # Fewer episodes for fine-tuning
        else:
            lr = base_lr
            num_episodes = 15000
        
        return {
            'learning_rate': lr,
            'num_episodes': num_episodes,
            'gamma': 0.99,
            'entropy_coef': 0.01
        }

def train_policy_gradient_with_curriculum(model_type="PPO", total_timesteps=300000):
    """Train policy gradient model with curriculum learning and adaptive hyperparameters"""
    
    print(f"🚀 Starting Enhanced {model_type} Training")
    print("=" * 60)
    
    # Initialize curriculum learning
    curriculum = CurriculumLearning(B2BNewsSelectionEnv)
    
    # Training phases
    phases = [
        {"timesteps": total_timesteps // 3, "difficulty": 0.3},
        {"timesteps": total_timesteps // 3, "difficulty": 0.6},
        {"timesteps": total_timesteps // 3, "difficulty": 1.0}
    ]
    
    best_model = None
    best_performance = -np.inf
    
    for phase_idx, phase in enumerate(phases):
        print(f"\n📚 Phase {phase_idx + 1}/3 - Difficulty: {phase['difficulty']:.1f}")
        print(f"⏱️  Training for {phase['timesteps']} timesteps")
        
        # Create environment for this phase
        env = curriculum.create_env(phase['difficulty'])
        
        # Initialize model
        if phase_idx == 0:
            # First phase - create new model
            if model_type == "PPO":
                params = AdaptiveHyperparameters.get_ppo_params(0.5)
                model = PPO("MlpPolicy", env, verbose=1, **params)
            elif model_type == "A2C":
                params = AdaptiveHyperparameters.get_a2c_params(0.5)
                model = A2C("MlpPolicy", env, verbose=1, **params)
            elif model_type == "REINFORCE":
                params = AdaptiveHyperparameters.get_reinforce_params(0.5)
                model = REINFORCE(env, **params)
        else:
            # Continue training existing model
            if model_type != "REINFORCE":
                model.set_env(env)
            else:
                model.env = env
        
        # Train based on model type
        if model_type == "REINFORCE":
            # REINFORCE training
            model.train()
            
            # Create a simple callback for REINFORCE
            callback = EnhancedCallback()
            callback.rewards = model.rewards
            callback.mean_rewards = model.mean_rewards
            callback.best_mean_reward = model.best_mean_reward
            callback.episode_lengths = model.episode_lengths
            callback.efficiency_scores = model.efficiency_scores
            callback.high_value_selections = model.high_value_selections
        else:
            # Stable-Baselines3 models
            callback = EnhancedCallback()
            checkpoint_callback = CheckpointCallback(
                save_freq=10000,
                save_path=f"models/{model_type.lower()}/",
                name_prefix=f"{model_type.lower()}_phase_{phase_idx + 1}"
            )
            
            # Train
            model.learn(
                total_timesteps=phase['timesteps'],
                callback=[callback, checkpoint_callback],
                progress_bar=True
            )
        
        # Evaluate performance
        performance = np.mean(callback.mean_rewards[-50:]) if callback.mean_rewards else 0
        print(f"📊 Phase {phase_idx + 1} Performance: {performance:.3f}")
        
        # Save best model
        if performance > best_performance:
            best_performance = performance
            best_model = model
            if model_type == "REINFORCE":
                model.save(f"models/reinforce/best_reinforce.pt")
            elif model_type == "PPO":
                model.save(f"models/ppo/best_ppo")
            elif model_type == "A2C":
                model.save(f"models/a2c/best_a2c")
            else:
                model.save(f"models/{model_type.lower()}/best_{model_type.lower()}")
        
        # Save training statistics
        stats = {
            'phase': phase_idx + 1,
            'difficulty': phase['difficulty'],
            'performance': float(performance),
            'mean_rewards': [float(x) for x in callback.mean_rewards],
            'episode_lengths': [int(x) for x in callback.episode_lengths],
            'efficiency_scores': [float(x) for x in callback.efficiency_scores],
            'high_value_selections': [int(x) for x in callback.high_value_selections]
        }
        
        if model_type == "REINFORCE":
            stats_path = f"models/reinforce/phase_{phase_idx + 1}_stats.json"
        elif model_type == "PPO":
            stats_path = f"models/ppo/phase_{phase_idx + 1}_stats.json"
        elif model_type == "A2C":
            stats_path = f"models/a2c/phase_{phase_idx + 1}_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
    
    print(f"\n✅ Enhanced {model_type} Training Complete!")
    print(f"🏆 Best Performance: {best_performance:.3f}")
    print(f"💾 Best model saved to: models/{model_type.lower()}/best_{model_type.lower()}")
    
    return best_model, callback

def create_training_report(model_type, callback):
    """Create comprehensive training report"""
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{model_type} Training Report', fontsize=16)
    
    # Reward progression
    axes[0, 0].plot(callback.mean_rewards)
    axes[0, 0].set_title('Mean Reward Progression')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Mean Reward (last 100)')
    axes[0, 0].grid(True)
    
    # Episode lengths
    axes[0, 1].plot(callback.episode_lengths)
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].grid(True)
    
    # Efficiency scores
    if callback.efficiency_scores:
        axes[1, 0].plot(callback.efficiency_scores)
        axes[1, 0].set_title('Efficiency Scores')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Efficiency')
        axes[1, 0].grid(True)
    
    # High value selections
    if callback.high_value_selections:
        axes[1, 1].plot(callback.high_value_selections)
        axes[1, 1].set_title('High Value Selections')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    if model_type == "REINFORCE":
        report_path = f"models/reinforce/training_report.png"
        summary_path = f"models/reinforce/training_summary.json"
    elif model_type == "PPO":
        report_path = f"models/ppo/training_report.png"
        summary_path = f"models/ppo/training_summary.json"
    elif model_type == "A2C":
        report_path = f"models/a2c/training_report.png"
        summary_path = f"models/a2c/training_summary.json"
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save summary statistics
    summary = {
        'model_type': model_type,
        'total_episodes': len(callback.rewards),
        'best_mean_reward': float(callback.best_mean_reward),
        'final_mean_reward': float(callback.mean_rewards[-1]) if callback.mean_rewards else 0.0,
        'avg_episode_length': float(np.mean(callback.episode_lengths)),
        'avg_efficiency': float(np.mean(callback.efficiency_scores)) if callback.efficiency_scores else 0.0,
        'total_high_value_selections': int(sum(callback.high_value_selections)),
        'training_date': datetime.now().isoformat()
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"📊 Training report saved to: {os.path.dirname(report_path)}/")

if __name__ == "__main__":
    # Create directories
    os.makedirs("models/ppo", exist_ok=True)
    os.makedirs("models/a2c", exist_ok=True)
    os.makedirs("models/reinforce", exist_ok=True)
    
    # Train all policy gradient models
    models = ["PPO", "A2C", "REINFORCE"]
    
    for model_type in models:
        print(f"\n{'='*60}")
        print(f"🎯 Training {model_type} with Enhanced Curriculum Learning")
        print(f"{'='*60}")
        
        model, callback = train_policy_gradient_with_curriculum(model_type, total_timesteps=300000)
        
        # Create training report
        create_training_report(model_type, callback)
    
    print(f"\n🎉 All Policy Gradient Training Complete!")
    print(f"📁 Models saved in: models/")
    print(f"📊 Reports saved for each model")