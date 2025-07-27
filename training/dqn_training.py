import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import json
from datetime import datetime
from environment.custom_env import B2BNewsSelectionEnv

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
                # --- Log action distribution if available ---
                if hasattr(self.training_env.envs[0], 'get_action_distribution'):
                    action_dist = self.training_env.envs[0].get_action_distribution()
                    print(f"  Action Distribution: Skip={action_dist[0]:.2f}, Select={action_dist[1]:.2f}, Prioritize={action_dist[2]:.2f}")
                    self.last_action_distribution = action_dist
            
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
    def get_dqn_params(performance_ratio):
        """Get DQN parameters based on performance"""
        base_lr = 0.0001
        base_exploration = 0.2
        
        # Adjust learning rate based on performance
        if performance_ratio < 0.3:
            lr = base_lr * 2.0  # Increase learning rate for poor performance
            exploration = base_exploration * 1.5  # More exploration
        elif performance_ratio > 0.8:
            lr = base_lr * 0.5  # Decrease learning rate for good performance
            exploration = base_exploration * 0.7  # Less exploration
        else:
            lr = base_lr
            exploration = base_exploration
        
        return {
            'learning_rate': lr,
            'exploration_fraction': exploration,
            'exploration_final_eps': 0.01,
            'batch_size': 64,
            'buffer_size': 50000,
            'train_freq': 4,
            'gradient_steps': 1,
            'target_update_interval': 1000,
            'learning_starts': 1000,
            'gamma': 0.99,
            'tau': 1.0
        }

def train_dqn_with_curriculum(total_timesteps=300000):
    """Train DQN with curriculum learning and adaptive hyperparameters"""
    
    print(f"🚀 Starting Enhanced DQN Training")
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
            params = AdaptiveHyperparameters.get_dqn_params(0.5)
            model = DQN("MlpPolicy", env, verbose=1, **params)
        else:
            # Continue training existing model
            model.set_env(env)
        
        # Train
        callback = EnhancedCallback()
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=f"models/dqn/",
            name_prefix=f"dqn_phase_{phase_idx + 1}"
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
            model.save(f"models/dqn/best_dqn")
        
        # Save training statistics
        stats = {
            'phase': phase_idx + 1,
            'difficulty': phase['difficulty'],
            'performance': float(performance),
            'mean_rewards': [float(x) for x in callback.mean_rewards],
            'episode_lengths': [int(x) for x in callback.episode_lengths],
            'efficiency_scores': [float(x) for x in callback.efficiency_scores],
            'high_value_selections': [int(x) for x in callback.high_value_selections],
            # --- Save last action distribution ---
            'last_action_distribution': getattr(callback, 'last_action_distribution', None)
        }
        
        with open(f"models/dqn/phase_{phase_idx + 1}_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
    
    print(f"\n✅ Enhanced DQN Training Complete!")
    print(f"🏆 Best Performance: {best_performance:.3f}")
    print(f"💾 Best model saved to: models/dqn/best_dqn")
    
    return best_model, callback

def create_training_report(callback):
    """Create comprehensive training report"""
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('DQN Training Report', fontsize=16)
    
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
    plt.savefig(f"models/dqn/training_report.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save summary statistics
    summary = {
        'model_type': 'DQN',
        'total_episodes': len(callback.rewards),
        'best_mean_reward': float(callback.best_mean_reward),
        'final_mean_reward': float(callback.mean_rewards[-1]) if callback.mean_rewards else 0.0,
        'avg_episode_length': float(np.mean(callback.episode_lengths)),
        'avg_efficiency': float(np.mean(callback.efficiency_scores)) if callback.efficiency_scores else 0.0,
        'total_high_value_selections': int(sum(callback.high_value_selections)),
        'training_date': datetime.now().isoformat()
    }
    
    with open(f"models/dqn/training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📊 Training report saved to: models/dqn/")

if __name__ == "__main__":
    # Create directories
    os.makedirs("models/dqn", exist_ok=True)
    
    # Train DQN with enhanced training
    print(f"\n{'='*60}")
    print(f"🎯 Training DQN with Enhanced Curriculum Learning")
    print(f"{'='*60}")
    
    model, callback = train_dqn_with_curriculum(total_timesteps=300000)
    
    # Create training report
    create_training_report(callback)
    
    print(f"\n🎉 DQN Training Complete!")
    print(f"📁 Model saved in: models/dqn/")
    print(f"📊 Report saved for DQN")