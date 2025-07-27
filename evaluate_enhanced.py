import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import DQN, PPO, A2C
from environment.custom_env import B2BNewsSelectionEnv
from environment.fallback_rendering import render_enhanced_dashboard, close_enhanced_rendering
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EnhancedEvaluator:
    """Enhanced model evaluator with comprehensive analysis"""
    
    def __init__(self, num_episodes=50, max_steps_per_episode=20):
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.results = {}
        self.detailed_stats = {}
        
    def evaluate_model(self, model, model_name, model_path):
        """Evaluate a single model comprehensively"""
        print(f"\n🔍 Evaluating {model_name}...")
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            try:
                if model_name == "DQN":
                    model = DQN.load(model_path)
                elif model_name == "PPO":
                    model = PPO.load(model_path)
                elif model_name == "A2C":
                    model = A2C.load(model_path)
                elif model_name == "REINFORCE":
                    model.load(model_path)
                print(f"✅ Loaded {model_name} from {model_path}")
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
                return None
        
        env = B2BNewsSelectionEnv(max_articles=self.max_steps_per_episode)
        
        episode_rewards = []
        episode_lengths = []
        action_distributions = {0: 0, 1: 0, 2: 0}  # Skip, Select, Prioritize
        quality_scores = []
        efficiency_scores = []
        high_value_selections = []
        
        for episode in range(self.num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_actions = []
            episode_qualities = []
            
            for step in range(self.max_steps_per_episode):
                # Get model prediction
                action, _ = model.predict(obs, deterministic=True)
                # Convert action to int if it's a numpy array
                if isinstance(action, np.ndarray):
                    action = int(action.item())
                episode_actions.append(action)
                action_distributions[action] += 1
                
                # Take step
                obs, reward, done, _, info = env.step(action)
                episode_reward += reward
                episode_qualities.append(info.get('quality_score', 0))
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(len(episode_actions))
            quality_scores.extend(episode_qualities)
            efficiency_scores.append(env.episode_stats['efficiency_score'])
            high_value_selections.append(env.episode_stats['high_value_selections'])
            
            if (episode + 1) % 10 == 0:
                print(f"  Episode {episode + 1}/{self.num_episodes}: Reward = {episode_reward:.2f}")
        
        # Calculate comprehensive statistics
        stats = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'median_reward': np.median(episode_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'action_distribution': action_distributions,
            'mean_quality_score': np.mean(quality_scores),
            'mean_efficiency': np.mean(efficiency_scores),
            'total_high_value_selections': sum(high_value_selections),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'quality_scores': quality_scores,
            'efficiency_scores': efficiency_scores,
            'high_value_selections': high_value_selections
        }
        
        self.results[model_name] = stats
        print(f"✅ {model_name} evaluation complete!")
        print(f"   Mean Reward: {stats['mean_reward']:.3f} ± {stats['std_reward']:.3f}")
        print(f"   Mean Efficiency: {stats['mean_efficiency']:.3f}")
        print(f"   Total High-Value Selections: {stats['total_high_value_selections']}")
        
        return stats
    
    def compare_models(self, model_paths):
        """Compare multiple models"""
        print("🚀 Starting Enhanced Model Comparison")
        print("=" * 60)
        
        for model_name, model_path in model_paths.items():
            if model_name == "DQN":
                model = DQN("MlpPolicy", B2BNewsSelectionEnv())
            elif model_name == "PPO":
                model = PPO("MlpPolicy", B2BNewsSelectionEnv())
            elif model_name == "A2C":
                model = A2C("MlpPolicy", B2BNewsSelectionEnv())
            elif model_name == "REINFORCE":
                # REINFORCE needs special handling
                from training.enhanced_training import REINFORCE
                model = REINFORCE(B2BNewsSelectionEnv())
            
            self.evaluate_model(model, model_name, model_path)
        
        return self.results
    
    def create_comprehensive_report(self, output_dir="evaluation_results"):
        """Create comprehensive evaluation report"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create comparison plots
        self._create_comparison_plots(output_dir)
        
        # Create detailed statistics
        self._create_detailed_stats(output_dir)
        
        # Create summary report
        self._create_summary_report(output_dir)
        
        print(f"\n📊 Comprehensive evaluation report saved to: {output_dir}/")
    
    def _create_comparison_plots(self, output_dir):
        """Create comparison plots"""
        model_names = list(self.results.keys())
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Enhanced Model Comparison Analysis', fontsize=16, fontweight='bold')
        
        # 1. Mean Rewards Comparison
        mean_rewards = [self.results[name]['mean_reward'] for name in model_names]
        std_rewards = [self.results[name]['std_reward'] for name in model_names]
        
        bars = axes[0, 0].bar(model_names, mean_rewards, yerr=std_rewards, 
                             capsize=5, alpha=0.7, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0, 0].set_title('Mean Rewards Comparison', fontweight='bold')
        axes[0, 0].set_ylabel('Mean Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, mean_rewards):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Efficiency Scores
        efficiency_scores = [self.results[name]['mean_efficiency'] for name in model_names]
        bars = axes[0, 1].bar(model_names, efficiency_scores, alpha=0.7, 
                             color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0, 1].set_title('Efficiency Scores', fontweight='bold')
        axes[0, 1].set_ylabel('Efficiency Score')
        axes[0, 1].grid(True, alpha=0.3)
        
        for bar, value in zip(bars, efficiency_scores):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. High-Value Selections
        high_value_counts = [self.results[name]['total_high_value_selections'] for name in model_names]
        bars = axes[0, 2].bar(model_names, high_value_counts, alpha=0.7,
                             color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0, 2].set_title('Total High-Value Selections', fontweight='bold')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].grid(True, alpha=0.3)
        
        for bar, value in zip(bars, high_value_counts):
            axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{value}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Reward Distribution
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        for i, name in enumerate(model_names):
            axes[1, 0].hist(self.results[name]['episode_rewards'], alpha=0.6, 
                           label=name, bins=15, color=colors[i % len(colors)])
        axes[1, 0].set_title('Reward Distribution', fontweight='bold')
        axes[1, 0].set_xlabel('Episode Reward')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Action Distribution
        action_names = ['Skip', 'Select', 'Prioritize']
        x = np.arange(len(model_names))
        width = 0.25
        
        for i, action in enumerate([0, 1, 2]):
            action_counts = [self.results[name]['action_distribution'][action] for name in model_names]
            axes[1, 1].bar(x + i*width, action_counts, width, label=action_names[i], alpha=0.7)
        
        axes[1, 1].set_title('Action Distribution', fontweight='bold')
        axes[1, 1].set_xlabel('Model')
        axes[1, 1].set_ylabel('Action Count')
        axes[1, 1].set_xticks(x + width)
        axes[1, 1].set_xticklabels(model_names)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Quality Score Distribution
        for i, name in enumerate(model_names):
            axes[1, 2].hist(self.results[name]['quality_scores'], alpha=0.6,
                           label=name, bins=15, color=colors[i % len(colors)])
        axes[1, 2].set_title('Quality Score Distribution', fontweight='bold')
        axes[1, 2].set_xlabel('Article Quality Score')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_detailed_stats(self, output_dir):
        """Create detailed statistics table"""
        # Create detailed statistics DataFrame
        stats_data = []
        for model_name, stats in self.results.items():
            stats_data.append({
                'Model': model_name,
                'Mean Reward': f"{stats['mean_reward']:.3f} ± {stats['std_reward']:.3f}",
                'Min Reward': f"{stats['min_reward']:.3f}",
                'Max Reward': f"{stats['max_reward']:.3f}",
                'Median Reward': f"{stats['median_reward']:.3f}",
                'Mean Efficiency': f"{stats['mean_efficiency']:.3f}",
                'Mean Quality Score': f"{stats['mean_quality_score']:.3f}",
                'Total High-Value Selections': stats['total_high_value_selections'],
                'Mean Episode Length': f"{stats['mean_episode_length']:.1f}",
                'Skip Actions (%)': f"{stats['action_distribution'][0]/sum(stats['action_distribution'].values())*100:.1f}",
                'Select Actions (%)': f"{stats['action_distribution'][1]/sum(stats['action_distribution'].values())*100:.1f}",
                'Prioritize Actions (%)': f"{stats['action_distribution'][2]/sum(stats['action_distribution'].values())*100:.1f}"
            })
        
        df = pd.DataFrame(stats_data)
        df.to_csv(f"{output_dir}/detailed_statistics.csv", index=False)
        
        # Create summary table
        summary_data = []
        for model_name, stats in self.results.items():
            summary_data.append({
                'Model': model_name,
                'Mean Reward': stats['mean_reward'],
                'Std Reward': stats['std_reward'],
                'Mean Efficiency': stats['mean_efficiency'],
                'High-Value Selections': stats['total_high_value_selections']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Mean Reward', ascending=False)
        summary_df.to_csv(f"{output_dir}/summary_statistics.csv", index=False)
        
        print(f"📋 Detailed statistics saved to: {output_dir}/detailed_statistics.csv")
        print(f"📋 Summary statistics saved to: {output_dir}/summary_statistics.csv")
    
    def _create_summary_report(self, output_dir):
        """Create summary report"""
        # Find best model
        best_model = max(self.results.items(), key=lambda x: x[1]['mean_reward'])
        
        # Convert numpy types to native Python types for JSON serialization
        serializable_results = {}
        for model_name, stats in self.results.items():
            serializable_results[model_name] = {
                'mean_reward': float(stats['mean_reward']),
                'std_reward': float(stats['std_reward']),
                'min_reward': float(stats['min_reward']),
                'max_reward': float(stats['max_reward']),
                'median_reward': float(stats['median_reward']),
                'mean_episode_length': float(stats['mean_episode_length']),
                'action_distribution': {str(k): int(v) for k, v in stats['action_distribution'].items()},
                'mean_quality_score': float(stats['mean_quality_score']),
                'mean_efficiency': float(stats['mean_efficiency']),
                'total_high_value_selections': int(stats['total_high_value_selections']),
                'episode_rewards': [float(x) for x in stats['episode_rewards']],
                'episode_lengths': [int(x) for x in stats['episode_lengths']],
                'quality_scores': [float(x) for x in stats['quality_scores']],
                'efficiency_scores': [float(x) for x in stats['efficiency_scores']],
                'high_value_selections': [int(x) for x in stats['high_value_selections']]
            }
        
        report = {
            'evaluation_date': datetime.now().isoformat(),
            'num_episodes': self.num_episodes,
            'max_steps_per_episode': self.max_steps_per_episode,
            'best_model': {
                'name': best_model[0],
                'mean_reward': float(best_model[1]['mean_reward']),
                'mean_efficiency': float(best_model[1]['mean_efficiency']),
                'total_high_value_selections': int(best_model[1]['total_high_value_selections'])
            },
            'model_rankings': sorted(
                [(name, float(stats['mean_reward'])) for name, stats in self.results.items()],
                key=lambda x: x[1], reverse=True
            ),
            'detailed_results': serializable_results
        }
        
        with open(f"{output_dir}/evaluation_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print(f"\n🏆 EVALUATION SUMMARY")
        print(f"=" * 50)
        print(f"Best Model: {best_model[0]}")
        print(f"Best Mean Reward: {best_model[1]['mean_reward']:.3f}")
        print(f"Best Efficiency: {best_model[1]['mean_efficiency']:.3f}")
        print(f"Best High-Value Selections: {best_model[1]['total_high_value_selections']}")
        print(f"\nModel Rankings:")
        for i, (name, reward) in enumerate(report['model_rankings'], 1):
            print(f"  {i}. {name}: {reward:.3f}")

def main():
    """Main evaluation function"""
    print("🎯 Enhanced Model Evaluation System")
    print("=" * 50)
    
    # Define model paths (update these paths based on your trained models)
    model_paths = {
        "DQN": "models/dqn/best_dqn",
        "PPO": "models/ppo/best_ppo", 
        "A2C": "models/a2c/best_a2c",
        "REINFORCE": "models/reinforce/best_reinforce"
    }
    
    # Check which models exist
    available_models = {}
    for name, path in model_paths.items():
        if name == "REINFORCE":
            if os.path.exists(path + ".pt"):
                available_models[name] = path
                print(f"✅ Found {name} model: {path}")
            else:
                print(f"❌ {name} model not found: {path}")
        else:
            if os.path.exists(path + ".zip"):
                available_models[name] = path
                print(f"✅ Found {name} model: {path}")
            else:
                print(f"❌ {name} model not found: {path}")
    
    if not available_models:
        print("❌ No trained models found! Please train models first.")
        return
    
    # Create evaluator
    evaluator = EnhancedEvaluator(num_episodes=50, max_steps_per_episode=20)
    
    # Run evaluation
    results = evaluator.compare_models(available_models)
    
    # Create comprehensive report
    evaluator.create_comprehensive_report()
    
    print(f"\n🎉 Evaluation complete! Check the 'evaluation_results' directory for detailed reports.")

if __name__ == "__main__":
    main() 