# B2B News Selection Reinforcement Learning Project

This project demonstrates how reinforcement learning can be applied to business-to-business news filtering and content curation. It simulates an AI agent that learns to select, skip, or prioritize news articles based on their relevance to business offerings and target companies.

## 🎯 What This Project Does

### Core Concept
The project simulates a **business-to-business news filtering system** where an AI agent learns to make intelligent decisions about news articles:

- **Skip**: Ignore irrelevant articles
- **Select**: Choose relevant articles for review
- **Prioritize**: Mark highly relevant articles for immediate attention

### Key Features
- **Custom RL Environment**: Simulates realistic B2B news scenarios
- **Multiple RL Algorithms**: DQN, PPO, A2C, and REINFORCE
- **Real-time Visualization**: Modern Pygame dashboard showing decisions
- **Performance Tracking**: Detailed metrics and CSV results
- **Business Context**: Realistic simulation of content curation tasks

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### 1. Quick Demo (Random Actions)
```bash
# Basic demo
python main.py
```
Shows a 20-step visualization with random actions to understand the environment.

### 2. Train Models
```bash
# Train DQN (Deep Q-Network)
python training/dqn_training.py

# Train PPO, A2C, and REINFORCE
python training/pg_training.py
```

### 3. Play with Trained Models
```bash
# Basic model evaluation
python play_models.py

# Enhanced model evaluation (recommended - 3D visualization)
python play_models_enhanced.py
```
Loads and runs all trained models, showing their performance in real-time.

## 📊 Project Structure

```
custom env/
├── environment/
│   ├── __init__.py
│   ├── custom_env.py          # B2B News Selection Environment
│   └── rendering.py           # Pygame visualization
├── training/
│   ├── __init__.py
│   ├── dqn_training.py        # DQN training script
│   └── pg_training.py         # PPO, A2C, REINFORCE training
├── models/                    # Trained model storage
│   ├── dqn/
│   └── pg/
├── main.py                    # Quick demo script
├── play_models.py             # Model evaluation script
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 🧠 Reinforcement Learning Algorithms

### 1. **DQN (Deep Q-Network)**
- **Type**: Value-based learning
- **Use Case**: Learning optimal action values
- **Strengths**: Good for discrete action spaces, stable learning

### 2. **PPO (Proximal Policy Optimization)**
- **Type**: Policy-based learning
- **Use Case**: Direct policy optimization
- **Strengths**: Sample efficient, stable training

### 3. **A2C (Advantage Actor-Critic)**
- **Type**: Policy-based learning
- **Use Case**: Combining policy and value learning
- **Strengths**: Good balance of exploration and exploitation

### 4. **REINFORCE**
- **Type**: Policy gradient method
- **Use Case**: Direct policy optimization
- **Strengths**: Simple, effective for policy learning

## 🎮 Environment Details

### State Space
The agent observes 4 features for each article:
1. **Topic Score**: Relevance to business offerings
2. **Sentiment Score**: Article sentiment (-1 to 1)
3. **Recency**: How recent the article is (0 to 1)
4. **Company Match**: Whether the article mentions target companies

### Action Space
- **0**: Skip (ignore article)
- **1**: Select (choose for review)
- **2**: Prioritize (mark as high priority)

### Reward System
- **Positive rewards** for selecting relevant articles
- **Negative rewards** for selecting irrelevant articles
- **Small positive rewards** for skipping irrelevant articles
- **Higher rewards** for prioritizing highly relevant articles

## 📈 Performance Metrics

The project tracks:
- **Episode rewards**: Total reward per training episode
- **Mean rewards**: Average over last 100 episodes
- **Action distribution**: How often each action is chosen
- **CSV results**: Detailed performance data for analysis

## 🎨 Visualization Features

### Basic Visualization
The Pygame dashboard shows:
- **Article details**: Company, topic, sentiment, recency
- **Action decisions**: Skip/Select/Prioritize with color coding
- **Reward tracking**: Real-time reward display
- **Performance metrics**: Episode and mean rewards
- **Modern UI**: Clean, professional interface

### Enhanced 3D Visualization 🚀
The enhanced rendering system features:
- **3D Effects**: Shadows, gradients, and depth without OpenGL
- **Neural Network Visualization**: Real-time display of network state and connections
- **Particle System**: Dynamic visual effects for action feedback
- **Enhanced UI**: 3D cards with multiple shadow layers
- **Interactive Elements**: Hover effects and animations
- **Professional Design**: Dark theme with gradient backgrounds
- **Real-time Analysis**: Action probability visualization
- **Advanced Metrics**: Detailed state and probability tracking
- **Compatibility**: Works on all systems (no OpenGL required)

## 🔧 Technical Details

### Dependencies
- `gymnasium`: RL environment framework
- `stable-baselines3`: RL algorithms implementation
- `pygame`: Visualization and UI
- `numpy`: Numerical computations
- `torch`: Deep learning framework
- `pandas`: Data analysis
- `matplotlib`: Plotting (optional)

### Environment Configuration
- **Business Offerings**: "tech consulting" (configurable)
- **Target Companies**: Company A (Nairobi), Company B (Lagos)
- **Article Topics**: tech, finance, partnership, market
- **Max Articles**: 10 per episode

## 📝 Usage Examples

### Customizing the Environment
```python
from environment.custom_env import B2BNewsSelectionEnv

# Custom business offerings and target companies
env = B2BNewsSelectionEnv(
    business_offerings="financial services",
    target_companies=[
        {"name": "TechCorp", "location": "San Francisco"},
        {"name": "FinanceInc", "location": "New York"}
    ]
)
```

### Training Custom Models
```python
from stable_baselines3 import PPO

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)
model.save("my_custom_model")
```

## 🎯 Business Applications

This project demonstrates RL applications for:
- **Content Curation**: Automatically filtering relevant business news
- **Lead Generation**: Identifying potential business opportunities
- **Market Intelligence**: Tracking competitor and industry news
- **Customer Relationship Management**: Prioritizing relevant content

## 📊 Results Analysis

After training, you can analyze:
- **Model comparison**: Which algorithm performs best
- **Learning curves**: How performance improves over time
- **Action patterns**: How different models make decisions
- **Business insights**: Understanding what makes content relevant

## 🤝 Contributing

Feel free to:
- Add new RL algorithms
- Improve the visualization
- Enhance the reward system
- Add more realistic business scenarios
- Optimize training parameters

## 📄 License

This project is open source and available under the MIT License.

---

**Happy Learning! 🚀** 