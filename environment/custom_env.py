import gymnasium as gym
import numpy as np
import random
from gymnasium import spaces
from typing import Dict, Any, Optional

class B2BNewsSelectionEnv(gym.Env):
    """
    Enhanced B2B News Selection Environment
    
    An agent must select relevant B2B news articles based on:
    - Topic relevance to business offerings
    - Sentiment analysis
    - Recency of the news
    - Company match with target companies
    """
    
    def __init__(self, max_articles=20):
        super().__init__()
        
        self.max_articles = max_articles
        self.current_article_idx = 0
        
        # Enhanced observation space with more features
        self.observation_space = spaces.Dict({
            'topic_relevance': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'sentiment': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            'recency': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'company_match': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'time_pressure': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'quality_score': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        })
        
        # Action space: Skip (0), Select (1), Prioritize (2)
        self.action_space = spaces.Discrete(3)
        
        # Enhanced business context
        self.business_offerings = [
            "cloud computing", "AI/ML services", "cybersecurity", 
            "data analytics", "digital transformation", "IoT solutions",
            "blockchain", "edge computing", "5G networks", "automation"
        ]
        
        self.target_companies = [
            "Microsoft", "Google", "Amazon", "IBM", "Oracle", "Salesforce",
            "Adobe", "SAP", "Cisco", "Intel", "NVIDIA", "AMD", "Dell",
            "HP", "VMware", "ServiceNow", "Workday", "Palantir", "Snowflake"
        ]
        
        # Enhanced article generation with more diversity
        self.articles = self._generate_enhanced_articles()
        
        # Performance tracking
        self.total_reward = 0
        self.selected_articles = []
        self.prioritized_articles = []
        self.episode_stats = {
            'high_value_selections': 0,
            'missed_opportunities': 0,
            'efficiency_score': 0
        }
        
        # Time pressure mechanism
        self.time_pressure = 1.0  # Decreases over time
        # Action distribution tracking
        self.action_counts = [0, 0, 0]  # [Skip, Select, Prioritize]
        
    def _generate_enhanced_articles(self):
        """Generate more diverse and realistic articles"""
        articles = []
        qualities = []
        
        # Enhanced topics with better categorization
        topics = [
            "AI and Machine Learning", "Cloud Computing", "Cybersecurity", 
            "Data Analytics", "Digital Transformation", "IoT and Edge Computing",
            "Blockchain Technology", "5G and Networking", "Automation and RPA",
            "Quantum Computing", "Green Technology", "Fintech Innovation",
            "Healthcare Technology", "Supply Chain Tech", "Marketing Technology"
        ]
        
        companies = [
            "Microsoft", "Google", "Amazon", "IBM", "Oracle", "Salesforce",
            "Adobe", "SAP", "Cisco", "Intel", "NVIDIA", "AMD", "Dell",
            "HP", "VMware", "ServiceNow", "Workday", "Palantir", "Snowflake",
            "Zoom", "Slack", "Atlassian", "MongoDB", "Databricks", "Stripe",
            "Twilio", "Shopify", "Square", "Airbnb", "Uber", "Tesla"
        ]
        
        for i in range(self.max_articles):
            topic = random.choice(topics)
            company = random.choice(companies)
            
            # Improved sentiment distribution
            sentiment_bias = random.choice([-0.8, -0.4, 0, 0.4, 0.8])
            sentiment = np.clip(sentiment_bias + random.gauss(0, 0.3), -1, 1)
            
            # Enhanced recency with time decay
            recency = max(0.1, 1.0 - (i * 0.05) + random.gauss(0, 0.1))
            recency = np.clip(recency, 0, 1)
            
            # Calculate topic relevance
            topic_relevance = self._calculate_topic_relevance(topic)
            
            # Calculate company match
            company_match = 1.0 if company in self.target_companies else 0.3
            
            # Wider quality distribution to encourage Skip
            quality_score = np.clip(
                (topic_relevance + company_match + (sentiment + 1) / 2 + recency) / 4 +
                random.gauss(0, 0.15), 0, 1  # Increased std from 0.1 to 0.15
            )
            qualities.append(quality_score)
            
            # Generate realistic titles and summaries
            title = self._generate_title(topic, company, sentiment)
            summary = self._generate_summary(topic, company, sentiment)
            
            articles.append({
                'id': i,
                'title': title,
                'summary': summary,
                'topic': topic,
                'company': company,
                'sentiment': sentiment,
                'recency': recency,
                'topic_relevance': topic_relevance,
                'company_match': company_match,
                'quality_score': quality_score
            })
        
        # Log quality distribution for debugging
        print(f"Article Quality Stats: Mean={np.mean(qualities):.2f}, Std={np.std(qualities):.2f}, "
              f"Min={np.min(qualities):.2f}, Max={np.max(qualities):.2f}")
        return articles
    
    def _calculate_topic_relevance(self, topic):
        """Calculate topic relevance to business offerings"""
        topic_lower = topic.lower()
        relevance = 0.0
        
        for offering in self.business_offerings:
            if offering in topic_lower:
                relevance += 0.3
            elif any(word in topic_lower for word in offering.split()):
                relevance += 0.15
        
        # Add bonus for high-value topics
        high_value_topics = ["AI", "machine learning", "cloud", "cybersecurity", "data analytics"]
        if any(topic in topic_lower for topic in high_value_topics):
            relevance += 0.2
            
        return min(1.0, relevance)
    
    def _generate_title(self, topic, company, sentiment):
        """Generate realistic article titles"""
        templates = [
            f"{company} Announces New {topic} Initiative",
            f"Breaking: {company} Leads {topic} Innovation",
            f"{company} Reports Strong {topic} Performance",
            f"Industry Analysis: {topic} Trends in 2024",
            f"{company} Partners with {topic} Leaders",
            f"Expert Opinion: {topic} Future Outlook",
            f"{company} Expands {topic} Portfolio",
            f"Market Update: {topic} Sector Growth"
        ]
        return random.choice(templates)
    
    def _generate_summary(self, topic, company, sentiment):
        """Generate realistic article summaries"""
        if sentiment > 0.3:
            templates = [
                f"{company} demonstrates exceptional leadership in {topic.lower()}, showing strong growth potential.",
                f"Positive developments in {topic.lower()} position {company} for market success.",
                f"{company}'s innovative approach to {topic.lower()} receives industry recognition."
            ]
        elif sentiment < -0.3:
            templates = [
                f"{company} faces challenges in {topic.lower()} implementation.",
                f"Market concerns arise over {company}'s {topic.lower()} strategy.",
                f"{company} reports setbacks in {topic.lower()} development."
            ]
        else:
            templates = [
                f"{company} maintains steady progress in {topic.lower()} initiatives.",
                f"Mixed results for {company} in {topic.lower()} sector.",
                f"{company} continues development of {topic.lower()} solutions."
            ]
        return random.choice(templates)
    
    def _calculate_enhanced_reward(self, action, article):
        """Calculate enhanced reward based on action and article quality"""
        base_reward = 0.0
        quality = article['quality_score']
        sentiment = article['sentiment']
        company_match = article['company_match']
        recency = article['recency']
        
        if action == 0:  # Skip
            if quality < 0.5:  # Increased threshold from 0.4
                base_reward = 1.2 + recency * 0.5 + abs(min(0, sentiment)) * 0.7  # Even stronger reward for low quality, negative sentiment
            elif quality > 0.75:  # Reduced penalty for high-quality
                base_reward = -0.25 * quality  # Slightly reduced penalty
                self.episode_stats['missed_opportunities'] += 1
            else:
                base_reward = 0.4  # Slightly higher neutral reward for medium-quality
                
        elif action == 1:  # Select
            # Balanced rewards for fair comparison across algorithms
            quality_bonus = quality * 1.6  # Slightly increased for better balance
            company_bonus = company_match * 0.65  # Slightly increased
            sentiment_bonus = max(0, sentiment) * 0.65  # Slightly increased
            recency_bonus = recency * 0.55  # Slightly increased
            base_reward = quality_bonus + company_bonus + sentiment_bonus + recency_bonus
            if quality > 0.8:
                base_reward += 0.7  # Slightly increased
                self.episode_stats['high_value_selections'] += 1
            if sentiment > 0.5:
                base_reward += 0.45  # Slightly increased
                
        elif action == 2:  # Prioritize
            if quality > 0.75:  # Increased threshold from 0.7
                quality_bonus = quality * 1.7  # Increased from 1.5
                company_bonus = company_match * 0.7  # Increased from 0.6
                sentiment_bonus = max(0, sentiment) * 0.5  # Increased from 0.4
                recency_bonus = recency * 0.4  # Increased from 0.3
                base_reward = min(3.5, quality_bonus + company_bonus + sentiment_bonus + recency_bonus)  # Slightly higher cap
                if quality > 0.9:
                    base_reward += 0.9  # Increased from 0.8
                    self.episode_stats['high_value_selections'] += 2
                elif quality > 0.75:
                    base_reward += 0.5  # Increased from 0.4
                    self.episode_stats['high_value_selections'] += 1
            else:
                base_reward = -0.3 * (1 - quality)  # Reduced penalty for medium-quality
                
        # Time pressure adjustment
        time_factor = 1.0 + (1.0 - self.time_pressure) * 0.5
        base_reward *= time_factor
        
        # Stronger per-step diversity incentive
        total_actions = sum(self.action_counts) + 1
        action_dist = [c / total_actions for c in self.action_counts]
        if action_dist[action] < 0.25:  # Increased bonus for underrepresented actions
            base_reward += 0.5  # Increased from 0.3
            
        return base_reward
    
    def _get_enhanced_state(self):
        """Get enhanced state representation as a dict"""
        if self.current_article_idx >= len(self.articles):
            return {
                'topic_relevance': np.array([0.0], dtype=np.float32),
                'sentiment': np.array([0.0], dtype=np.float32),
                'recency': np.array([0.0], dtype=np.float32),
                'company_match': np.array([0.0], dtype=np.float32),
                'time_pressure': np.array([0.0], dtype=np.float32),
                'quality_score': np.array([0.0], dtype=np.float32)
            }
        article = self.articles[self.current_article_idx]
        state = {
            'topic_relevance': np.array([article['topic_relevance']], dtype=np.float32),
            'sentiment': np.array([article['sentiment']], dtype=np.float32),
            'recency': np.array([article['recency']], dtype=np.float32),
            'company_match': np.array([article['company_match']], dtype=np.float32),
            'time_pressure': np.array([self.time_pressure], dtype=np.float32),
            'quality_score': np.array([article['quality_score']], dtype=np.float32)
        }
        return state

    def step(self, action):
        """Execute one step in the environment"""
        if self.current_article_idx >= len(self.articles):
            return self._get_enhanced_state(), 0, True, False, {}
        article = self.articles[self.current_article_idx]
        # Track actions
        self.action_counts[action] += 1
        # Dynamic penalty for overusing any action
        total_so_far = sum(self.action_counts)
        action_dist = [c / max(1, total_so_far) for c in self.action_counts]
        penalty = 0.0
        if action_dist[action] > 0.4:  # Penalty for any action exceeding 40%
            penalty = -0.6 * action_dist[action]  # Dynamic penalty
        # Calculate enhanced reward
        reward = self._calculate_enhanced_reward(action, article) + penalty
        self.total_reward += reward
        # Track actions
        if action == 1:  # Select
            self.selected_articles.append(article)
        elif action == 2:  # Prioritize
            self.prioritized_articles.append(article)
        # Update time pressure
        self.time_pressure = max(0.1, 1.0 - (self.current_article_idx / self.max_articles))
        # Move to next article
        self.current_article_idx += 1
        # Check if episode is done
        done = self.current_article_idx >= len(self.articles)
        if done:
            # Stronger diversity bonus if all actions used
            if all(c > 0 for c in self.action_counts):
                reward += 2.0  # Increased from 1.5
            # Calculate efficiency score
            total_high_quality = sum(1 for a in self.articles if a['quality_score'] > 0.75)
            selected_high_quality = sum(1 for a in self.selected_articles + self.prioritized_articles 
                                      if a['quality_score'] > 0.75)
            self.episode_stats['efficiency_score'] = selected_high_quality / max(1, total_high_quality)
            # Final efficiency bonus
            if self.episode_stats['efficiency_score'] > 0.8:
                reward += 2.0
            elif self.episode_stats['efficiency_score'] > 0.6:
                reward += 1.0
        # Enhanced info dict
        info = {
            'article_id': article['id'],
            'quality_score': article['quality_score'],
            'efficiency_score': self.episode_stats['efficiency_score'],
            'high_value_selections': self.episode_stats['high_value_selections'],
            'time_pressure': self.time_pressure,
            'action_distribution': self.get_action_distribution()
        }
        return self._get_enhanced_state(), reward, done, False, info
    
    def reset(self, seed=None):
        """Reset the environment"""
        super().reset(seed=seed)
        self.current_article_idx = 0
        self.total_reward = 0
        self.selected_articles = []
        self.prioritized_articles = []
        self.time_pressure = 1.0
        # Reset action distribution
        self.action_counts = [0, 0, 0]
        # Reset episode stats
        self.episode_stats = {
            'high_value_selections': 0,
            'missed_opportunities': 0,
            'efficiency_score': 0
        }
        # Regenerate articles for variety
        self.articles = self._generate_enhanced_articles()
        return self._get_enhanced_state(), {}
    
    def render(self):
        """Render the current state"""
        if self.current_article_idx >= len(self.articles):
            return
        
        article = self.articles[self.current_article_idx]
        action_dist = self.get_action_distribution()
        print(f"\nArticle {self.current_article_idx + 1}/{self.max_articles}")
        print(f"Title: {article['title']}")
        print(f"Company: {article['company']}")
        print(f"Topic: {article['topic']}")
        print(f"Sentiment: {article['sentiment']:.3f}")
        print(f"Recency: {article['recency']:.3f}")
        print(f"Quality Score: {article['quality_score']:.3f}")
        print(f"Time Pressure: {self.time_pressure:.3f}")
        print(f"Total Reward: {self.total_reward:.3f}")
        print(f"Action Distribution: Skip={action_dist[0]:.3f}, Select={action_dist[1]:.3f}, Prioritize={action_dist[2]:.3f}")
    
    def get_article_info(self):
        """Get current article information for visualization"""
        if self.current_article_idx >= len(self.articles):
            return None
        
        article = self.articles[self.current_article_idx]
        return {
            'title': article['title'],
            'summary': article['summary'],
            'company': article['company'],
            'topic': article['topic'],
            'sentiment': article['sentiment'],
            'recency': article['recency'],
            'quality_score': article['quality_score'],
            'time_pressure': self.time_pressure,
            'efficiency_score': self.episode_stats['efficiency_score']
        }

    def get_action_distribution(self):
        """Return the action distribution for the last episode."""
        total = sum(self.action_counts)
        if total == 0:
            return [0.0, 0.0, 0.0]
        return [c / total for c in self.action_counts]