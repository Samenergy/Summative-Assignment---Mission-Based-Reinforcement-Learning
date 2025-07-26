import pygame
import pygame.gfxdraw
import numpy as np
from typing import Optional, Tuple, List
import math
import time

# Enhanced color scheme (no OpenGL required)
COLORS = {
    'background': (15, 23, 42),  # Dark blue background
    'card_bg': (30, 41, 59),     # Darker card background
    'primary': (59, 130, 246),    # Blue
    'secondary': (107, 114, 128), # Gray
    'success': (34, 197, 94),     # Green
    'warning': (251, 191, 36),    # Yellow
    'danger': (239, 68, 68),      # Red
    'text_primary': (248, 250, 252), # Light text
    'text_secondary': (148, 163, 184), # Muted text
    'border': (51, 65, 85),       # Dark border
    'accent': (139, 92, 246),     # Purple accent
    'gradient_start': (59, 130, 246),
    'gradient_end': (139, 92, 246)
}

class SimpleParticleSystem:
    """Simple particle system without OpenGL"""
    def __init__(self, screen_width, screen_height):
        self.particles = []
        self.screen_width = screen_width
        self.screen_height = screen_height
        
    def add_particle(self, x, y, color, velocity=(0, -2), life=60):
        self.particles.append({
            'x': x, 'y': y, 'vx': velocity[0], 'vy': velocity[1],
            'color': color, 'life': life, 'max_life': life
        })
    
    def update(self):
        for particle in self.particles[:]:
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['life'] -= 1
            
            if particle['life'] <= 0:
                self.particles.remove(particle)
    
    def draw(self, screen):
        for particle in self.particles:
            alpha = particle['life'] / particle['max_life']
            color = particle['color'][:3]  # Simple color without alpha
            size = int(3 * alpha)
            if size > 0:
                pygame.draw.circle(screen, color, 
                                 (int(particle['x']), int(particle['y'])), size)

class SimpleNeuralNetworkVisualizer:
    """Simple neural network visualization without OpenGL"""
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.nodes = []
        self.connections = []
        
    def update_network(self, state_values, action_probs):
        # Create nodes for state (input layer)
        self.nodes = []
        self.connections = []
        
        # Ensure action_probs is a list/array
        if not isinstance(action_probs, (list, np.ndarray)):
            action_probs = [0.33, 0.33, 0.34]  # Default probabilities
        
        # Input layer (state values) - now 6 features
        input_nodes = 6
        for i in range(input_nodes):
            x_pos = self.x + 50
            y_pos = self.y + 30 + i * 35  # Adjusted spacing for 6 nodes
            self.nodes.append({
                'x': x_pos, 'y': y_pos, 'value': state_values[i] if i < len(state_values) else 0,
                'layer': 0, 'type': 'input'
            })
        
        # Hidden layer
        hidden_nodes = 6
        for i in range(hidden_nodes):
            x_pos = self.x + 150
            y_pos = self.y + 30 + i * 35
            self.nodes.append({
                'x': x_pos, 'y': y_pos, 'value': 0.5,
                'layer': 1, 'type': 'hidden'
            })
        
        # Output layer (actions)
        output_nodes = 3
        for i in range(output_nodes):
            x_pos = self.x + 250
            y_pos = self.y + 60 + i * 40
            self.nodes.append({
                'x': x_pos, 'y': y_pos, 'value': action_probs[i] if i < len(action_probs) else 0,
                'layer': 2, 'type': 'output'
            })
        
        # Create connections
        for input_node in self.nodes[:input_nodes]:
            for hidden_node in self.nodes[input_nodes:input_nodes+hidden_nodes]:
                self.connections.append((input_node, hidden_node))
        
        for hidden_node in self.nodes[input_nodes:input_nodes+hidden_nodes]:
            for output_node in self.nodes[input_nodes+hidden_nodes:]:
                self.connections.append((hidden_node, output_node))
    
    def draw(self, screen):
        # Draw connections
        for connection in self.connections:
            start_node = connection[0]
            end_node = connection[1]
            strength = (start_node['value'] + end_node['value']) / 2
            color_intensity = int(255 * strength)
            color = (color_intensity, color_intensity, color_intensity)
            
            pygame.draw.line(screen, color, 
                           (start_node['x'], start_node['y']),
                           (end_node['x'], end_node['y']), 2)
        
        # Draw nodes
        for node in self.nodes:
            # Node color based on type and value
            if node['type'] == 'input':
                color = COLORS['primary']
            elif node['type'] == 'hidden':
                color = COLORS['accent']
            else:  # output
                color = COLORS['success']
            
            # Adjust brightness based on activation
            brightness = int(255 * node['value'])
            color = tuple(min(255, c + brightness) for c in color[:3])
            
            # Draw node
            pygame.draw.circle(screen, color, (node['x'], node['y']), 8)
            
            # Draw value text
            font = pygame.font.SysFont("Arial", 10)
            text = font.render(f"{node['value']:.2f}", True, COLORS['text_primary'])
            screen.blit(text, (node['x'] - 15, node['y'] + 10))

class EnhancedButton:
    """Enhanced button with simple 3D effects"""
    def __init__(self, x: int, y: int, width: int, height: int, text: str, 
                 color: Tuple[int, int, int], hover_color: Tuple[int, int, int]):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.current_color = color
        self.font = pygame.font.SysFont("Arial", 16, bold=True)
        self.hover_animation = 0
        
    def draw(self, screen: pygame.Surface):
        # Simple shadow effect
        shadow_rect = self.rect.copy()
        shadow_rect.x += 2
        shadow_rect.y += 2
        pygame.draw.rect(screen, (0, 0, 0), shadow_rect, border_radius=8)
        
        # Button with hover effect
        if self.hover_animation > 0:
            # Add hover glow
            glow_rect = self.rect.inflate(int(self.hover_animation * 4), int(self.hover_animation * 4))
            glow_color = tuple(min(255, c + 30) for c in self.current_color[:3])
            pygame.draw.rect(screen, glow_color, glow_rect, border_radius=12)
        
        # Main button
        pygame.draw.rect(screen, self.current_color, self.rect, border_radius=8)
        
        # 3D highlight
        highlight_rect = self.rect.copy()
        highlight_rect.height = self.rect.height // 2
        highlight_color = tuple(min(255, c + 30) for c in self.current_color[:3])
        pygame.draw.rect(screen, highlight_color, highlight_rect, border_radius=8)
        
        # Text with shadow
        text_surface = self.font.render(self.text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=self.rect.center)
        
        # Text shadow
        shadow_surface = self.font.render(self.text, True, (0, 0, 0))
        shadow_rect = text_rect.copy()
        shadow_rect.x += 1
        shadow_rect.y += 1
        screen.blit(shadow_surface, shadow_rect)
        
        # Main text
        screen.blit(text_surface, text_rect)
        
    def handle_mouse(self, mouse_pos: Tuple[int, int]):
        if self.rect.collidepoint(mouse_pos):
            self.current_color = self.hover_color
            self.hover_animation = min(1.0, self.hover_animation + 0.1)
        else:
            self.current_color = self.color
            self.hover_animation = max(0.0, self.hover_animation - 0.1)

class EnhancedProgressBar:
    """Enhanced progress bar with simple 3D effects"""
    def __init__(self, x: int, y: int, width: int, height: int, max_value: int):
        self.rect = pygame.Rect(x, y, width, height)
        self.max_value = max_value
        self.current_value = 0
        self.target_value = 0
        self.animation_speed = 0.1
        
    def update(self, value: int):
        self.target_value = min(value, self.max_value)
        
    def animate(self):
        diff = self.target_value - self.current_value
        if abs(diff) > 0.01:
            self.current_value += diff * self.animation_speed
        
    def draw(self, screen: pygame.Surface):
        # Background with shadow
        bg_rect = self.rect.copy()
        bg_rect.y += 2
        pygame.draw.rect(screen, (0, 0, 0), bg_rect, border_radius=self.rect.height//2)
        pygame.draw.rect(screen, COLORS['border'], self.rect, border_radius=self.rect.height//2)
        
        # Progress with gradient effect
        progress_width = int((self.current_value / self.max_value) * self.rect.width)
        if progress_width > 0:
            progress_rect = pygame.Rect(self.rect.x, self.rect.y, progress_width, self.rect.height)
            
            # Simple gradient effect
            for i in range(progress_rect.height):
                alpha = 1.0 - (i / progress_rect.height) * 0.3
                color = tuple(int(c * alpha) for c in COLORS['primary'][:3])
                line_rect = pygame.Rect(progress_rect.x, progress_rect.y + i, progress_rect.width, 1)
                pygame.draw.rect(screen, color, line_rect, border_radius=1)
            
            # Highlight
            highlight_rect = progress_rect.copy()
            highlight_rect.height = progress_rect.height // 3
            highlight_color = tuple(min(255, c + 50) for c in COLORS['primary'][:3])
            pygame.draw.rect(screen, highlight_color, highlight_rect, border_radius=self.rect.height//2)

def draw_enhanced_card(surface: pygame.Surface, rect: pygame.Rect, title: str, content: list, 
                      color: Tuple[int, int, int] = COLORS['card_bg'], elevation: int = 4):
    """Draw an enhanced card with simple 3D effects"""
    # Multiple shadow layers for 3D effect
    for i in range(elevation, 0, -1):
        shadow_rect = rect.copy()
        shadow_rect.x += i
        shadow_rect.y += i
        pygame.draw.rect(surface, (0, 0, 0), shadow_rect, border_radius=12)
    
    # Main card
    pygame.draw.rect(surface, color, rect, border_radius=12)
    
    # 3D border effect
    border_rect = rect.copy()
    border_rect.height = rect.height // 2
    border_color = tuple(min(255, c + 20) for c in color[:3])
    pygame.draw.rect(surface, border_color, border_rect, border_radius=12)
    
    # Title with shadow
    font_title = pygame.font.SysFont("Arial", 18, bold=True)
    title_surface = font_title.render(title, True, COLORS['text_primary'])
    title_rect = title_surface.get_rect(x=rect.x + 15, y=rect.y + 15)
    
    # Title shadow
    shadow_surface = font_title.render(title, True, (0, 0, 0))
    shadow_rect = title_rect.copy()
    shadow_rect.x += 1
    shadow_rect.y += 1
    surface.blit(shadow_surface, shadow_rect)
    surface.blit(title_surface, title_rect)
    
    # Content with enhanced styling
    font_content = pygame.font.SysFont("Arial", 14)
    y_offset = rect.y + 45
    for i, line in enumerate(content):
        # Alternating colors for better readability
        text_color = COLORS['text_primary'] if i % 2 == 0 else COLORS['text_secondary']
        text_surface = font_content.render(line, True, text_color)
        text_rect = text_surface.get_rect(x=rect.x + 15, y=y_offset)
        surface.blit(text_surface, text_rect)
        y_offset += 25

def render_enhanced_dashboard(env, action: Optional[int] = None, model_name: str = None, 
                            episode_info: str = None, state_values: List[float] = None,
                            action_probs: List[float] = None):
    """Render an enhanced dashboard with simple 3D effects"""
    if not hasattr(env, 'screen') or env.screen is None:
        pygame.init()
        env.screen = pygame.display.set_mode((1400, 900))
        env.font = pygame.font.SysFont("Arial", 24)
        pygame.display.set_caption("Enhanced B2B News Selection Dashboard")
        
        # Initialize enhanced UI components
        env.progress_bar = EnhancedProgressBar(50, 50, 1300, 25, env.max_articles)
        env.skip_button = EnhancedButton(50, 800, 180, 60, "Skip", COLORS['secondary'], (75, 85, 99))
        env.select_button = EnhancedButton(250, 800, 180, 60, "Select", COLORS['primary'], (37, 99, 235))
        env.prioritize_button = EnhancedButton(450, 800, 180, 60, "Prioritize", COLORS['success'], (22, 163, 74))
        
        # Initialize particle system
        env.particle_system = SimpleParticleSystem(1400, 900)
        
        # Initialize neural network visualizer
        env.nn_visualizer = SimpleNeuralNetworkVisualizer(1000, 200, 350, 300)

    # Animate progress bar
    env.progress_bar.animate()
    
    # Update particle system
    env.particle_system.update()
    
    # Fill background with gradient
    for y in range(900):
        alpha = y / 900
        color = tuple(int(c1 * (1 - alpha) + c2 * alpha) 
                     for c1, c2 in zip(COLORS['background'], COLORS['gradient_start']))
        pygame.draw.line(env.screen, color, (0, y), (1400, y))

    # Get current article
    article = env.articles[env.current_article_idx]
    
    # Update progress bar
    env.progress_bar.update(env.current_article_idx + 1)
    env.progress_bar.draw(env.screen)
    
    # Progress text with shadow
    font_progress = pygame.font.SysFont("Arial", 16, bold=True)
    progress_text = f"Article {env.current_article_idx + 1} of {env.max_articles}"
    progress_surface = font_progress.render(progress_text, True, COLORS['text_primary'])
    env.screen.blit(progress_surface, (50, 85))
    
    # Main article card with enhanced effects
    article_rect = pygame.Rect(50, 120, 900, 350)
    article_content = [
        f"Company: {article['company']}",
        f"Topic: {article['topic']}",
        f"Sentiment Score: {article['sentiment']:.3f}",
        f"Recency Score: {article['recency']:.3f}",
        f"Quality Score: {article.get('quality_score', 0):.3f}",
        f"Time Pressure: {article.get('time_pressure', 1.0):.3f}",
        f"Title: {article.get('title', 'N/A')}",
        f"Summary: {article.get('summary', 'N/A')[:100]}..."
    ]
    draw_enhanced_card(env.screen, article_rect, "Current Article", article_content, elevation=6)
    
    # Enhanced metrics cards
    metrics_rect = pygame.Rect(50, 500, 280, 200)
    sentiment_color = get_sentiment_color(article['sentiment'])
    recency_color = get_recency_color(article['recency'])
    
    metrics_content = [
        f"Sentiment: {article['sentiment']:.3f}",
        f"Recency: {article['recency']:.3f}",
        f"Overall Score: {(article['sentiment'] + article['recency']) / 2:.3f}"
    ]
    draw_enhanced_card(env.screen, metrics_rect, "Article Metrics", metrics_content, elevation=4)
    
    # Sentiment indicator with enhanced effects
    sentiment_rect = pygame.Rect(350, 500, 200, 100)
    sentiment_text = "Positive" if article['sentiment'] > 0.3 else "Negative" if article['sentiment'] < -0.3 else "Neutral"
    sentiment_content = [f"Sentiment: {sentiment_text}"]
    draw_enhanced_card(env.screen, sentiment_rect, "Sentiment Analysis", sentiment_content, sentiment_color, elevation=5)
    
    # Recency indicator with enhanced effects
    recency_rect = pygame.Rect(570, 500, 200, 100)
    recency_text = "Recent" if article['recency'] > 0.7 else "Moderate" if article['recency'] > 0.4 else "Old"
    recency_content = [f"Recency: {recency_text}"]
    draw_enhanced_card(env.screen, recency_rect, "Recency Analysis", recency_content, recency_color, elevation=5)
    
    # Action history card with enhanced info
    history_rect = pygame.Rect(790, 500, 280, 200)
    history_content = [
        f"Last Action: {'Skip' if action == 0 else 'Select' if action == 1 else 'Prioritize' if action == 2 else 'None'}",
        f"Articles Processed: {env.current_article_idx}",
        f"Remaining: {env.max_articles - env.current_article_idx - 1}"
    ]
    if model_name:
        history_content.append(f"Model: {model_name}")
    if episode_info:
        history_content.append(f"Episode: {episode_info}")
    draw_enhanced_card(env.screen, history_rect, "Action History", history_content, elevation=4)
    
    # Neural Network Visualization
    if state_values and action_probs:
        env.nn_visualizer.update_network(state_values, action_probs)
        env.nn_visualizer.draw(env.screen)
        
        # NN title
        font_nn = pygame.font.SysFont("Arial", 16, bold=True)
        nn_title = font_nn.render("Neural Network State", True, COLORS['text_primary'])
        env.screen.blit(nn_title, (1000, 170))
    
    # Enhanced action buttons
    mouse_pos = pygame.mouse.get_pos()
    env.skip_button.handle_mouse(mouse_pos)
    env.select_button.handle_mouse(mouse_pos)
    env.prioritize_button.handle_mouse(mouse_pos)
    
    env.skip_button.draw(env.screen)
    env.select_button.draw(env.screen)
    env.prioritize_button.draw(env.screen)
    
    # Particle effects for action feedback
    if action is not None:
        button_x = 50 + action * 200 + 90  # Center of the action button
        button_y = 800 + 30
        particle_color = COLORS['success'] if action == 2 else COLORS['primary'] if action == 1 else COLORS['secondary']
        env.particle_system.add_particle(button_x, button_y, particle_color, velocity=(0, -3), life=30)
    
    # Draw particles
    env.particle_system.draw(env.screen)
    
    # Enhanced instructions with shadow
    font_instructions = pygame.font.SysFont("Arial", 14, bold=True)
    instructions = [
        "Use the buttons below to make your decision:",
        "• Skip: Pass on this article",
        "• Select: Choose this article for review", 
        "• Prioritize: Mark this article as high priority"
    ]
    
    for i, instruction in enumerate(instructions):
        color = COLORS['text_primary'] if i == 0 else COLORS['text_secondary']
        instruction_surface = font_instructions.render(instruction, True, color)
        # Add text shadow for 3D effect
        shadow_surface = font_instructions.render(instruction, True, (0, 0, 0))
        shadow_rect = instruction_surface.get_rect(x=700, y=800 + i * 25)
        shadow_rect.x += 1
        shadow_rect.y += 1
        env.screen.blit(shadow_surface, shadow_rect)
        env.screen.blit(instruction_surface, (700, 800 + i * 25))
    
    pygame.display.flip()

def get_sentiment_color(sentiment: float) -> Tuple[int, int, int]:
    """Get enhanced color based on sentiment score"""
    if sentiment > 0.3:
        return COLORS['success']
    elif sentiment < -0.3:
        return COLORS['danger']
    else:
        return COLORS['warning']

def get_recency_color(recency: float) -> Tuple[int, int, int]:
    """Get enhanced color based on recency score"""
    if recency > 0.7:
        return COLORS['success']
    elif recency > 0.4:
        return COLORS['warning']
    else:
        return COLORS['danger']

def close_enhanced_rendering(env):
    """Close the enhanced rendering and clean up"""
    if hasattr(env, 'screen') and env.screen is not None:
        pygame.quit()
        env.screen = None 