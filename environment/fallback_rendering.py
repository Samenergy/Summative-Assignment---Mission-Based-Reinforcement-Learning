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
            val = state_values[i] if i < len(state_values) else 0
            if isinstance(val, np.ndarray):
                val = float(val.squeeze())
            else:
                val = float(val)
            self.nodes.append({
                'x': x_pos, 'y': y_pos, 'value': val,
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
            val = action_probs[i] if i < len(action_probs) else 0
            if isinstance(val, np.ndarray):
                val = float(val.squeeze())
            else:
                val = float(val)
            self.nodes.append({
                'x': x_pos, 'y': y_pos, 'value': val,
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
            color_intensity = max(0, min(255, int(255 * strength)))
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
            brightness = max(0, min(255, int(255 * node['value'])))
            color = tuple(max(0, min(255, c + brightness)) for c in color[:3])
            
            # Draw node
            pygame.draw.circle(screen, color, (node['x'], node['y']), 8)
            
            # Draw value text
            font = pygame.font.SysFont("Arial", 10)
            text = font.render(f"{node['value']:.2f}", True, COLORS['text_primary'])
            screen.blit(text, (node['x'] - 15, node['y'] + 10))

# --- GAME ICONS (simple unicode for now, can be replaced with images) ---
BUTTON_ICONS = {
    'Skip': '\u23ED',        # Fast-forward
    'Select': '\u2714',     # Checkmark
    'Prioritize': '\u2605', # Star
}
LIFE_ICON = '\u2764'        # Heart
SCORE_ICON = '\u2606'       # Star outline
LEVEL_ICON = '\u25B6'       # Play triangle
HELP_ICON = '\u2753'        # Question mark

# --- ENHANCED BUTTON WITH ICON & TOOLTIP ---
class EnhancedButton:
    """Enhanced button with icon and tooltip support"""
    def __init__(self, x: int, y: int, width: int, height: int, text: str, 
                 color: Tuple[int, int, int], hover_color: Tuple[int, int, int], icon: str = None, tooltip: str = None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.current_color = color
        self.font = pygame.font.SysFont("Arial", 16, bold=True)
        self.hover_animation = 0
        self.icon = icon
        self.tooltip = tooltip
        self.is_hovered = False
    
    def draw(self, screen: pygame.Surface):
        # Simple shadow effect
        shadow_rect = self.rect.copy()
        shadow_rect.x += 2
        shadow_rect.y += 2
        pygame.draw.rect(screen, (0, 0, 0), shadow_rect, border_radius=8)
        
        # Button with hover effect
        if self.hover_animation > 0:
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
        
        # Icon (if any)
        if self.icon:
            icon_font = pygame.font.SysFont("Arial", 24, bold=True)
            icon_surface = icon_font.render(self.icon, True, (255, 255, 255))
            icon_rect = icon_surface.get_rect(center=(self.rect.centerx, self.rect.centery - 10))
            screen.blit(icon_surface, icon_rect)
            # Text below icon
            text_surface = self.font.render(self.text, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(self.rect.centerx, self.rect.centery + 18))
            # Text shadow
            shadow_surface = self.font.render(self.text, True, (0, 0, 0))
            shadow_rect = text_rect.copy()
            shadow_rect.x += 1
            shadow_rect.y += 1
            screen.blit(shadow_surface, shadow_rect)
            screen.blit(text_surface, text_rect)
        else:
            # Text with shadow (no icon)
            text_surface = self.font.render(self.text, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=self.rect.center)
            shadow_surface = self.font.render(self.text, True, (0, 0, 0))
            shadow_rect = text_rect.copy()
            shadow_rect.x += 1
            shadow_rect.y += 1
            screen.blit(shadow_surface, shadow_rect)
            screen.blit(text_surface, text_rect)
        
        # Tooltip (if hovered)
        if self.is_hovered and self.tooltip:
            tooltip_font = pygame.font.SysFont("Arial", 14)
            tooltip_surface = tooltip_font.render(self.tooltip, True, (255,255,255))
            tooltip_bg = pygame.Surface((tooltip_surface.get_width()+12, tooltip_surface.get_height()+8))
            tooltip_bg.fill((30,30,30))
            tooltip_bg.blit(tooltip_surface, (6,4))
            screen.blit(tooltip_bg, (self.rect.right+10, self.rect.centery-10))

    def handle_mouse(self, mouse_pos: Tuple[int, int]):
        self.is_hovered = self.rect.collidepoint(mouse_pos)
        if self.is_hovered:
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

# --- TOP BAR DRAW FUNCTION ---
def draw_top_bar(screen, score, lives, level, animate_score=0, animate_lives=0):
    bar_rect = pygame.Rect(0, 0, 1400, 48)
    pygame.draw.rect(screen, COLORS['card_bg'], bar_rect)
    pygame.draw.line(screen, COLORS['border'], (0, 48), (1400, 48), 2)
    font = pygame.font.SysFont("Arial", 22, bold=True)
    # Score
    score_color = (255, 215, 0) if animate_score > 0 else COLORS['success']
    score_text = f"{SCORE_ICON} Score: {score}"
    score_surface = font.render(score_text, True, score_color)
    screen.blit(score_surface, (30, 10))
    # Lives
    lives_color = (255, 80, 80) if animate_lives > 0 else (255, 0, 0)
    lives_text = f"{LIFE_ICON} Lives: {lives}"
    lives_surface = font.render(lives_text, True, lives_color)
    screen.blit(lives_surface, (250, 10))
    # Level
    level_text = f"{LEVEL_ICON} Stage: {level}"
    level_surface = font.render(level_text, True, COLORS['primary'])
    screen.blit(level_surface, (500, 10))
    # Help button (icon)
    help_font = pygame.font.SysFont("Arial", 26, bold=True)
    help_surface = help_font.render(HELP_ICON, True, COLORS['accent'])
    help_rect = help_surface.get_rect(topright=(1380, 8))
    screen.blit(help_surface, help_rect)
    return help_rect

# --- INSTRUCTIONS OVERLAY ---
def draw_help_overlay(screen):
    overlay = pygame.Surface((1400, 900), pygame.SRCALPHA)
    overlay.fill((20, 20, 40, 220))
    font = pygame.font.SysFont("Arial", 28, bold=True)
    text = "How to Play:"
    text_surface = font.render(text, True, (255,255,255))
    overlay.blit(text_surface, (80, 80))
    font2 = pygame.font.SysFont("Arial", 20)
    lines = [
        "You are the News Editor!",
        "Review each article and decide:",
        "- Skip: Pass on the article (\u23ED)",
        "- Select: Choose for review (\u2714)",
        "- Prioritize: Mark as high priority (\u2605)",
        "Earn points for good choices. Don't run out of lives!",
        "Try to get the highest score by the end of the stage.",
        "",
        "Tip: Hover over buttons for tooltips. Click the ? icon to close this help."
    ]
    for i, line in enumerate(lines):
        line_surface = font2.render(line, True, (230,230,230))
        overlay.blit(line_surface, (100, 140 + i*32))
    screen.blit(overlay, (0,0))

# --- MAIN RENDER FUNCTION ---
def render_enhanced_dashboard(env, action: Optional[int] = None, model_name: str = None, 
                            episode_info: str = None, state_values: List[float] = None,
                            action_probs: List[float] = None):
    """Render an enhanced dashboard with simple 3D effects and game-like UI"""
    if not hasattr(env, 'screen') or env.screen is None:
        pygame.init()
        env.screen = pygame.display.set_mode((1400, 900))
        env.font = pygame.font.SysFont("Arial", 24)
        pygame.display.set_caption("News Editor: Arcade Mode")
        # --- GAME STATE ---
        if not hasattr(env, 'score'): env.score = 0
        if not hasattr(env, 'lives'): env.lives = 3
        if not hasattr(env, 'level'): env.level = 1
        if not hasattr(env, 'show_help'): env.show_help = False
        if not hasattr(env, 'score_anim'): env.score_anim = 0
        if not hasattr(env, 'lives_anim'): env.lives_anim = 0
        # Enhanced UI components
        env.progress_bar = EnhancedProgressBar(50, 60, 1300, 25, env.max_articles)
        env.skip_button = EnhancedButton(50, 800, 180, 60, "Skip", COLORS['secondary'], (75, 85, 99), icon=BUTTON_ICONS['Skip'], tooltip="Skip this article")
        env.select_button = EnhancedButton(250, 800, 180, 60, "Select", COLORS['primary'], (37, 99, 235), icon=BUTTON_ICONS['Select'], tooltip="Select for review")
        env.prioritize_button = EnhancedButton(450, 800, 180, 60, "Prioritize", COLORS['success'], (22, 163, 74), icon=BUTTON_ICONS['Prioritize'], tooltip="Mark as high priority")
        env.particle_system = SimpleParticleSystem(1400, 900)
        env.nn_visualizer = SimpleNeuralNetworkVisualizer(1000, 200, 350, 300)
    # Animate score/lives
    if env.score_anim > 0: env.score_anim -= 1
    if env.lives_anim > 0: env.lives_anim -= 1
    env.progress_bar.animate()
    env.particle_system.update()
    # Fill background with gradient
    for y in range(900):
        alpha = y / 900
        color = tuple(int(c1 * (1 - alpha) + c2 * alpha) 
                     for c1, c2 in zip(COLORS['background'], COLORS['gradient_start']))
        pygame.draw.line(env.screen, color, (0, y), (1400, y))
    # --- TOP BAR ---
    help_rect = draw_top_bar(env.screen, env.score, env.lives, env.level, env.score_anim, env.lives_anim)
    # Get current article
    article = env.articles[env.current_article_idx]
    env.progress_bar.update(env.current_article_idx + 1)
    env.progress_bar.draw(env.screen)
    # Progress text
    font_progress = pygame.font.SysFont("Arial", 16, bold=True)
    progress_text = f"Article {env.current_article_idx + 1} of {env.max_articles}"
    progress_surface = font_progress.render(progress_text, True, COLORS['text_primary'])
    env.screen.blit(progress_surface, (50, 95))
    # Main article card
    article_rect = pygame.Rect(50, 140, 900, 350)
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
    # Metrics cards
    metrics_rect = pygame.Rect(50, 510, 280, 200)
    sentiment_color = get_sentiment_color(article['sentiment'])
    recency_color = get_recency_color(article['recency'])
    metrics_content = [
        f"Sentiment: {article['sentiment']:.3f}",
        f"Recency: {article['recency']:.3f}",
        f"Overall Score: {(article['sentiment'] + article['recency']) / 2:.3f}"
    ]
    draw_enhanced_card(env.screen, metrics_rect, "Article Metrics", metrics_content, elevation=4)
    # Sentiment indicator
    sentiment_rect = pygame.Rect(350, 510, 200, 100)
    sentiment_text = "Positive" if article['sentiment'] > 0.3 else "Negative" if article['sentiment'] < -0.3 else "Neutral"
    sentiment_content = [f"Sentiment: {sentiment_text}"]
    draw_enhanced_card(env.screen, sentiment_rect, "Sentiment Analysis", sentiment_content, sentiment_color, elevation=5)
    # Recency indicator
    recency_rect = pygame.Rect(570, 510, 200, 100)
    recency_text = "Recent" if article['recency'] > 0.7 else "Moderate" if article['recency'] > 0.4 else "Old"
    recency_content = [f"Recency: {recency_text}"]
    draw_enhanced_card(env.screen, recency_rect, "Recency Analysis", recency_content, recency_color, elevation=5)
    # Action history card
    history_rect = pygame.Rect(790, 510, 280, 200)
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
        font_nn = pygame.font.SysFont("Arial", 16, bold=True)
        nn_title = font_nn.render("Neural Network State", True, COLORS['text_primary'])
        env.screen.blit(nn_title, (1000, 170))
    # --- ACTION BUTTONS ---
    mouse_pos = pygame.mouse.get_pos()
    env.skip_button.handle_mouse(mouse_pos)
    env.select_button.handle_mouse(mouse_pos)
    env.prioritize_button.handle_mouse(mouse_pos)
    env.skip_button.draw(env.screen)
    env.select_button.draw(env.screen)
    env.prioritize_button.draw(env.screen)
    # Show tooltips if hovered
    for btn in [env.skip_button, env.select_button, env.prioritize_button]:
        if btn.is_hovered and btn.tooltip:
            btn.draw(env.screen)  # Redraw to show tooltip
    # Particle effects for action feedback
    if action is not None:
        button_x = 50 + action * 200 + 90
        button_y = 800 + 30
        particle_color = COLORS['success'] if action == 2 else COLORS['primary'] if action == 1 else COLORS['secondary']
        env.particle_system.add_particle(button_x, button_y, particle_color, velocity=(0, -3), life=30)
        # Animate score/lives for feedback
        if action == 1:
            env.score += 10
            env.score_anim = 10
        elif action == 2:
            env.score += 20
            env.score_anim = 15
        elif action == 0:
            env.lives -= 1
            env.lives_anim = 10
    env.particle_system.draw(env.screen)
    # --- INSTRUCTIONS BAR ---
    font_instructions = pygame.font.SysFont("Arial", 15, bold=True)
    instructions = "[Skip] Fast-forward | [Select] Approve | [Prioritize] Star | Earn points, don't lose all lives!"
    instruction_surface = font_instructions.render(instructions, True, COLORS['text_primary'])
    env.screen.blit(instruction_surface, (700, 860))
    # --- HELP OVERLAY ---
    if hasattr(env, 'show_help') and env.show_help:
        draw_help_overlay(env.screen)
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