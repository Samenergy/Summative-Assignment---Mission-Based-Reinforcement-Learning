import pygame
import pygame.gfxdraw
from typing import Optional, Tuple

# Color scheme
COLORS = {
    'background': (245, 247, 250),
    'card_bg': (255, 255, 255),
    'primary': (59, 130, 246),
    'secondary': (107, 114, 128),
    'success': (34, 197, 94),
    'warning': (251, 191, 36),
    'danger': (239, 68, 68),
    'text_primary': (17, 24, 39),
    'text_secondary': (107, 114, 128),
    'border': (229, 231, 235),
    'shadow': (0, 0, 0, 20)
}

class ModernButton:
    def __init__(self, x: int, y: int, width: int, height: int, text: str, color: Tuple[int, int, int], hover_color: Tuple[int, int, int]):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.current_color = color
        self.font = pygame.font.SysFont("Arial", 16, bold=True)
        
    def draw(self, screen: pygame.Surface):
        # Draw shadow
        shadow_rect = self.rect.copy()
        shadow_rect.x += 2
        shadow_rect.y += 2
        pygame.draw.rect(screen, (0, 0, 0, 30), shadow_rect, border_radius=8)
        
        # Draw button
        pygame.draw.rect(screen, self.current_color, self.rect, border_radius=8)
        
        # Draw text
        text_surface = self.font.render(self.text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)
        
    def handle_mouse(self, mouse_pos: Tuple[int, int]):
        if self.rect.collidepoint(mouse_pos):
            self.current_color = self.hover_color
        else:
            self.current_color = self.color

class ProgressBar:
    def __init__(self, x: int, y: int, width: int, height: int, max_value: int):
        self.rect = pygame.Rect(x, y, width, height)
        self.max_value = max_value
        self.current_value = 0
        
    def update(self, value: int):
        self.current_value = min(value, self.max_value)
        
    def draw(self, screen: pygame.Surface):
        # Background
        pygame.draw.rect(screen, COLORS['border'], self.rect, border_radius=self.rect.height//2)
        
        # Progress
        progress_width = int((self.current_value / self.max_value) * self.rect.width)
        if progress_width > 0:
            progress_rect = pygame.Rect(self.rect.x, self.rect.y, progress_width, self.rect.height)
            pygame.draw.rect(screen, COLORS['primary'], progress_rect, border_radius=self.rect.height//2)

def draw_rounded_rect(surface: pygame.Surface, rect: pygame.Rect, color: Tuple[int, int, int], radius: int = 10):
    """Draw a rounded rectangle"""
    pygame.draw.rect(surface, color, rect, border_radius=radius)

def draw_card(surface: pygame.Surface, rect: pygame.Rect, title: str, content: list, color: Tuple[int, int, int] = COLORS['card_bg']):
    """Draw a card with title and content"""
    # Shadow
    shadow_rect = rect.copy()
    shadow_rect.x += 3
    shadow_rect.y += 3
    pygame.draw.rect(surface, (0, 0, 0, 30), shadow_rect, border_radius=12)
    
    # Card background
    draw_rounded_rect(surface, rect, color, 12)
    
    # Border
    pygame.draw.rect(surface, COLORS['border'], rect, 2, border_radius=12)
    
    # Title
    font_title = pygame.font.SysFont("Arial", 18, bold=True)
    title_surface = font_title.render(title, True, COLORS['text_primary'])
    title_rect = title_surface.get_rect(x=rect.x + 15, y=rect.y + 15)
    surface.blit(title_surface, title_rect)
    
    # Content
    font_content = pygame.font.SysFont("Arial", 14)
    y_offset = rect.y + 45
    for line in content:
        text_surface = font_content.render(line, True, COLORS['text_secondary'])
        text_rect = text_surface.get_rect(x=rect.x + 15, y=y_offset)
        surface.blit(text_surface, text_rect)
        y_offset += 25

def get_sentiment_color(sentiment: float) -> Tuple[int, int, int]:
    """Get color based on sentiment score"""
    if sentiment > 0.3:
        return COLORS['success']
    elif sentiment < -0.3:
        return COLORS['danger']
    else:
        return COLORS['warning']

def get_recency_color(recency: float) -> Tuple[int, int, int]:
    """Get color based on recency score"""
    if recency > 0.7:
        return COLORS['success']
    elif recency > 0.4:
        return COLORS['warning']
    else:
        return COLORS['danger']

def render_dashboard(env, action: Optional[int] = None, model_name: str = None, episode_info: str = None):
    """Render a modern, high-quality dashboard"""
    if not hasattr(env, 'screen') or env.screen is None:
        pygame.init()
        env.screen = pygame.display.set_mode((1200, 800))
        env.font = pygame.font.SysFont("Arial", 24)
        pygame.display.set_caption("B2B News Selection Dashboard")
        
        # Initialize UI components
        env.progress_bar = ProgressBar(50, 50, 1100, 20, env.max_articles)
        env.skip_button = ModernButton(50, 700, 150, 50, "Skip", COLORS['secondary'], (75, 85, 99))
        env.select_button = ModernButton(220, 700, 150, 50, "Select", COLORS['primary'], (37, 99, 235))
        env.prioritize_button = ModernButton(390, 700, 150, 50, "Prioritize", COLORS['success'], (22, 163, 74))

    # Fill background
    env.screen.fill(COLORS['background'])
    
    # Get current article
    article = env.articles[env.current_article_idx]
    
    # Update progress bar
    env.progress_bar.update(env.current_article_idx + 1)
    env.progress_bar.draw(env.screen)
    
    # Progress text
    font_progress = pygame.font.SysFont("Arial", 16)
    progress_text = f"Article {env.current_article_idx + 1} of {env.max_articles}"
    progress_surface = font_progress.render(progress_text, True, COLORS['text_secondary'])
    env.screen.blit(progress_surface, (50, 80))
    
    # Main article card
    article_rect = pygame.Rect(50, 120, 1100, 300)
    article_content = [
        f"Company: {article['company']}",
        f"Topic: {article['topic']}",
        f"Sentiment Score: {article['sentiment']:.3f}",
        f"Recency Score: {article['recency']:.3f}",
        f"Title: {article.get('title', 'N/A')}",
        f"Summary: {article.get('summary', 'N/A')[:100]}..."
    ]
    draw_card(env.screen, article_rect, "Current Article", article_content)
    
    # Metrics cards
    metrics_rect = pygame.Rect(50, 450, 350, 200)
    sentiment_color = get_sentiment_color(article['sentiment'])
    recency_color = get_recency_color(article['recency'])
    
    metrics_content = [
        f"Sentiment: {article['sentiment']:.3f}",
        f"Recency: {article['recency']:.3f}",
        f"Overall Score: {(article['sentiment'] + article['recency']) / 2:.3f}"
    ]
    draw_card(env.screen, metrics_rect, "Article Metrics", metrics_content)
    
    # Sentiment indicator
    sentiment_rect = pygame.Rect(420, 450, 200, 80)
    sentiment_text = "Positive" if article['sentiment'] > 0.3 else "Negative" if article['sentiment'] < -0.3 else "Neutral"
    sentiment_content = [f"Sentiment: {sentiment_text}"]
    draw_card(env.screen, sentiment_rect, "Sentiment Analysis", sentiment_content, sentiment_color)
    
    # Recency indicator
    recency_rect = pygame.Rect(640, 450, 200, 80)
    recency_text = "Recent" if article['recency'] > 0.7 else "Moderate" if article['recency'] > 0.4 else "Old"
    recency_content = [f"Recency: {recency_text}"]
    draw_card(env.screen, recency_rect, "Recency Analysis", recency_content, recency_color)
    
    # Action history card
    history_rect = pygame.Rect(860, 450, 290, 200)
    history_content = [
        f"Last Action: {'Skip' if action == 0 else 'Select' if action == 1 else 'Prioritize' if action == 2 else 'None'}",
        f"Articles Processed: {env.current_article_idx}",
        f"Remaining: {env.max_articles - env.current_article_idx - 1}"
    ]
    if model_name:
        history_content.append(f"Model: {model_name}")
    if episode_info:
        history_content.append(f"Episode: {episode_info}")
    draw_card(env.screen, history_rect, "Action History", history_content)
    
    # Action buttons
    mouse_pos = pygame.mouse.get_pos()
    env.skip_button.handle_mouse(mouse_pos)
    env.select_button.handle_mouse(mouse_pos)
    env.prioritize_button.handle_mouse(mouse_pos)
    
    env.skip_button.draw(env.screen)
    env.select_button.draw(env.screen)
    env.prioritize_button.draw(env.screen)
    
    # Instructions
    font_instructions = pygame.font.SysFont("Arial", 14)
    instructions = [
        "Use the buttons below to make your decision:",
        "• Skip: Pass on this article",
        "• Select: Choose this article for review",
        "• Prioritize: Mark this article as high priority"
    ]
    
    for i, instruction in enumerate(instructions):
        color = COLORS['text_primary'] if i == 0 else COLORS['text_secondary']
        instruction_surface = font_instructions.render(instruction, True, color)
        env.screen.blit(instruction_surface, (600, 700 + i * 20))
    
    pygame.display.flip()

def close_rendering(env):
    """Close the rendering and clean up"""
    if hasattr(env, 'screen') and env.screen is not None:
        pygame.quit()
        env.screen = None

def render_enhanced_dashboard(env, action: Optional[int] = None, model_name: str = None, episode_info: str = None, state_values: list = None, action_probs: list = None):
    """Render an enhanced dashboard with neural network visualization"""
    if not hasattr(env, 'screen') or env.screen is None:
        pygame.init()
        env.screen = pygame.display.set_mode((1400, 900))
        env.font = pygame.font.SysFont("Arial", 24)
        pygame.display.set_caption("Enhanced B2B News Selection Dashboard")
        
        # Initialize UI components
        env.progress_bar = ProgressBar(50, 50, 1300, 20, env.max_articles)
        env.skip_button = ModernButton(50, 800, 150, 50, "Skip", COLORS['secondary'], (75, 85, 99))
        env.select_button = ModernButton(220, 800, 150, 50, "Select", COLORS['primary'], (37, 99, 235))
        env.prioritize_button = ModernButton(390, 800, 150, 50, "Prioritize", COLORS['success'], (22, 163, 74))

    # Fill background with gradient
    env.screen.fill(COLORS['background'])
    
    # Get current article
    article = env.articles[env.current_article_idx]
    
    # Update progress bar
    env.progress_bar.update(env.current_article_idx + 1)
    env.progress_bar.draw(env.screen)
    
    # Progress text
    font_progress = pygame.font.SysFont("Arial", 16)
    progress_text = f"Article {env.current_article_idx + 1} of {env.max_articles}"
    progress_surface = font_progress.render(progress_text, True, COLORS['text_secondary'])
    env.screen.blit(progress_surface, (50, 80))
    
    # Main article card with enhanced styling
    article_rect = pygame.Rect(50, 120, 650, 300)
    article_content = [
        f"Company: {article['company']}",
        f"Topic: {article['topic']}",
        f"Sentiment Score: {article['sentiment']:.3f}",
        f"Recency Score: {article['recency']:.3f}",
        f"Quality Score: {article['quality_score']:.3f}",
        f"Title: {article.get('title', 'N/A')}",
        f"Summary: {article.get('summary', 'N/A')[:80]}..."
    ]
    draw_card(env.screen, article_rect, "Current Article", article_content)
    
    # Neural Network Visualization
    if state_values and action_probs:
        nn_rect = pygame.Rect(720, 120, 650, 300)
        
        # Convert numpy arrays to float values
        state_vals = [float(val) if hasattr(val, 'item') else float(val) for val in state_values]
        action_probs_vals = [float(val) if hasattr(val, 'item') else float(val) for val in action_probs]
        
        nn_content = [
            f"State Values:",
            f"  Topic Relevance: {state_vals[0]:.3f}",
            f"  Sentiment: {state_vals[1]:.3f}",
            f"  Recency: {state_vals[2]:.3f}",
            f"  Company Match: {state_vals[3]:.3f}",
            f"  Time Pressure: {state_vals[4]:.3f}",
            f"  Quality Score: {state_vals[5]:.3f}",
            f"",
            f"Action Probabilities:",
            f"  Skip: {action_probs_vals[0]:.3f}",
            f"  Select: {action_probs_vals[1]:.3f}",
            f"  Prioritize: {action_probs_vals[2]:.3f}"
        ]
        draw_card(env.screen, nn_rect, f"Neural Network ({model_name})", nn_content, (240, 248, 255))
    
    # Metrics cards
    metrics_rect = pygame.Rect(50, 450, 350, 200)
    sentiment_color = get_sentiment_color(article['sentiment'])
    recency_color = get_recency_color(article['recency'])
    
    metrics_content = [
        f"Sentiment: {article['sentiment']:.3f}",
        f"Recency: {article['recency']:.3f}",
        f"Quality Score: {article['quality_score']:.3f}",
        f"Overall Score: {(article['sentiment'] + article['recency'] + article['quality_score']) / 3:.3f}"
    ]
    draw_card(env.screen, metrics_rect, "Article Metrics", metrics_content)
    
    # Sentiment indicator
    sentiment_rect = pygame.Rect(420, 450, 200, 80)
    sentiment_text = "Positive" if article['sentiment'] > 0.3 else "Negative" if article['sentiment'] < -0.3 else "Neutral"
    sentiment_content = [f"Sentiment: {sentiment_text}"]
    draw_card(env.screen, sentiment_rect, "Sentiment Analysis", sentiment_content, sentiment_color)
    
    # Recency indicator
    recency_rect = pygame.Rect(640, 450, 200, 80)
    recency_text = "Recent" if article['recency'] > 0.7 else "Moderate" if article['recency'] > 0.4 else "Old"
    recency_content = [f"Recency: {recency_text}"]
    draw_card(env.screen, recency_rect, "Recency Analysis", recency_content, recency_color)
    
    # Action history card
    history_rect = pygame.Rect(860, 450, 290, 200)
    history_content = [
        f"Last Action: {'Skip' if action == 0 else 'Select' if action == 1 else 'Prioritize' if action == 2 else 'None'}",
        f"Articles Processed: {env.current_article_idx}",
        f"Remaining: {env.max_articles - env.current_article_idx - 1}",
        f"Total Reward: {env.total_reward:.2f}"
    ]
    if model_name:
        history_content.append(f"Model: {model_name}")
    if episode_info:
        history_content.append(f"Episode: {episode_info}")
    draw_card(env.screen, history_rect, "Action History", history_content)
    
    # Action buttons
    mouse_pos = pygame.mouse.get_pos()
    env.skip_button.handle_mouse(mouse_pos)
    env.select_button.handle_mouse(mouse_pos)
    env.prioritize_button.handle_mouse(mouse_pos)
    
    env.skip_button.draw(env.screen)
    env.select_button.draw(env.screen)
    env.prioritize_button.draw(env.screen)
    
    # Instructions
    font_instructions = pygame.font.SysFont("Arial", 14)
    instructions = [
        "AI Model Decision Making:",
        "• Skip: Pass on this article",
        "• Select: Choose this article for review", 
        "• Prioritize: Mark this article as high priority"
    ]
    
    for i, instruction in enumerate(instructions):
        color = COLORS['text_primary'] if i == 0 else COLORS['text_secondary']
        instruction_surface = font_instructions.render(instruction, True, color)
        env.screen.blit(instruction_surface, (600, 800 + i * 20))
    
    pygame.display.flip()

def close_enhanced_rendering(env):
    """Close the enhanced rendering and clean up"""
    if hasattr(env, 'screen') and env.screen is not None:
        pygame.quit()
        env.screen = None

def draw_5x5_grid(surface: pygame.Surface, x: int, y: int, size: int, articles: list, current_idx: int, env):
    """Draw a 5x5 grid representing article selection process"""
    cell_size = size // 5
    grid_colors = {
        'empty': (240, 240, 240),
        'current': (255, 255, 200),
        'selected': (200, 255, 200),
        'prioritized': (255, 200, 200),
        'skipped': (220, 220, 220),
        'border': (100, 100, 100)
    }
    
    # Draw grid cells
    for row in range(5):
        for col in range(5):
            cell_x = x + col * cell_size
            cell_y = y + row * cell_size
            cell_rect = pygame.Rect(cell_x, cell_y, cell_size, cell_size)
            
            # Determine cell state
            article_idx = row * 5 + col
            if article_idx >= len(articles):
                # Empty cell
                color = grid_colors['empty']
                text = ""
            else:
                article = articles[article_idx]
                if article_idx == current_idx:
                    color = grid_colors['current']
                    text = "CURRENT"
                elif article_idx < current_idx:
                    # Processed article - determine action taken
                    if hasattr(env, 'selected_articles') and article in env.selected_articles:
                        color = grid_colors['selected']
                        text = "SELECT"
                    elif hasattr(env, 'prioritized_articles') and article in env.prioritized_articles:
                        color = grid_colors['prioritized']
                        text = "PRIORITY"
                    else:
                        color = grid_colors['skipped']
                        text = "SKIP"
                else:
                    # Future article
                    color = grid_colors['empty']
                    text = f"#{article_idx+1}"
            
            # Draw cell background
            pygame.draw.rect(surface, color, cell_rect)
            pygame.draw.rect(surface, grid_colors['border'], cell_rect, 2)
            
            # Draw text
            if text:
                font = pygame.font.SysFont("Arial", 10)
                text_surface = font.render(text, True, (0, 0, 0))
                text_rect = text_surface.get_rect(center=cell_rect.center)
                surface.blit(text_surface, text_rect)
    
    # Draw grid title
    font_title = pygame.font.SysFont("Arial", 16, bold=True)
    title_surface = font_title.render("Article Selection Grid (5x5)", True, COLORS['text_primary'])
    surface.blit(title_surface, (x, y - 30))

def render_grid_dashboard(env, action: Optional[int] = None, model_name: str = None, episode_info: str = None):
    """Render a grid-based dashboard with 5x5 article selection visualization"""
    if not hasattr(env, 'screen') or env.screen is None:
        pygame.init()
        env.screen = pygame.display.set_mode((1400, 900))
        env.font = pygame.font.SysFont("Arial", 24)
        pygame.display.set_caption("B2B News Selection - Grid View")
        
        # Initialize UI components
        env.progress_bar = ProgressBar(50, 50, 1300, 20, env.max_articles)
        env.skip_button = ModernButton(50, 800, 150, 50, "Skip", COLORS['secondary'], (75, 85, 99))
        env.select_button = ModernButton(220, 800, 150, 50, "Select", COLORS['primary'], (37, 99, 235))
        env.prioritize_button = ModernButton(390, 800, 150, 50, "Prioritize", COLORS['success'], (22, 163, 74))

    # Fill background
    env.screen.fill(COLORS['background'])
    
    # Get current article
    article = env.articles[env.current_article_idx]
    
    # Update progress bar
    env.progress_bar.update(env.current_article_idx + 1)
    env.progress_bar.draw(env.screen)
    
    # Progress text
    font_progress = pygame.font.SysFont("Arial", 16)
    progress_text = f"Article {env.current_article_idx + 1} of {env.max_articles}"
    progress_surface = font_progress.render(progress_text, True, COLORS['text_secondary'])
    env.screen.blit(progress_surface, (50, 80))
    
    # Draw 5x5 grid
    grid_size = 400
    grid_x = 50
    grid_y = 120
    draw_5x5_grid(env.screen, grid_x, grid_y, grid_size, env.articles, env.current_article_idx, env)
    
    # Current article details (right side of grid)
    article_rect = pygame.Rect(500, 120, 850, 300)
    article_content = [
        f"Company: {article['company']}",
        f"Topic: {article['topic']}",
        f"Sentiment Score: {article['sentiment']:.3f}",
        f"Recency Score: {article['recency']:.3f}",
        f"Quality Score: {article['quality_score']:.3f}",
        f"Title: {article.get('title', 'N/A')}",
        f"Summary: {article.get('summary', 'N/A')[:100]}..."
    ]
    draw_card(env.screen, article_rect, "Current Article Details", article_content)
    
    # Grid legend
    legend_rect = pygame.Rect(50, 550, 400, 150)
    legend_content = [
        "Grid Legend:",
        "• CURRENT: Article being processed",
        "• SELECT: Articles chosen for review",
        "• PRIORITY: High-priority articles",
        "• SKIP: Articles passed over",
        "• #N: Future articles to process"
    ]
    draw_card(env.screen, legend_rect, "Grid Legend", legend_content, (248, 250, 252))
    
    # Metrics cards
    metrics_rect = pygame.Rect(500, 450, 350, 200)
    sentiment_color = get_sentiment_color(article['sentiment'])
    recency_color = get_recency_color(article['recency'])
    
    metrics_content = [
        f"Sentiment: {article['sentiment']:.3f}",
        f"Recency: {article['recency']:.3f}",
        f"Quality Score: {article['quality_score']:.3f}",
        f"Overall Score: {(article['sentiment'] + article['recency'] + article['quality_score']) / 3:.3f}"
    ]
    draw_card(env.screen, metrics_rect, "Article Metrics", metrics_content)
    
    # Action history card
    history_rect = pygame.Rect(870, 450, 480, 200)
    history_content = [
        f"Last Action: {'Skip' if action == 0 else 'Select' if action == 1 else 'Prioritize' if action == 2 else 'None'}",
        f"Articles Processed: {env.current_article_idx}",
        f"Remaining: {env.max_articles - env.current_article_idx - 1}",
        f"Total Reward: {env.total_reward:.2f}",
        f"Selected: {len(env.selected_articles)}",
        f"Prioritized: {len(env.prioritized_articles)}"
    ]
    if model_name:
        history_content.append(f"Model: {model_name}")
    if episode_info:
        history_content.append(f"Episode: {episode_info}")
    draw_card(env.screen, history_rect, "Action History", history_content)
    
    # Action buttons
    mouse_pos = pygame.mouse.get_pos()
    env.skip_button.handle_mouse(mouse_pos)
    env.select_button.handle_mouse(mouse_pos)
    env.prioritize_button.handle_mouse(mouse_pos)
    
    env.skip_button.draw(env.screen)
    env.select_button.draw(env.screen)
    env.prioritize_button.draw(env.screen)
    
    # Instructions
    font_instructions = pygame.font.SysFont("Arial", 14)
    instructions = [
        "Grid View: Visualize article selection process across 5x5 grid",
        "• Each cell represents an article in the sequence",
        "• Colors show the action taken for each article",
        "• Current article is highlighted in yellow"
    ]
    
    for i, instruction in enumerate(instructions):
        color = COLORS['text_primary'] if i == 0 else COLORS['text_secondary']
        instruction_surface = font_instructions.render(instruction, True, color)
        env.screen.blit(instruction_surface, (600, 800 + i * 20))
    
    pygame.display.flip()