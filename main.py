import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pygame
from environment.custom_env import B2BNewsSelectionEnv
from environment.rendering import render_dashboard, close_rendering

def main():
    pygame.init()
    env = B2BNewsSelectionEnv()
    obs, _ = env.reset()
    clock = pygame.time.Clock()
    
    print("Starting random action visualization...")
    for step in range(20):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                close_rendering(env)
                return
        
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        print(f"Step {step + 1}: Action={['Skip', 'Select', 'Prioritize'][action]}, Reward={reward:.2f}, Done={done}")
        render_dashboard(env, action, "Random Agent", f"Step {step + 1}/20")
        clock.tick(2)  # Slow down to 2 FPS for visibility
        
        if done:
            print("Episode finished. Resetting environment.")
            obs, _ = env.reset()
    
    close_rendering(env)
    print("Visualization complete.")

if __name__ == "__main__":
    main()