import os
import subprocess

def make_gif_and_video():
    if not os.path.exists("frames"):
        print("No frames directory found. Run play_models.py first.")
        return
    
    # Generate GIF
    subprocess.run([
        "ffmpeg", "-y", "-framerate", "1", "-i", "frames/frame_%04d.png",
        "-vf", "scale=640:-1,fps=10", "-loop", "0", "random_actions.gif"
    ])
    
    # Generate Video
    subprocess.run([
        "ffmpeg", "-y", "-framerate", "1", "-i", "frames/frame_%04d.png",
        "-vf", "scale=1280:720", "-c:v", "libx264", "-pix_fmt", "yuv420p", "model_video.mp4"
    ])
    
    print("GIF and video generated: random_actions.gif, model_video.mp4")

if __name__ == "__main__":
    make_gif_and_video()