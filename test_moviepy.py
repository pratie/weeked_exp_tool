try:
    from moviepy.editor import VideoFileClip
    print("Successfully imported VideoFileClip from moviepy.editor")
except ImportError as e:
    print(f"Error importing: {e}")
    
try:
    from moviepy.video.io.VideoFileClip import VideoFileClip
    print("Successfully imported VideoFileClip directly")
except ImportError as e:
    print(f"Error importing: {e}")
