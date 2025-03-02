import os
import random
import logging
import pandas as pd
from moviepy.editor import *
import time
from tqdm import tqdm
import numpy as np
from anthropic import Anthropic
from dotenv import load_dotenv
import os
import glob
import re
import csv
import io

# Load environment variables from .env file
load_dotenv()

# ---- CONFIGURATION ----
HOOKS_CSV = "hooks.csv"  # Path to your hooks CSV file (columns: id, text)
HOOK_VIDEOS_FOLDER = "hook_videos"  # Folder containing all hook videos
CTA_VIDEOS_FOLDER = "cta_videos"  # Folder containing all CTA videos
MUSIC_FOLDER = "music"  # Folder containing background music files
OUTPUT_FOLDER = "final_videos"  # Folder to save the final videos
USED_HOOKS_FILE = "used_hooks.txt"  # File to track used hooks
NUM_VIDEOS = 3  # Number of final videos to create

# Font configurations
FONTS = [
    "./fonts/BeVietnamPro-Medium.ttf",     # Clean and modern
    "./fonts/Poppins-SemiBold.ttf",        # Popular for headlines
    "./fonts/Montserrat-Bold.ttf",         # Bold and impactful
    "./fonts/OpenSans-SemiBold.ttf",       # Highly readable
    "./fonts/Roboto-Bold.ttf"              # Clean and professional
]

# Font size ranges based on text length
FONT_SIZE_RANGES = {
    'short': (65, 80),    # For text < 30 chars
    'medium': (50, 65),   # For text 30-60 chars
    'long': (40, 50)      # For text > 60 chars
}

# Color combinations (text_color, background_color) for visually appealing results
COLOR_COMBINATIONS = [
    ("white", "rgba(0,0,0,0.6)"),      # Classic white on semi-transparent black
    ("white", "rgba(233,30,99,0.7)"),  # White on pink
    ("white", "rgba(63,81,181,0.7)"),  # White on indigo
    ("white", "rgba(76,175,80,0.7)"),  # White on green
    ("black", "rgba(255,235,59,0.7)"), # Black on yellow
    ("white", "rgba(156,39,176,0.7)"), # White on purple
    ("black", "rgba(255,255,255,0.7)") # Black on semi-transparent white
]

TEXT_POSITIONS = ['top', 'center']

# ---- SETUP LOGGING ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("video_creation.log"),
        logging.StreamHandler()  # This ensures logs are also printed to console
    ]
)

# ---- FUNCTION DEFINITIONS ----

def setup_output_folder(folder_path):
    """Ensure the output folder exists."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        logging.info(f"Created output folder: {folder_path}")
    else:
        logging.info(f"Output folder already exists: {folder_path}")

def load_hooks(csv_file=HOOKS_CSV):
    """Load hooks from CSV file."""
    try:
        if not os.path.exists(csv_file):
            logging.warning(f"Hooks file {csv_file} does not exist")
            return pd.DataFrame(columns=['text'])
            
        df = pd.read_csv(csv_file)
        if 'text' not in df.columns:
            logging.error(f"Invalid hooks file format: 'text' column not found")
            return pd.DataFrame(columns=['text'])
            
        return df
        
    except Exception as e:
        logging.error(f"Error loading hooks from CSV: {str(e)}")
        return pd.DataFrame(columns=['text'])

def load_used_hooks(file_path):
    """Load the list of already used hooks from a file."""
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            used_hooks = set(f.read().splitlines())
            logging.info(f"Loaded {len(used_hooks)} used hooks.")
            return used_hooks
    else:
        logging.info("No used hooks file found. Starting fresh.")
        return set()

def save_used_hook(file_path, hook_text):
    """Save a used hook to the tracking file."""
    with open(file_path, "a") as f:
        f.write(hook_text + "\n")
    logging.info(f"Saved used hook: {hook_text}")

def get_unused_hook(hooks, used_hooks):
    """Get a random unused hook from the hooks list."""
    unused_hooks = hooks[~hooks["text"].isin(used_hooks)]
    if unused_hooks.empty:
        logging.error("No unused hooks available! All hooks have been used.")
        raise ValueError("No unused hooks available.")
    selected_hook = unused_hooks.sample(1).iloc[0]["text"]
    return selected_hook

def get_random_video(folder_path):
    """Pick a random video file from a folder."""
    if not os.path.exists(folder_path):
        logging.error(f"Folder not found: {folder_path}")
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    video_files = [f for f in os.listdir(folder_path) if f.endswith((".mp4", ".mov"))]
    if not video_files:
        logging.error(f"No video files found in {folder_path}")
        raise FileNotFoundError(f"No video files found in {folder_path}")
    selected_video = random.choice(video_files)
    logging.info(f"Selected video: {selected_video}")
    return os.path.join(folder_path, selected_video)

def get_random_music(folder_path):
    """Pick a random music file from the folder."""
    if not os.path.exists(folder_path):
        logging.error(f"Music folder not found: {folder_path}")
        raise FileNotFoundError(f"Music folder not found: {folder_path}")
    
    music_files = []
    for f in os.listdir(folder_path):
        if f.endswith((".mp3", ".wav", ".m4a")):
            full_path = os.path.join(folder_path, f)
            if os.path.isfile(full_path):
                music_files.append(f)
    
    if not music_files:
        logging.error(f"No music files found in {folder_path}")
        raise FileNotFoundError(f"No music files found in {folder_path}")
    
    selected_music = random.choice(music_files)
    full_path = os.path.join(folder_path, selected_music)
    logging.info(f"Selected music: {selected_music}")
    return full_path

def get_font_size(text):
    """Determine appropriate font size range based on text length"""
    text_length = len(text)
    if text_length < 30:
        return random.randint(*FONT_SIZE_RANGES['short'])
    elif text_length < 60:
        return random.randint(*FONT_SIZE_RANGES['medium'])
    else:
        return random.randint(*FONT_SIZE_RANGES['long'])

def create_video(hook_video_path, hook_text, cta_video_path, music_file_path, output_path):
    try:
        print(f"\nProcessing video with hook: {hook_text}")
        
        # Load hook video
        print("Loading hook video...")
        hook_clip = VideoFileClip(hook_video_path)
        hook_clip = hook_clip.resize((1080, 1920))
        
        # Randomly select text position, color combination, and font
        text_position = random.choice(TEXT_POSITIONS)
        text_color, bg_color = random.choice(COLOR_COMBINATIONS)
        selected_font = random.choice(FONTS)
        font_size = get_font_size(hook_text)
        
        print(f"Using text position: {text_position}")
        print(f"Using colors - Text: {text_color}, Background: {bg_color}")
        print(f"Using font: {os.path.basename(selected_font)} at size: {font_size}")
        
        # Create text overlay with transparent background
        print("Adding text overlay...")
        text_clip = TextClip(
            txt=hook_text,
            fontsize=font_size,
            color=text_color,
            font=selected_font,
            method='caption',
            align='center',
            size=(1080 - 80, None),  # Width with margins
            bg_color=bg_color  # Using randomly selected background color
        ).set_position(('center', text_position))
        
        # Set text duration to match video
        text_clip = text_clip.set_duration(hook_clip.duration)
        
        # Add logo if it exists
        if os.path.exists("logo.webp"):
            print("Adding logo overlay...")
            logo = ImageClip("logo.webp")
            logo = logo.resize(height=100)  # Resize logo
            logo = logo.set_position(('right', 'bottom'))
            logo = logo.set_duration(hook_clip.duration)
            
            # Combine hook, text, and logo
            hook_with_text = CompositeVideoClip(
                [hook_clip, text_clip, logo],
                size=(1080, 1920)
            )
        else:
            # Combine hook and text (text on top of video)
            hook_with_text = CompositeVideoClip(
                [hook_clip, text_clip],
                size=(1080, 1920)
            )
        
        # Load CTA video
        print("Loading CTA video...")
        cta_clip = VideoFileClip(cta_video_path)
        cta_clip = cta_clip.resize((1080, 1920))
        
        # Create final video
        print("Creating final video...")
        final_clip = concatenate_videoclips([hook_with_text, cta_clip])
        
        # Add background music
        print("Adding background music...")
        audio = AudioFileClip(music_file_path)
        
        # Handle audio duration
        if audio.duration < final_clip.duration:
            n_loops = int(np.ceil(final_clip.duration / audio.duration))
            audio = audio.subclip(0, final_clip.duration % audio.duration)
        else:
            audio = audio.subclip(0, final_clip.duration)
        
        # Set volume to 30%
        audio = audio.volumex(0.3)
        
        # Set the audio
        final_clip = final_clip.set_audio(audio)
        
        # Write final video
        print(f"Writing final video to {output_path}...")
        final_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            threads=4
        )
        
        # Get duration and file size
        duration = final_clip.duration
        file_size = os.path.getsize(output_path)
        
        # Close all clips to free up memory
        hook_clip.close()
        text_clip.close()
        cta_clip.close()
        audio.close()
        final_clip.close()
        
        logging.info(f"Created video: {output_path} at resolution (1080, 1920)")
        print(f"✅ Video created successfully: {output_path}")
        return duration, file_size
        
    except Exception as e:
        logging.error(f"Error during video creation for video {output_path.split('_')[-1].split('.')[0]}: {str(e)}")
        print(f"❌ Error creating video: {str(e)}")
        raise  # Re-raise the exception to be handled by the caller

def generate_hooks_with_claude(business_description: str, num_hooks=10):
    """Generate hook lines using Claude AI
    
    Args:
        business_description (str): Description of the business/product
        num_hooks (int): Number of hooks to generate
    """
    try:
        logging.info("Initializing Claude client...")
        client = Anthropic()
        
        prompt = f"""# UGC Hook Generation Prompt

Create 50 viral, attention-grabbing hooks for TikTok and Instagram videos promoting a SaaS product. The hooks should be short, impactful statements that create curiosity and highlight specific benefits.

## Business Information:
{business_description}

## Content Requirements:
- Format each hook as a single line that would be spoken directly to camera in a UGC-style video
- Each hook should be 10-15 words maximum
- Focus on benefits, not features
- Create curiosity or suggest a transformation
- Include hooks that mention time/money saved
- Include hooks that imply insider knowledge
- Include hooks that suggest ease of use
- Include hooks that address pain points
- Use casual, conversational language
- Avoid clickbait that overpromises
- Avoid generic statements that could apply to any business

## Output Format:
Return exactly 50 hooks in this CSV format:

```
id,text
1,"[First hook text here]"
2,"[Second hook text here]"
3,"[Third hook text here]"
```

Continue to 50, with each hook being unique and compelling.

## Hook Themes to Include:
- The ease of finding leads while you sleep
- The untapped potential of Reddit for B2B leads
- Time saved through automation
- Real-world results (clients gained, deals closed)
- The AI-powered advantage
- Comparison to traditional lead generation methods
- Ethical lead generation without spamming
- Specific use cases
- Cost savings
- Testimonial-style hooks ("I never thought...")"""

        logging.info("Sending request to Claude API...")
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        logging.info("Received response from Claude API")
        logging.debug(f"Raw response: {response.content[0].text}")
        
        # Parse CSV format from response
        hooks = []
        csv_text = response.content[0].text
        
        # Skip any text before the CSV content
        if "id,text" in csv_text:
            csv_text = csv_text[csv_text.find("id,text"):]
        
        # Use StringIO to parse CSV
        csv_file = io.StringIO(csv_text)
        csv_reader = csv.DictReader(csv_file)
        
        for row in csv_reader:
            if 'text' in row:
                hook_text = row['text'].strip('"')  # Remove quotes if present
                if hook_text:
                    hooks.append(hook_text)
        
        if not hooks:
            logging.error("No hooks were generated by Claude")
            return []
            
        logging.info(f"Generated {len(hooks)} hooks")
        for i, hook in enumerate(hooks, 1):
            logging.info(f"Hook {i}: {hook}")
            
        return hooks[:num_hooks]  # Return only requested number of hooks
        
    except Exception as e:
        logging.error(f"Error generating hooks with Claude: {str(e)}")
        return []

def save_hooks_to_csv(hooks, output_file=HOOKS_CSV):
    """Save generated hooks to CSV file."""
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['text'])  # Header
            for hook in hooks:
                writer.writerow([hook])
        logging.info(f"Saved {len(hooks)} hooks to {output_file}")
    except Exception as e:
        logging.error(f"Error saving hooks to CSV: {str(e)}")
        raise

def save_video_details(hook_video, hook_text, cta_video, music_file, output_file):
    """Save video details to CSV"""
    try:
        with open('video_details.csv', 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([hook_video, hook_text, cta_video, music_file, output_file])
        logging.info(f"Saved video details: {hook_video},{hook_text},{cta_video},{music_file},{output_file}")
    except Exception as e:
        logging.error(f"Error saving video details: {str(e)}")

def get_last_video_number():
    """Get the last video number from final_videos directory"""
    try:
        video_files = glob.glob(os.path.join("final_videos", "final_video_*.mp4"))
        if not video_files:
            return 0
            
        numbers = []
        for file in video_files:
            try:
                # Extract just the number between final_video_ and .mp4
                match = re.search(r'final_video_(\d+)\.mp4$', file)
                if match:
                    numbers.append(int(match.group(1)))
            except ValueError as e:
                logging.error(f"Error parsing video number from {file}: {str(e)}")
                continue
                
        return max(numbers) if numbers else 0
        
    except Exception as e:
        logging.error(f"Error getting last video number: {str(e)}")
        return 0

def main():
    """Main script to automate video creation."""
    start_time = time.time()
    
    print("\n🎬 Starting UGC Reel Generator...")
    logging.info("Starting video generation process")
    
    try:
        # Get last video number
        last_number = get_last_video_number()
        print(f"📝 Last video number: {last_number}")
        
        # Ensure the output folder exists
        setup_output_folder(OUTPUT_FOLDER)

        # First check if we need to generate hooks
        hooks_df = load_hooks(HOOKS_CSV)
        if len(hooks_df) == 0:
            logging.info("No hooks found. Generating new hooks...")
            business_description = """Our AI-powered tool helps B2B companies find qualified leads on Reddit by automatically identifying 
            relevant discussions and people asking for recommendations in your industry. It monitors Reddit 24/7, 
            finds high-intent prospects, and helps you engage with them naturally and ethically."""
            
            new_hooks = generate_hooks_with_claude(business_description, num_hooks=10)
            if new_hooks:
                save_hooks_to_csv(new_hooks)
                hooks_df = load_hooks(HOOKS_CSV)
            else:
                logging.error("Failed to generate hooks. Please check your API key and try again.")
                return
        
        used_hooks = load_used_hooks(USED_HOOKS_FILE)
        
        for i in range(NUM_VIDEOS):
            print(f"\n🎥 Creating video {i+1}/{NUM_VIDEOS}")
            
            try:
                # Get a random unused hook
                hook_text = get_unused_hook(hooks_df, used_hooks)
                
                # Get random video files
                hook_video = get_random_video(HOOK_VIDEOS_FOLDER)
                cta_video = get_random_video(CTA_VIDEOS_FOLDER)
                music_file = get_random_music(MUSIC_FOLDER)
                
                # Generate output filename
                video_number = get_last_video_number() + 1
                final_video = os.path.join(OUTPUT_FOLDER, f"final_video_{video_number}.mp4")
                
                # Create the video
                duration, file_size = create_video(hook_video, hook_text, cta_video, music_file, final_video)
                save_used_hook(USED_HOOKS_FILE, hook_text)
                save_video_details(hook_video, hook_text, cta_video, music_file, final_video)
                print(f"✅ Successfully created video {i+1} with duration {duration} seconds and file size {file_size} bytes")
            except Exception as e:
                print(f"❌ Error in video creation loop: {str(e)}")
                continue
                
    except Exception as e:
        logging.error(f"Process stopped due to error: {str(e)}")

# ---- RUN SCRIPT ----
if __name__ == "__main__":
    main()