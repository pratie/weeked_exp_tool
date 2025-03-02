from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import Optional, List
from pydantic import BaseModel
import os
from datetime import datetime

from db.config import get_db
from api.auth.routes import router as auth_router
from api.auth.config import get_current_user
from db.models import Hook, Video, Project, User
from UGCReelGen import (
    generate_hooks_with_claude, 
    create_video,
    get_random_video,
    get_random_music,
    save_used_hook,
    setup_output_folder,
    HOOK_VIDEOS_FOLDER,
    CTA_VIDEOS_FOLDER,
    MUSIC_FOLDER,
    OUTPUT_FOLDER,
    USED_HOOKS_FILE
)

app = FastAPI(title="UGC Video Generator")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include authentication router
app.include_router(auth_router)

# Request models
class VideoGenerationRequest(BaseModel):
    description: str
    project_id: Optional[int] = None
    num_videos: int = 3

class VideoResponse(BaseModel):
    video_id: int
    output_path: str
    hook_text: str
    duration: float
    file_size: int
    created_at: datetime

def save_hook_to_db(db: Session, hook_text: str) -> Hook:
    hook = Hook(text=hook_text)
    db.add(hook)
    db.commit()
    db.refresh(hook)
    return hook

def save_video_to_db(
    db: Session,
    user_id: int,
    project_id: Optional[int],
    hook_id: int,
    hook_video_path: str,
    cta_video_path: str,
    output_path: str,
    file_size: int,
    duration: float
) -> Video:
    # Create video object with optional project_id
    video_data = {
        "user_id": user_id,
        "hook_id": hook_id,
        "hook_video_path": hook_video_path,
        "cta_video_path": cta_video_path,
        "output_path": output_path,
        "file_size": file_size,
        "duration": duration
    }
    
    # Only add project_id if it's not None
    if project_id is not None:
        video_data["project_id"] = project_id
    
    video = Video(**video_data)
    db.add(video)
    db.commit()
    db.refresh(video)
    return video

async def generate_video_task(
    hook_text: str,
    user_id: int,
    project_id: Optional[int],
    db: Session
) -> VideoResponse:
    try:
        # 1. Save hook to database
        hook = save_hook_to_db(db, hook_text)
        save_used_hook(USED_HOOKS_FILE, hook_text)
        
        # 2. Get random videos
        hook_video = get_random_video(HOOK_VIDEOS_FOLDER)
        cta_video = get_random_video(CTA_VIDEOS_FOLDER)
        music_file = get_random_music(MUSIC_FOLDER)
        
        # 3. Generate output path
        output_filename = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        # Ensure output directory exists
        setup_output_folder(OUTPUT_FOLDER)
        
        # 4. Create the video
        duration, file_size = create_video(
            hook_video,
            hook_text,
            cta_video,
            music_file,
            output_path
        )
        
        # 5. Save video details to database
        video = save_video_to_db(
            db,
            user_id,
            project_id,
            hook.id,
            hook_video,
            cta_video,
            output_path,
            file_size,
            duration
        )
        
        return VideoResponse(
            video_id=video.id,
            output_path=video.output_path,
            hook_text=hook_text,
            duration=duration,
            file_size=file_size,
            created_at=video.created_at
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating video: {str(e)}"
        )

@app.post("/generate-video/", response_model=List[VideoResponse])
async def generate_videos(
    request: VideoGenerationRequest,
    background_tasks: BackgroundTasks,
    current_user_email: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Generate videos based on description"""
    try:
        # 1. Get user
        user = db.query(User).filter(User.email == current_user_email).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # 2. Validate project_id if provided
        project_id = None
        if request.project_id:
            project = db.query(Project).filter(Project.id == request.project_id).first()
            if not project:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Project with id {request.project_id} not found"
                )
            project_id = project.id
        
        # 3. Generate hooks using Claude
        hooks = generate_hooks_with_claude(request.description)
        
        # 4. Start video generation tasks
        video_responses = []
        for i in range(min(request.num_videos, len(hooks))):
            hook_text = hooks[i]
            video_response = await generate_video_task(
                hook_text,
                user.id,
                project_id,  # Pass None if no project_id
                db
            )
            video_responses.append(video_response)
        
        return video_responses
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
