from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime
from typing import List, Dict, Optional
from . import models

class DBManager:
    def __init__(self, db: Session):
        self.db = db

    def create_hook(self, text: str) -> models.Hook:
        """Create a new hook"""
        hook = models.Hook(text=text)
        self.db.add(hook)
        self.db.commit()
        self.db.refresh(hook)
        return hook

    def get_unused_hooks(self) -> List[models.Hook]:
        """Get all unused hooks"""
        return self.db.query(models.Hook).filter_by(used=False).all()

    def mark_hook_used(self, hook_id: int):
        """Mark a hook as used"""
        hook = self.db.query(models.Hook).filter_by(id=hook_id).first()
        if hook:
            hook.used = True
            hook.last_used = datetime.utcnow()
            self.db.commit()

    def create_video(
        self,
        user_id: int,
        hook_id: int,
        project_id: Optional[int],
        paths: Dict[str, str],
        size: int,
        duration: float
    ) -> models.Video:
        """Create a new video entry"""
        video = models.Video(
            user_id=user_id,
            project_id=project_id,
            hook_id=hook_id,
            hook_video_path=paths['hook'],
            cta_video_path=paths['cta'],
            music_file_path=paths['music'],
            output_path=paths['output'],
            file_size=size,
            duration=duration
        )
        self.db.add(video)
        self.db.commit()
        self.db.refresh(video)
        return video

    def get_user_storage_usage(self, user_id: int) -> int:
        """Get total storage usage for a user in bytes"""
        result = self.db.query(func.sum(models.Video.file_size))\
            .filter_by(user_id=user_id)\
            .filter_by(cleaned=False)\
            .scalar()
        return result or 0

    def check_storage_quota(self, user_id: int, file_size: int) -> bool:
        """Check if user has enough storage quota"""
        user = self.db.query(models.User).filter_by(id=user_id).first()
        if not user:
            return False
        
        current_usage = self.get_user_storage_usage(user_id)
        return (current_usage + file_size) <= user.storage_quota

    def create_project(self, user_id: int, name: str, description: Optional[str] = None) -> models.Project:
        """Create a new project"""
        project = models.Project(
            user_id=user_id,
            name=name,
            description=description
        )
        self.db.add(project)
        self.db.commit()
        self.db.refresh(project)
        return project
