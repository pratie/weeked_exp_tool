from sqlalchemy import Column, Integer, BigInteger, String, Float, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .config import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)
    storage_quota = Column(BigInteger, default=5368709120)  # 5GB
    is_active = Column(Boolean, default=True)
    
    # Relationships
    videos = relationship("Video", back_populates="user")
    projects = relationship("Project", back_populates="user")

class Project(Base):
    __tablename__ = "projects"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String)
    description = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="projects")
    videos = relationship("Video", back_populates="project")

class Video(Base):
    __tablename__ = "videos"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    project_id = Column(Integer, ForeignKey("projects.id"))
    hook_video_path = Column(String)
    hook_id = Column(Integer, ForeignKey("hooks.id"))
    cta_video_path = Column(String)
    music_file_path = Column(String)
    output_path = Column(String)
    file_size = Column(Integer)
    duration = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    cleaned = Column(Boolean, default=False)
    
    # Relationships
    user = relationship("User", back_populates="videos")
    project = relationship("Project", back_populates="videos")
    hook = relationship("Hook", back_populates="videos")

class Hook(Base):
    __tablename__ = "hooks"
    id = Column(Integer, primary_key=True)
    text = Column(String)
    used = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_used = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    videos = relationship("Video", back_populates="hook")
