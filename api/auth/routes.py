from fastapi import APIRouter, Depends, HTTPException, status
from google.oauth2 import id_token
from google.auth.transport import requests
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
from db.models import User
from db.config import SessionLocal, DATABASE_URL
from sqlalchemy.orm import Session
from .crud import UserCRUD
from .config import get_current_user, security

router = APIRouter(prefix="/auth", tags=["auth"])

# Get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/test-login")
async def test_login(
    email: str,
    db: Session = Depends(get_db)
):
    """Development only: Create a test user and return access token"""
    try:
        # Get or create user
        user = UserCRUD.get_user_by_email(db, email)
        if not user:
            user = UserCRUD.create_user(db, email)
            print("\n=== New Test User Created ===")
            print(f"Email: {email}")
            print(f"Database: {DATABASE_URL}")
            print("=========================\n")
        else:
            UserCRUD.update_last_login(db, email)
            print("\n=== Test User Login ===")
            print(f"Email: {email}")
            print(f"Database: {DATABASE_URL}")
            print("====================\n")
        
        # Create access token
        access_token = create_access_token(data={"sub": email})
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "email": email
        }
    
    except Exception as e:
        print(f"\n=== Test Login Error ===")
        print(f"Error: {str(e)}")
        print("=====================\n")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.get("/me")
async def get_current_user_info(
    current_user_email: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get current user information"""
    user = UserCRUD.get_user_by_email(db, current_user_email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return {
        "email": user.email,
        "created_at": user.created_at,
        "last_login": user.last_login,
        "storage_quota": user.storage_quota,
        "is_active": user.is_active
    }

@router.post("/check-token")
async def check_token_validity(
    current_user_email: str = Depends(get_current_user)
):
    """Check if the current token is valid"""
    return {"valid": True, "email": current_user_email}

@router.post("/google-login")
async def google_login(token: str, db: Session = Depends(get_db)):
    try:
        # Verify Google token
        idinfo = id_token.verify_oauth2_token(
            token, requests.Request(), "687096586613-6o5s1hp797gu2q8jbe4bv1d2k1dbdq0j.apps.googleusercontent.com"
        )

        if idinfo['iss'] not in ['accounts.google.com', 'https://accounts.google.com']:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid issuer"
            )

        # Get or create user
        user = UserCRUD.get_user_by_email(db, idinfo['email'])
        if not user:
            user = UserCRUD.create_user(db, idinfo['email'])
        else:
            UserCRUD.update_last_login(db, idinfo['email'])

        # Create access token
        access_token = create_access_token(data={"sub": user.email})
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "email": user.email,
                "created_at": user.created_at,
                "last_login": user.last_login,
                "storage_quota": user.storage_quota,
                "is_active": user.is_active
            }
        }
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, "4N6FWci9s7iFIftc", algorithm="HS256")
    return encoded_jwt
