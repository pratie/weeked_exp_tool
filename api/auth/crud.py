from sqlalchemy.orm import Session
from datetime import datetime
from db.models import User

class UserCRUD:
    @staticmethod
    def get_user_by_email(db: Session, email: str) -> User:
        return db.query(User).filter(User.email == email).first()
    
    @staticmethod
    def create_user(db: Session, email: str) -> User:
        user = User(
            email=email,
            last_login=datetime.utcnow(),
            storage_quota=5368709120,  # 5GB
            is_active=True
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    
    @staticmethod
    def update_last_login(db: Session, email: str) -> User:
        user = UserCRUD.get_user_by_email(db, email)
        if user:
            user.last_login = datetime.utcnow()
            db.commit()
            db.refresh(user)
        return user
