from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from db.config import get_db
from .crud import UserCRUD

# Using HTTPBearer instead of OAuth2PasswordBearer for simpler token handling
security = HTTPBearer()

async def get_current_user(token = Depends(security), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token.credentials, "4N6FWci9s7iFIftc", algorithms=["HS256"])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = UserCRUD.get_user_by_email(db, email)
    if user is None:
        raise credentials_exception
    return user.email
