from pwdlib import PasswordHash
from jwt import encode, decode
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
import os
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session
from app.utils.connect_db import get_session
from app.repositories.user_repository import UserRepository
from http import HTTPStatus

load_dotenv()


pwd_context = PasswordHash.recommended()


def hash_password(password: str):
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)

def create_acess_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(tz=ZoneInfo("UTC"))+ timedelta(minutes=int(os.getenv("ACCESS_TOKEN_EXPIRES_MINUTES")))
    to_encode.update({"exp": expire})

    generate_jwt = encode(to_encode, os.getenv("SECRET_KEY"), os.getenv("ALGORITHM"))

    return generate_jwt 


security = HTTPBearer()

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    session: Session = Depends(get_session)
):
    credentials_exception = HTTPException(
        status_code=HTTPStatus.UNAUTHORIZED,
        detail="",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = decode(credentials.credentials, os.getenv("SECRET_KEY"), algorithms=[os.getenv("ALGORITHM")])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except:
        raise credentials_exception
    
    user_repository = UserRepository(session)
    user = user_repository.find_by_email(email)
    if user is None:
        raise credentials_exception
    
    return user

def get_admin_user(current_user = Depends(get_current_user)):
    if not current_user.is_admin:
        raise HTTPException(
            status_code=HTTPStatus.FORBIDDEN,
            detail="Not enough permissions. Admin access required.",
        )
    return current_user