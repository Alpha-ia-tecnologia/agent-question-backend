from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from http import HTTPStatus
from app.schemas.auth_schema import UserAuthSchema, UserAuthTokenSchema
from app.utils.connect_db import get_session
from app.services.auth_service import AuthService
from app.repositories.user_repository import UserRepository

auth_router = APIRouter(prefix = "/auth")


def get_auth_service(session: Session = Depends(get_session)) -> AuthService:
    user_repository = UserRepository(session)
    return AuthService(user_repository)

@auth_router.post("/login", status_code=HTTPStatus.OK, response_model=UserAuthTokenSchema)
def login(credentials: UserAuthSchema, auth_service: AuthService = Depends(get_auth_service)):
    try:
        token = auth_service.login(credentials)
        return {
            "token_type": "Bearer",
            "token": token
        }
    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.UNAUTHORIZED, detail=str(e))



