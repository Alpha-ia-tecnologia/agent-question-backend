from fastapi import APIRouter, Depends, HTTPException
from http import HTTPStatus
from app.schemas.user_schema import UserPublic, ListUserResponse, UserResponse, UpdateUserSchema
from app.schemas.message_schema import MessageSchema
from app.repositories.user_repository import UserRepository
from app.services.user_service import UserService
from app.utils.connect_db import get_session
from sqlalchemy.orm import Session


def get_user_service(session: Session = Depends(get_session)) -> UserService:
    user_repository = UserRepository(session)
    return UserService(user_repository)

user_router = APIRouter(prefix="/user")


@user_router.get("/", status_code=HTTPStatus.OK, response_model=ListUserResponse)
def list_user(offset: int = 0, limit: int = 10, user_service: UserService = Depends(get_user_service)):
    users = user_service.list(offset, limit)
    return {
        "users": users
    }


@user_router.post("/", status_code=HTTPStatus.CREATED, response_model=UserResponse)
def create_user(user: UserPublic, user_service: UserService = Depends(get_user_service)):
    try:
        new_user = user_service.create(user)
        return {
            "id": new_user.id,
            "name": new_user.name,
            "email": new_user.email,
            "is_admin": new_user.is_admin
        }
    except Exception as e:
       raise HTTPException(HTTPStatus.BAD_REQUEST, detail=str(e))
    
@user_router.put("/{user_id}", response_model=UserResponse)
def update_user(user_id: int, user: UpdateUserSchema, user_service: UserService = Depends(get_user_service)):
    try:
        updated_user = user_service.update(user_id, user)
        return {
            "id": updated_user.id,
            "name": updated_user.name,
            "email": updated_user.email,
            "is_admin": updated_user.is_admin
        }
    except Exception as e:
        raise HTTPException(HTTPStatus.BAD_REQUEST,detail=str(e))


@user_router.delete("/{user_id}")
def delete_user(user_id: int, user_service: UserService = Depends(get_user_service)):
    try:
        user_service.delete(user_id)
        return {
            "message": "User deleted with success"
        }
    except Exception as e:
        raise HTTPException(status_code= HTTPStatus.BAD_REQUEST, detail=str(e))
    