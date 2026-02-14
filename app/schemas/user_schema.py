from pydantic import BaseModel, EmailStr
from typing import List, Optional

class UserPublic(BaseModel):
    name: str
    email: str
    password: str
    is_admin: bool


class UserResponse(BaseModel):
    id: int
    name: str
    email: EmailStr
    is_admin: bool

class ListUserResponse(BaseModel):
    users: List[UserResponse]


class UpdateUserSchema(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    password: Optional[str] =  None
    is_admin: Optional[bool] = None

