from app.repositories.user_repository import UserRepository
from app.utils.security import hash_password
from app.schemas.user_schema import UserPublic, UpdateUserSchema
from app.models.user_model import User
class UserService:
    def __init__(self, user_repository: UserRepository):
        self.user_repository = user_repository

    def list(self, offset, limit):
        return self.user_repository.list(offset, limit)

    def create(self, user: UserPublic):
        exist_user = self.user_repository.find_by_email(user.email)
        if exist_user:
            raise Exception("Email already register")
        new_user = User(
            email = user.email,
            password = hash_password(user.password),
            name = user.name,
            is_admin= user.is_admin
        )
        return self.user_repository.create(new_user)

    def update(self, user_id: int, user: UpdateUserSchema):
        exist_user = self.user_repository.find_by_id(user_id)
        if not exist_user:
            raise Exception(f"User not found with id: {user_id}")
        
        update_data = user.model_dump(exclude_unset=True)

        if "password" in update_data.keys():
            update_data["password"] = hash_password(update_data["password"])

        for field, value in update_data.items():
            setattr(exist_user, field, value)
        
        return self.user_repository.update(exist_user)


    def delete(self, user_id: int):
        exist_user = self.user_repository.find_by_id(user_id)
        if not exist_user:
            raise Exception(f"User not found with id: {user_id}")
        self.user_repository.update(exist_user)
        

