from app.repositories.user_repository import UserRepository
from app.schemas.auth_schema import UserAuthSchema
from app.utils.security import verify_password,  create_acess_token

class AuthService:
    def __init__(self, user_repository: UserRepository):
        self.user_repository = user_repository

    def login(self, user: UserAuthSchema):
        exist_user = self.user_repository.find_by_email(user.email)
        if not exist_user:
            raise Exception("User or password incorrect")
        macth_password = verify_password(user.password, exist_user.password)
        if not macth_password:
            raise Exception("User or password incorrect")
        
        token_jwt = create_acess_token({"sub": user.email, "is_admin": exist_user.is_admin})
        return token_jwt
        



