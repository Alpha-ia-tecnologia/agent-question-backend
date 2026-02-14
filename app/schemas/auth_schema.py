from pydantic import BaseModel
class UserAuthSchema(BaseModel):
    email: str
    password: str


class UserAuthTokenSchema(BaseModel):
    token_type: str 
    token: str