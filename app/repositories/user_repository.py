from sqlalchemy import select
from sqlalchemy.orm import Session
from app.models.user_model import User


class UserRepository:
    def __init__(self, session: Session):
        self.session = session

    def list(self, offset, limit):
        return self.session.scalars(select(User).limit(limit).offset(offset))
    
    def find_by_id(self, user_id: int):
        return self.session.scalar(select(User).where(User.id == user_id))

    def create(self, user:User):
        try:
            self.session.add(user)
            self.session.commit()
            self.session.refresh(user)
            return user
        except Exception as e:
            self.session.rollback()
            raise Exception(f"Error in create user: {str(e)}")
        
    def find_by_email(self, email: str):
        return self.session.scalar(
            select(User)
            .where(User.email == email)
        )

    def update(self, user: User):
        try:
            self.session.commit()
            self.session.refresh(user)
            return user
        except Exception as e:
            self.session.rollback()
            raise e
        
    def delete(self, user: User):
        try:
            self.session.delete(user)
        except Exception as e:
            self.session.rollback()
            raise e 
    

        


