from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import func, String
from datetime import datetime
from app.models.table_resgitry import table_registry


@table_registry.mapped_as_dataclass
class User:

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(init=False, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True)
    name: Mapped[str]  = mapped_column(String(255))
    password: Mapped[str] = mapped_column(String(255))
    is_admin: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(init=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(init=False, server_default= func.now(), onupdate=func.now())

    
