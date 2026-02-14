from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os


load_dotenv()

engine = create_engine(os.getenv('DATABASE_URL'))


with engine.connect() as conn:
    result = conn.execute(
        text("SELECT COUNT(*) FROM users WHERE email = :email"),
        {"email": "alpha.ia.tecnologia@gmail.com"}
    )
    user_count = result.scalar()
    
    if user_count == 0:
        conn.execute(
                text("""
                INSERT INTO users (name, email, password, is_admin)
                VALUES (:name, :email, :password, :is_admin)
                """),
                {
                    "name": "Alpha ADM",
                    "email": "alpha.ia.tecnologia@gmail.com",
                    "password": "$argon2id$v=19$m=65536,t=3,p=4$20IkzJVs2flJSfoNVGCiMw$2nCmboSUEjmphA8SZZGswCooiq9pOW+EHdkEJmgOIDs",
                    "is_admin": True,

                }
            )
        conn.commit()
        print("✅ Admin user created successfully!")
    else:
        print("ℹ️ Admin user already exists!")

