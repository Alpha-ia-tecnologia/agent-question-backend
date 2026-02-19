"""
Database Connection Module.

Configuração de conexão com banco de dados.
Suporta MySQL (produção) e SQLite (desenvolvimento local).
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from dotenv import load_dotenv
import os

load_dotenv()

# Obtém DATABASE_URL do ambiente (obrigatório)
DATABASE_URL = os.getenv('DATABASE_URL')

if DATABASE_URL is None:
    raise RuntimeError("❌ DATABASE_URL não configurada. Defina a variável no .env")

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=1800,
    pool_size=5,
    max_overflow=10,
)


def get_session():
    """Gera uma sessão de banco de dados."""
    with Session(engine) as session:
        yield session


def init_db():
    """Inicializa o banco de dados criando todas as tabelas."""
    from app.models.table_resgitry import table_registry
    # Importa modelos para registro no metadata
    from app.models.user_model import User  # noqa: F401
    from app.models.question_model import Question, Alternative, GenerationHistory, QuestionGroup  # noqa: F401
    
    table_registry.metadata.create_all(bind=engine)
    print("✅ Banco de dados inicializado")


from contextlib import contextmanager

@contextmanager
def get_session_context():
    """Context manager para sessão de banco fora do FastAPI DI."""
    session = Session(engine)
    try:
        yield session
    finally:
        session.close()