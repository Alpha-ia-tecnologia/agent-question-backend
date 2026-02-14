from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.routes.agent_route import agent_router
from app.routes.doc_routes import doc_router
from app.routes.user_routes import user_router
from app.routes.auth_routes import auth_router
from app.routes.question_routes import question_router, group_router
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Eventos de startup e shutdown da aplicação."""
    # Startup: Inicializa o banco de dados
    from app.utils.connect_db import init_db
    init_db()
    yield
    # Shutdown: cleanup se necessário


app = FastAPI(
    title="Agent Question API",
    description="API para interagir com o agente gerador de questões educacionais",
    version="1.0.0",
    lifespan=lifespan
)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000,https://gerador-questoes-frontend.gkgtsp.easypanel.host").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(auth_router, tags=["Autenticação"])
app.include_router(user_router, tags=["Usuário"])
app.include_router(agent_router, tags=["Agente de questões"])
app.include_router(doc_router,  tags=["Documentos"])
app.include_router(question_router, tags=["Questões"])
app.include_router(group_router, tags=["Grupos de Questões"])

# Monta diretório estático para servir imagens
STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")
os.makedirs(os.path.join(STATIC_DIR, "images"), exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

