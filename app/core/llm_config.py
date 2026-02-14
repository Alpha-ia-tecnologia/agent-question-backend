"""
LLM Configuration Module - Configuração centralizada dos modelos de IA.

Este módulo fornece:
- Configuração centralizada dos modelos LLM
- Factory functions para criar instâncias dos modelos (OpenAI e Google Gemini)
- Tratamento de erros customizado
- Configuração de retry e rate limiting
- Callbacks para observabilidade
"""

from langchain_google_genai import ChatGoogleGenerativeAI, Modality
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from typing import Optional, List, Any
from dotenv import load_dotenv
from functools import lru_cache
import logging
import os

load_dotenv()

# Configuração de logging (Reload trigger)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Keys — carregadas exclusivamente de variáveis de ambiente
OPENAI_API_KEY_OVERRIDE = os.getenv("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# ============================================
# Custom Exceptions
# ============================================

class LLMError(Exception):
    """Erro base para operações com LLM."""
    pass


class QuestionGenerationError(LLMError):
    """Erro durante geração de questões."""
    pass


class ImageGenerationError(LLMError):
    """Erro durante geração de imagens."""
    pass


class ConfigurationError(LLMError):
    """Erro de configuração do LLM."""
    pass


# ============================================
# Configuration Classes
# ============================================

class LLMSettings(BaseModel):
    """Configurações do modelo LLM."""
    
    model: str = Field(description="Nome do modelo")
    temperature: float = Field(default=1.0, description="Temperatura para geração")
    max_retries: int = Field(default=3, description="Número máximo de tentativas")
    timeout: Optional[float] = Field(default=60.0, description="Timeout em segundos")
    
    class Config:
        frozen = True


class QuestionLLMSettings(LLMSettings):
    """Configurações específicas para geração de questões."""
    
    # DeepSeek para geração de questões
    model: str = Field(default="deepseek-chat")
    temperature: float = Field(default=0.7)
    timeout: float = Field(default=120.0)


class ImageLLMSettings(LLMSettings):
    """Configurações específicas para geração de imagens."""
    
    # Gemini 3 Pro Image Preview (Nano Banana Pro)
    model: str = Field(default="gemini-3-pro-image-preview")
    temperature: float = Field(default=1.0)


# ============================================
# Logging Callback Handler
# ============================================

class LoggingCallbackHandler(BaseCallbackHandler):
    """Callback handler para logging de operações do LLM."""
    
    def on_llm_start(self, serialized: dict, prompts: List[str], **kwargs: Any) -> None:
        logger.info(f"🚀 LLM iniciando - Model: {serialized.get('name', 'unknown')}")
        logger.debug(f"Prompts: {prompts[:100]}...")  # Log primeiros 100 chars
    
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        logger.info("✅ LLM finalizou com sucesso")
    
    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        logger.error(f"❌ Erro no LLM: {error}")
    
    def on_chain_start(self, serialized: dict, inputs: dict, **kwargs: Any) -> None:
        logger.info(f"⛓️ Chain iniciando: {serialized.get('name', 'unknown')}")
    
    def on_chain_end(self, outputs: dict, **kwargs: Any) -> None:
        logger.info("✅ Chain finalizada com sucesso")
    
    def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        logger.error(f"❌ Erro na chain: {error}")


# ============================================
# Factory Functions
# ============================================

def _get_api_key(type: str = "google") -> str:
    """Obtém a API key do ambiente ou override."""
    if type == "openai":
        return OPENAI_API_KEY_OVERRIDE or os.getenv("OPENAI_API_KEY")
    
    if type == "deepseek":
        return DEEPSEEK_API_KEY or os.getenv("DEEPSEEK_API_KEY")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key and type == "google":
        # Fallback opcional ou erro
        pass 
    return api_key


def _create_llm(
    settings: LLMSettings,
    response_modalities: Optional[List[Modality]] = None,
    callbacks: Optional[List[BaseCallbackHandler]] = None
) -> BaseChatModel:
    """
    Factory function para criar instância do LLM (OpenAI ou Gemini).
    """
    callbacks = callbacks or [LoggingCallbackHandler()]
    
    # Suporte a DeepSeek (API compatível com OpenAI)
    if "deepseek" in settings.model.lower():
        api_key = _get_api_key("deepseek")
        if not api_key:
            raise ConfigurationError("DEEPSEEK_API_KEY não encontrada.")
            
        logger.info(f"📦 Criando DeepSeek LLM: {settings.model}")
        return ChatOpenAI(
            model=settings.model,
            temperature=settings.temperature,
            api_key=api_key,
            base_url=DEEPSEEK_BASE_URL,
            max_retries=settings.max_retries,
            timeout=settings.timeout,
            callbacks=callbacks
        )
    
    # Suporte a OpenAI
    if "gpt" in settings.model.lower():
        api_key = _get_api_key("openai")
        if not api_key:
            raise ConfigurationError("OPENAI_API_KEY não encontrada.")
            
        logger.info(f"📦 Criando OpenAI LLM: {settings.model}")
        return ChatOpenAI(
            model=settings.model,
            temperature=settings.temperature,
            api_key=api_key,
            max_retries=settings.max_retries,
            timeout=settings.timeout,
            callbacks=callbacks
        )
    
    # Fallback para Gemini (Google)
    api_key = _get_api_key("google")
    if not api_key:
         raise ConfigurationError("GOOGLE_API_KEY não encontrada.")

    kwargs = {
        "model": settings.model,
        "temperature": settings.temperature,
        "api_key": api_key,
        "max_retries": settings.max_retries,
        "timeout": settings.timeout,
        "callbacks": callbacks,
    }
    
    if response_modalities:
        kwargs["response_modalities"] = response_modalities
    
    logger.info(f"📦 Criando Gemini LLM: {settings.model}")
    return ChatGoogleGenerativeAI(**kwargs)


@lru_cache(maxsize=1)
def get_question_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None
) -> BaseChatModel:
    """
    Obtém instância do LLM para geração de questões.
    """
    settings = QuestionLLMSettings(
        model=model or QuestionLLMSettings().model,
        temperature=temperature if temperature is not None else QuestionLLMSettings().temperature
    )
    
    return _create_llm(settings)


def get_image_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None
) -> BaseChatModel:
    """
    Obtém instância do LLM para geração de imagens (Mantido no Gemini por enquanto).
    """
    settings = ImageLLMSettings(
        model=model or ImageLLMSettings().model,
        temperature=temperature if temperature is not None else ImageLLMSettings().temperature
    )
    
    return _create_llm(
        settings,
        response_modalities=[Modality.TEXT, Modality.IMAGE]
    )


def get_runnable_config(
    run_name: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> RunnableConfig:
    """Cria configuração para execução de runnables."""
    return RunnableConfig(
        run_name=run_name,
        tags=tags or [],
        callbacks=[LoggingCallbackHandler()],
    )


# ============================================
# Retry Configuration
# ============================================

RETRY_CONFIG = {
    "stop_after_attempt": 3,
    "wait_exponential_multiplier": 1,
    "wait_exponential_min": 2,
    "wait_exponential_max": 10,
}
