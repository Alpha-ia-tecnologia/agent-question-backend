"""
Gerenciador de Progresso para Geração de Questões.

Sistema de eventos em tempo real para acompanhar o fluxo
de geração multi-agente via SSE (Server-Sent Events).

Emite eventos granulares para cada sub-etapa dentro dos agentes,
criando um timeline cronológico detalhado.
"""

import asyncio
import json
import logging
import threading
import time
from typing import AsyncGenerator, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Sentinel to signal stream termination via queue
_STREAM_END = object()

# Variável thread-local para acessar o progress manager de dentro dos agentes
_current_progress = threading.local()


def get_current_progress() -> Optional['ProgressManager']:
    """Retorna o ProgressManager da thread atual, se existir."""
    return getattr(_current_progress, 'manager', None)


def set_current_progress(manager: Optional['ProgressManager']):
    """Define o ProgressManager para a thread atual."""
    _current_progress.manager = manager


class ProgressManager:
    """
    Gerencia o estado de progresso e emite eventos SSE com sub-etapas detalhadas.
    
    Thread-safe: os agentes podem emitir eventos de qualquer thread.
    
    Uses queue-based termination instead of a flag to avoid race conditions
    between call_soon_threadsafe and the stream loop.
    """
    
    def __init__(self):
        self._queue = asyncio.Queue()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._finished = False
        self._start_time = time.time()
        self._event_count = 0
        self._thread: Optional[threading.Thread] = None
    
    def set_loop(self, loop: asyncio.AbstractEventLoop):
        """Define o event loop para enfileirar de forma thread-safe."""
        self._loop = loop
    
    def _emit(self, data: dict):
        """Enfileira um evento de forma thread-safe."""
        self._event_count += 1
        event = {
            "seq": self._event_count,
            "elapsed": round(time.time() - self._start_time, 1),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            **data
        }
        self._put_in_queue(event)
    
    def _put_in_queue(self, item):
        """Thread-safe queue put."""
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._queue.put_nowait, item)
        else:
            try:
                self._queue.put_nowait(item)
            except Exception:
                pass
    
    # ─── Eventos de alto nível (fases) ─────────────────────────────
    
    def phase_start(self, phase_id: str, label: str, icon: str = "⏳"):
        """Marca o início de uma fase principal do pipeline."""
        self._emit({
            "type": "phase_start",
            "phase_id": phase_id,
            "label": label,
            "icon": icon
        })
    
    def phase_end(self, phase_id: str, summary: str = ""):
        """Marca o fim de uma fase principal."""
        self._emit({
            "type": "phase_end",
            "phase_id": phase_id,
            "summary": summary
        })
    
    # ─── Eventos granulares (sub-etapas dentro dos agentes) ────────
    
    def log(self, phase_id: str, message: str, detail: str = "", icon: str = "•"):
        """Emite um evento de log cronológico dentro de uma fase."""
        self._emit({
            "type": "log",
            "phase_id": phase_id,
            "message": message,
            "detail": detail,
            "icon": icon
        })
    
    def metric(self, phase_id: str, label: str, value, icon: str = "📊"):
        """Emite uma métrica (score, contagem, etc)."""
        self._emit({
            "type": "metric",
            "phase_id": phase_id,
            "label": label,
            "value": value,
            "icon": icon
        })
    
    def retry(self, attempt: int, reason: str):
        """Notifica que houve retry no ciclo de qualidade."""
        self._emit({
            "type": "retry",
            "attempt": attempt,
            "reason": reason
        })
    
    # ─── Eventos finais ────────────────────────────────────────────
    
    def finish(self, result: dict):
        """Marca o processo como concluído com sucesso."""
        self._emit({
            "type": "finished",
            "total_time": round(time.time() - self._start_time, 1),
            "questions_count": len(result.get("questions", [])),
            "quality_score": result.get("quality_score", 0),
            "retry_count": result.get("retry_count", 0),
            "result": result
        })
        self._finished = True
        # Send sentinel AFTER the finished event is queued
        self._put_in_queue(_STREAM_END)
    
    def error(self, message: str):
        """Emite evento de erro fatal."""
        self._emit({
            "type": "error",
            "message": message,
            "total_time": round(time.time() - self._start_time, 1)
        })
        self._finished = True
        # Send sentinel AFTER the error event is queued
        self._put_in_queue(_STREAM_END)
    
    # ─── Stream SSE ────────────────────────────────────────────────
    
    async def stream(self) -> AsyncGenerator[str, None]:
        """
        Gera eventos SSE formatados para o cliente.
        
        Uses queue-based termination: waits for _STREAM_END sentinel
        instead of checking a flag, eliminating race conditions.
        
        Yields:
            Strings formatadas no padrão SSE (data: {...}\n\n)
        """
        # Salva o loop para uso thread-safe
        self._loop = asyncio.get_event_loop()
        
        max_timeout_count = 150  # 150 * 2s = 5 min safety limit
        timeout_count = 0
        
        while True:
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=2.0)
                
                # Check for termination sentinel
                if item is _STREAM_END:
                    break
                
                # Normal event — yield as SSE
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
                timeout_count = 0  # reset on successful event
                    
            except asyncio.TimeoutError:
                timeout_count += 1
                # Check if thread died without signaling
                if self._thread and not self._thread.is_alive() and self._queue.empty():
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Generation thread ended unexpectedly'})}\n\n"
                    break
                if timeout_count >= max_timeout_count:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Generation timed out'})}\n\n"
                    break
                yield f"data: {json.dumps({'type': 'heartbeat', 'elapsed': round(time.time() - self._start_time, 1)})}\n\n"
        
        # Drain any remaining events
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                if item is _STREAM_END:
                    continue
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
            except asyncio.QueueEmpty:
                break
