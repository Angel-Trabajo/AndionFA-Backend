"""
Engine Manager — gestiona el TradingServer como proceso de fondo.
Captura todos los print() del servidor y los expone como log buffer.
"""

import sys
import threading
from collections import deque

# ---------------------------------------------------------------------------
# Log capture — intercepta sys.stdout para capturar los prints del servidor
# ---------------------------------------------------------------------------

class _TeeStream:
    """Escribe en stdout real Y en el buffer de logs."""

    def __init__(self, real_stream):
        self._real = real_stream
        self._lock = threading.Lock()

    def write(self, msg):
        self._real.write(msg)
        stripped = msg.strip()
        if stripped:
            with _LOG_LOCK:
                _LOG_BUFFER.append(stripped)

    def flush(self):
        self._real.flush()

    def isatty(self):
        return False


_LOG_BUFFER: deque = deque(maxlen=1000)
_LOG_LOCK = threading.Lock()

# Activar captura de logs al importar este módulo
if not isinstance(sys.stdout, _TeeStream):
    sys.stdout = _TeeStream(sys.__stdout__)


# ---------------------------------------------------------------------------
# Estado global del servidor
# ---------------------------------------------------------------------------

_server_instance = None
_server_thread: threading.Thread | None = None


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def start() -> dict:
    global _server_instance, _server_thread

    if _server_thread is not None and _server_thread.is_alive():
        return {"status": "already_running"}

    try:
        from src.scripts.principal_script import TradingServer
        _server_instance = TradingServer()
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}

    _server_thread = threading.Thread(
        target=_server_instance.run,
        daemon=True,
        name="TradingServer",
    )
    _server_thread.start()
    return {"status": "started", "engines": len(_server_instance.engines)}


def stop() -> dict:
    global _server_instance
    if _server_instance is None or not (_server_thread and _server_thread.is_alive()):
        return {"status": "not_running"}
    _server_instance.stop()
    return {"status": "stopping"}


def get_status() -> dict:
    running = _server_thread is not None and _server_thread.is_alive()
    engines = []
    if _server_instance:
        engines = [
            {
                "symbol": e.principal_symbol,
                "mercado": e.mercado,
                "algo": e.algorithm,
                "is_open": e.is_open,
            }
            for e in _server_instance.engines
        ]
    return {
        "running": running,
        "engine_count": len(engines),
        "engines": engines,
    }


def get_logs(since: int = 0) -> list[str]:
    with _LOG_LOCK:
        logs = list(_LOG_BUFFER)
    return logs[since:]


# ---------------------------------------------------------------------------
# Auto-backup (cada 6 horas)
# ---------------------------------------------------------------------------

_auto_backup_thread: threading.Thread | None = None


def _auto_backup_loop() -> None:
    import time
    INTERVAL = 6 * 60 * 60  # 6 horas
    while True:
        time.sleep(INTERVAL)
        try:
            from src.db.backup_db import create_backup
            file_name = create_backup()
            print(f"[AutoBackup] Backup creado: {file_name}")
        except Exception as exc:
            print(f"[AutoBackup] Error: {exc}")


def _start_auto_backup() -> None:
    global _auto_backup_thread
    if _auto_backup_thread is None or not _auto_backup_thread.is_alive():
        _auto_backup_thread = threading.Thread(
            target=_auto_backup_loop, daemon=True, name="AutoBackup"
        )
        _auto_backup_thread.start()


def start_background_services() -> None:
    _start_auto_backup()
