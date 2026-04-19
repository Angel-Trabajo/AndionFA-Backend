"""
Engine Manager — gestiona el TradingServer como proceso de fondo.
Captura todos los print() del servidor y los expone como log buffer.
"""

import sys
import json
import threading
from pathlib import Path
from collections import deque

from src.utils.common_functions import evaluate_live_strategy_filter

PATH_GENERAL_CONFIG = Path("config/general_config.json")
PATH_LIVE_CONFIG = Path("config/live_config.json")
DEFAULT_LOT_SIZE = 0.01


def build_engine_id(symbol: str, mercado: str, algorithm: str) -> str:
    return f"{symbol}|{mercado}|{algorithm}"


def _default_live_payload() -> dict:
    return {
        "live": {
            "winrate": 0.45,
            "profit_factor": 1.2,
            "expectancy": 1.5,
            "probabilidad": 0.55,
            "cantidad_operaciones": 30,
            "lot_sizes": {},
            "filtered_algorithms": [],
            "last_filter_applied_at": None,
        }
    }


def _load_live_payload() -> dict:
    if not PATH_LIVE_CONFIG.exists():
        payload = _default_live_payload()
        PATH_LIVE_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        with open(PATH_LIVE_CONFIG, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)
        return payload

    with open(PATH_LIVE_CONFIG, "r", encoding="utf-8") as f:
        data = json.load(f)

    payload = _default_live_payload()
    payload["live"].update(data.get("live", {}))
    return payload


def _save_live_payload(payload: dict) -> None:
    PATH_LIVE_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    with open(PATH_LIVE_CONFIG, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)


def _load_general_config() -> dict:
    with open(PATH_GENERAL_CONFIG, "r", encoding="utf-8") as f:
        return json.load(f)

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
        live_payload = _load_live_payload()
        live = live_payload.get("live", {})
        filtered_algorithms = live.get("filtered_algorithms", [])
        lot_sizes = live.get("lot_sizes", {})
        _server_instance = TradingServer(
            filtered_algorithms=filtered_algorithms,
            lot_sizes=lot_sizes,
        )
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}

    _server_thread = threading.Thread(
        target=_server_instance.run,
        daemon=True,
        name="TradingServer",
    )
    _server_thread.start()
    return {"status": "started", "engines": len(_server_instance.engines)}


def stop(mode: str = "graceful") -> dict:
    global _server_instance
    if mode not in {"graceful", "immediate"}:
        return {"status": "error", "detail": "mode inválido"}
    if _server_instance is None or not (_server_thread and _server_thread.is_alive()):
        return {"status": "not_running"}
    _server_instance.stop(mode=mode)
    return {"status": "stopping", "mode": mode}


def get_status() -> dict:
    running = _server_thread is not None and _server_thread.is_alive()
    engines = []
    collective = None
    if _server_instance:
        stats = _server_instance.collect_stats()
        engines = stats.get("engines", [])
        collective = stats.get("collective")
    return {
        "running": running,
        "engine_count": len(engines),
        "engines": engines,
        "collective": collective,
    }


def get_live_config() -> dict:
    payload = _load_live_payload()
    live = payload.get("live", {})
    return {
        "live": {
            "winrate": live.get("winrate", 0),
            "profit_factor": live.get("profit_factor", 0),
            "expectancy": live.get("expectancy", 0),
            "probabilidad": live.get("probabilidad", 0),
            "cantidad_operaciones": live.get("cantidad_operaciones", 0),
            "lot_sizes": live.get("lot_sizes", {}),
            "filtered_algorithms": live.get("filtered_algorithms", []),
            "last_filter_applied_at": live.get("last_filter_applied_at"),
        }
    }


def update_live_filters(filters: dict) -> dict:
    payload = _load_live_payload()
    live = payload.setdefault("live", {})

    for key in ["winrate", "profit_factor", "expectancy", "probabilidad"]:
        if key in filters:
            live[key] = float(filters[key])
    if "cantidad_operaciones" in filters:
        live["cantidad_operaciones"] = int(filters["cantidad_operaciones"])

    _save_live_payload(payload)
    return get_live_config()


def apply_live_filter() -> dict:
    payload = _load_live_payload()
    live = payload.setdefault("live", {})
    general = _load_general_config()
    list_principal_symbols = general.get("list_principal_symbols", [])

    # Al aplicar filtro, el mapa de lotes se reconstruye SOLO con los que pasen.
    # Todos arrancan en 0.01 y luego el usuario puede cambiarlos en la tabla.
    lot_sizes: dict[str, float] = {}
    filtered = []
    total_checked = 0
    list_mercados = ["Asia", "Europa", "America"]
    list_algorithms = ["UP", "DOWN"]

    for principal_symbol in list_principal_symbols:
        for mercado in list_mercados:
            for algorithm in list_algorithms:
                score_path = (
                    Path("output")
                    / principal_symbol
                    / "data_for_neuronal"
                    / "best_score"
                    / f"score_{mercado}_{algorithm}.json"
                )
                if not score_path.exists():
                    continue

                total_checked += 1
                with open(score_path, "r", encoding="utf-8") as f:
                    score_data = json.load(f)
                metrics = score_data.get("metrics", {})
                evaluation = evaluate_live_strategy_filter(metrics, live_config=live)
                if not evaluation["passed"]:
                    continue

                engine_id = build_engine_id(principal_symbol, mercado, algorithm)
                lot_size = DEFAULT_LOT_SIZE
                lot_sizes[engine_id] = lot_size
                filtered.append(
                    {
                        "engine_id": engine_id,
                        "symbol": principal_symbol,
                        "mercado": mercado,
                        "algorithm": algorithm,
                        "lot_size": lot_size,
                        "metrics": {
                            "winrate": float(metrics.get("winrate", 0)),
                            "profit_factor": float(metrics.get("profit_factor", 0)),
                            "expectancy": float(metrics.get("expectancy", 0)),
                            "cantidad_operaciones": int(metrics.get("cantidad_operaciones", 0)),
                            "probabilidad": float(evaluation.get("probabilidad", 0)),
                        },
                    }
                )

    from datetime import datetime
    live["filtered_algorithms"] = filtered
    live["last_filter_applied_at"] = datetime.utcnow().isoformat()
    live["lot_sizes"] = lot_sizes
    _save_live_payload(payload)

    return {
        "status": "ok",
        "checked": total_checked,
        "passed": len(filtered),
        "filtered_algorithms": filtered,
        "last_filter_applied_at": live["last_filter_applied_at"],
    }


def set_engine_lot_size(engine_id: str, lot_size: float) -> dict:
    payload = _load_live_payload()
    live = payload.setdefault("live", {})
    lot_sizes = live.setdefault("lot_sizes", {})
    lot_sizes[engine_id] = float(lot_size)

    # Mantener sincronizado el snapshot de filtrados.
    for item in live.get("filtered_algorithms", []):
        if item.get("engine_id") == engine_id:
            item["lot_size"] = float(lot_size)
            break

    _save_live_payload(payload)

    if _server_instance and _server_thread and _server_thread.is_alive():
        _server_instance.update_engine_lot_size(engine_id, float(lot_size))

    return {"status": "ok", "engine_id": engine_id, "lot_size": float(lot_size)}


def stop_engine(engine_id: str, mode: str) -> dict:
    if mode not in {"graceful", "immediate"}:
        return {"status": "error", "detail": "mode inválido"}
    if _server_instance is None or not (_server_thread and _server_thread.is_alive()):
        return {"status": "not_running"}
    return _server_instance.stop_engine(engine_id, mode)


def get_live_stats() -> dict:
    if _server_instance is None:
        return {"running": False, "collective": None, "engines": []}
    stats = _server_instance.collect_stats()
    return {
        "running": _server_thread is not None and _server_thread.is_alive(),
        "collective": stats.get("collective"),
        "engines": stats.get("engines", []),
    }


def get_logs(since: int = 0) -> list[str]:
    with _LOG_LOCK:
        logs = list(_LOG_BUFFER)
    return logs[since:]


