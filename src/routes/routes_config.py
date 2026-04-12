import os, sys
import json
import shutil
import sqlite3
import subprocess
import psutil
import asyncio
import threading
from collections import deque
from pathlib import Path
import time
import concurrent.futures
import multiprocessing
from datetime import datetime


from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, WebSocket, WebSocketDisconnect, UploadFile, File, Body
from typing import List

from src.routes import peticiones 
from ..models.indicators import ConfigRequest, ExecuteRequest
from src.scripts.create_indicators import create_files
from src.scripts.node_builder import execute_node_builder
from src.scripts.crossing_builder_cpu import execute_crossing_builder
from dotenv import load_dotenv
from src.utils.common_functions import crear_carpeta_si_no_existe, get_previous_4_6, should_backtest_strategy
from src.neuronal.data_para_entrenar import execute_data_for_neuronal
from src.neuronal.entrenar import execute_entrenar
from src.neuronal.backtester import Backtester
from src.db.reset_db import reset_database
from src.db.backup_db import create_backup, restore_backup, list_backups
from src.db.query import get_top_quality_nodes

BACKTEST_RESULTS_DIR = 'output/x_backtest_results'
_backtest_run_state = {
    "running": False,
    "last_error": None,
    "last_started_at": None,
    "last_finished_at": None,
}


load_dotenv()

PYTHON_PATH = os.getenv("PYTHON_PATH")  
SCRIPT_PATH = os.getenv("SCRIPT_PATH")
SCRIPT_PATH_CROSS = os.getenv("SCRIPT_PATH_CROSS")
SCRIPT_PATH_CROSS_F = os.getenv("SCRIPT_PATH_CROSS_F")

PATH_GENERAL_CONFIG = 'config/general_config.json'
PATH_BACKTEST_CONFIG = 'config/backtest_config.json'
PATH_EXTRACTOR = 'config/extractor'


router = APIRouter()
process = None
process2 = None
connected_clients: List[WebSocket] = []

# ── Execute state ──────────────────────────────────────────────────────────
# El algoritmo corre en un proceso separado para poder matarlo inmediatamente.
# El progreso se comunica via multiprocessing.Queue → hilo drenador → _progress.
_algo_process: multiprocessing.Process | None = None
_progress: dict = {"running": False, "total": 0, "done": 0, "current": None, "step": None}
_logs: list = []          # lista con los últimos _LOGS_MAXLEN mensajes
_logs_offset: int = 0    # cuántos mensajes han sido descartados desde el inicio
_LOGS_MAXLEN: int = 10_000
_progress_queue: multiprocessing.Queue = multiprocessing.Queue()


def _drain_progress_queue():
    """Daemon thread: traslada actualizaciones del proceso hijo a _progress."""
    global _progress, _logs, _logs_offset
    while True:
        try:
            update = _progress_queue.get(timeout=1)
            if "log" in update:
                line = update["log"]
                print(line, flush=True)
                _logs.append(line)
                if len(_logs) > _LOGS_MAXLEN:
                    drop = len(_logs) - _LOGS_MAXLEN
                    del _logs[:drop]
                    _logs_offset += drop
            else:
                _progress.update(update)
        except Exception:
            pass


threading.Thread(target=_drain_progress_queue, daemon=True, name="ProgressDrain").start()


def _is_symbol_done(symbol: str, list_mercado: list, list_algorithms: list) -> bool:
    """True si existen todos los resultados de backtest para el símbolo."""
    base = Path(BACKTEST_RESULTS_DIR) / symbol
    return all(
        (base / f"{m}_{a}" / "results.csv").exists()
        for m in list_mercado for a in list_algorithms
    )


def _run_backtester(args: tuple[str, str, str]) -> list[str]:
    import io, sys
    buf = io.StringIO()
    sys.stdout = sys.stderr = buf
    try:
        symbol, mercado, algorithm = args
        backtester = Backtester(symbol, mercado, algorithm)
        backtester.run()
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
    return [line for line in buf.getvalue().splitlines() if line.strip()]


def _run_backtest_script_worker() -> None:
    try:
        backend_root = Path(__file__).resolve().parents[2]
        script_path = backend_root / 'src' / 'motor_backtest' / 'backtest.py'
        print(f"Ejecutando script de backtest: {script_path}")
        completed = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(backend_root),
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.stdout:
            print(completed.stdout)
        if completed.stderr:
            print(completed.stderr)
        if completed.returncode != 0:
            raise RuntimeError(f"backtest.py finalizó con código {completed.returncode}")
        _backtest_run_state["last_error"] = None
    except Exception as exc:
        _backtest_run_state["last_error"] = str(exc)
    finally:
        _backtest_run_state["running"] = False
        _backtest_run_state["last_finished_at"] = time.time()


def _load_backtest_config() -> dict:
    default_config = {
        "backtest_config": {
            "date_start": "2025-01-01",
            "date_end": "2026-01-01",
        }
    }
    path = Path(PATH_BACKTEST_CONFIG)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf8') as file:
            json.dump(default_config, file, indent=4, ensure_ascii=False)
        return default_config["backtest_config"]

    with open(path, 'r', encoding='utf8') as file:
        data = json.load(file)
    return data.get("backtest_config", default_config["backtest_config"])


def _save_backtest_config(date_start: str, date_end: str) -> dict:
    datetime.strptime(date_start, "%Y-%m-%d")
    datetime.strptime(date_end, "%Y-%m-%d")
    payload = {
        "backtest_config": {
            "date_start": date_start,
            "date_end": date_end,
        }
    }
    path = Path(PATH_BACKTEST_CONFIG)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf8') as file:
        json.dump(payload, file, indent=4, ensure_ascii=False)
    return payload["backtest_config"]


@router.get('/extractor-files')
async def extractor_file():
    peticiones.initialize_mt5()
    list_extractor_files = os.listdir(PATH_EXTRACTOR)
    return {"list_files": list_extractor_files}


@router.post('/extractor-files')
async def extractor_file(request: list[str]):
    with open(PATH_GENERAL_CONFIG, 'r', encoding='utf8') as file:
        data = json.load(file)
    data['indicators_files'] = request
    with open(PATH_GENERAL_CONFIG, 'w', encoding='utf8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
    return {
        "status": "ok",}


@router.get('/list-symbol')
async def list_simbol():
    peticiones.initialize_mt5()
    active_symbols = peticiones.get_active_symbols()['symbols']
    return {"symbols": active_symbols}


@router.get("/general-config") 
async def getnode_config():
    peticiones.initialize_mt5()
    with open(PATH_GENERAL_CONFIG, 'r', encoding='utf8') as file:
        data = json.load(file)
    return {"data":data}   


@router.post("/general-config") 
async def postnode_config(request: ConfigRequest):
    with open(PATH_GENERAL_CONFIG, 'r', encoding='utf8') as file:
        config = json.load(file)
    
    config["list_principal_symbols"] = request.symbols
    config["timeframe"] = request.timeFrame
    config["dateStart"] = request.dateStart
    config["dateEnd"] = request.dateEnd
    config["SimilarityMax"] = request.SimilarityMax
    config["NTotal"] = request.NTotal
    config["MinOperationsIS"] = request.MinOperationsIS
    config["MinOperationsOS"] = request.MinOperationsOS
    config["NumMaxOperations"] = request.NumMaxOperations
    config["min_operaciones"] = request.MinOperations
    config["MinSuccessRate"] = request.MinSuccessRate
    config["MaxSuccessRate"] = request.MaxSuccessRate
    config["ProgressiveVariation"] = request.ProgressiveVariation
    config["MinOpenSymbolConfirmations"] = request.MinOpenSymbolConfirmations
    config["robust_trade_penalty_center"] = request.robust_trade_penalty_center
    config["lot_size"] = request.lot_size
    config["stop_loss"] = request.stop_loss
    config["take_profit"] = request.take_profit
    config["use_proces"] = request.use_proces
    
    with open(PATH_GENERAL_CONFIG, 'w', encoding='utf8') as file:
        json.dump(config, file, indent=4, ensure_ascii=False)
    
    
    return {
        "status": "ok",
        "message": "Configuracion completado",
    } 


@router.post("/execute-algorithm")
def execute_algorithm(request: ExecuteRequest):
    global _algo_process
    if _algo_process and _algo_process.is_alive():
        return {"status": "already_running"}
    # Vaciar cola y logs de la ejecución anterior
    while not _progress_queue.empty():
        try:
            _progress_queue.get_nowait()
        except Exception:
            break
    _logs.clear()
    _logs_offset = 0
    ctx = multiprocessing.get_context("spawn")
    _algo_process = ctx.Process(
        target=_run_algorithm,
        args=(request.reset_db, _progress_queue),
        daemon=False,
    )
    _algo_process.start()
    return {"status": "started"}


@router.get("/execute-progress")
def execute_progress_get():
    result = dict(_progress)
    result["logs"] = list(_logs)
    result["logs_offset"] = _logs_offset
    return result


@router.post("/execute-stop")
def execute_stop():
    global _algo_process
    if _algo_process and _algo_process.is_alive():
        try:
            # Matar el árbol completo de procesos (incluye workers del ProcessPoolExecutor)
            parent = psutil.Process(_algo_process.pid)
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            parent.kill()
        except psutil.NoSuchProcess:
            pass
        _algo_process.join(timeout=5)
    _progress.update({"running": False, "current": None, "step": None})
    return {"status": "stopped"}


class _QueueStream:
    """Redirige stdout/stderr del proceso hijo al hilo drenador del padre via Queue."""
    def __init__(self, q):
        self._q = q
        self._buf = ""
    def write(self, text):
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._q.put({"log": line})
    def flush(self):
        pass
    def fileno(self):
        import io
        raise io.UnsupportedOperation("fileno")


def _clear_directory_contents(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=False)
        else:
            child.unlink(missing_ok=True)


def _run_algorithm(reset_db: bool, q: multiprocessing.Queue):
    """Corre en un proceso separado. Usa q para enviar actualizaciones de progreso."""
    import sys, threading as _threading
    _qs = _QueueStream(q)
    sys.stdout = _qs
    sys.stderr = _qs

    def report(**kwargs):
        q.put(kwargs)

    # Manager queue — proxy picklable para streaming en tiempo real desde procesos hijos spawn
    _mgr = multiprocessing.Manager()
    _wlq = _mgr.Queue()
    _wlq_stop = _threading.Event()

    def _drain_wlq():
        while not _wlq_stop.is_set():
            try:
                line = _wlq.get(timeout=0.05)
                print(line, flush=True)
            except Exception:
                pass
        while True:
            try:
                print(_wlq.get_nowait(), flush=True)
            except Exception:
                break

    _threading.Thread(target=_drain_wlq, daemon=True).start()

    report(running=True, total=0, done=0, current=None, step=None)
    ini = time.time()

    if reset_db:
        print("reset_db=true: limpiando carpetas output y config/divisas...")
        _clear_directory_contents(Path("output"))
        _clear_directory_contents(Path("config/divisas"))
        reset_database()

    with open(PATH_GENERAL_CONFIG, 'r', encoding='utf8') as file:
        config = json.load(file)

    list_mercado = ['Asia', 'Europa', 'America']
    list_algorithms = ['UP', 'DOWN']
    list_principal_symbols = config['list_principal_symbols']
    timeframe = config['timeframe']
    date_start_os = config['dateStart']
    date_end_os = config['dateEnd']
    indicadors_files = config['indicators_files']
    date_start_is, date_end_is = get_previous_4_6(date_start_os, date_end_os)
    file_workers = int(config.get('use_proces', 12))

    if reset_db:
        symbols_to_process = list_principal_symbols
    else:
        symbols_to_process = [
            s for s in list_principal_symbols
            if not _is_symbol_done(s, list_mercado, list_algorithms)
        ]
        skipped = len(list_principal_symbols) - len(symbols_to_process)
        if skipped:
            print(f"Reanudando: {skipped} pares ya completados, faltan {len(symbols_to_process)}")

    total = len(list_principal_symbols)
    done = len(list_principal_symbols) - len(symbols_to_process)
    report(total=total, done=done)

    # ── Fase 1: crear archivos para TODOS los símbolos ──
    # Siempre lista completa: reset_db=True recalcula todo, reset_db=False asegura
    # que el crossing_builder tenga los datos de todos los pares disponibles.
    print(f"Fase 1: creando archivos de indicadores para {total} símbolos...")
    for i, symbol in enumerate(list_principal_symbols):
        report(current=symbol, step="create_files")
        print(f"[create_files {i + 1}/{total}] {symbol}...")
        crear_carpeta_si_no_existe(f'output/symbol_data/{symbol}')
        create_files(symbol, timeframe, date_start_os, date_end_os, indicadors_files, 'extrac_os', file_workers)
        create_files(symbol, timeframe, date_start_is, date_end_is, indicadors_files, 'extrac', file_workers)

    # ── Fase 2: nodos, crossing, neuronal, backtest — símbolo a símbolo ──
    print(f"Fase 2: procesando pipeline completo para {len(symbols_to_process)} símbolos...")
    for symbol in symbols_to_process:
        report(current=symbol, step="node_builder")
        print(f"[{done + 1}/{total}] {symbol} — construyendo nodos...")
        crear_carpeta_si_no_existe(f'config/divisas/{symbol}')
        crear_carpeta_si_no_existe(f'output/{symbol}')
        execute_node_builder(symbol, list_mercado, log_q=_wlq)

        report(step="crossing_builder")
        print(f"[{done + 1}/{total}] {symbol} — cruzando nodos...")
        execute_crossing_builder(symbol, list_mercado, log_q=_wlq)

        report(step="data_neuronal")
        print(f"[{done + 1}/{total}] {symbol} — preparando datos neuronal...")
        execute_data_for_neuronal(symbol, list_mercado, list_algorithms=None, dict_pips_best=None)

        report(step="entrenar")
        print(f"[{done + 1}/{total}] {symbol} — entrenando red neuronal...")
        execute_entrenar(symbol, list_mercado, list_algorithms=None)

        report(step="backtest")
        print(f"[{done + 1}/{total}] {symbol} — backtest...")
        for mercado_t in list_mercado:
            for algorithm_t in list_algorithms:
                try:
                    print(f"Backtest: {symbol} {mercado_t} {algorithm_t}...")
                    backtester = Backtester(symbol, mercado_t, algorithm_t)
                    backtester.run()
                except ValueError as exc:
                    print(f"Backtester omitido por datos insuficientes: {exc}")
                except Exception as exc:
                    print(f"Error en backtester: {exc}")

        done += 1
        report(done=done, step=None)
        print(f"[{done}/{total}] {symbol} completado ✓")

    _wlq_stop.set()
    _mgr.shutdown()
    report(running=False, current=None, step=None)
    print(f"Tiempo total de ejecución final: {time.time() - ini:.2f} segundos")


# =============================================================================
# BACKTEST RESULTS
# =============================================================================

@router.get("/backtest/list")
def backtest_list():
    """Lista los backtests disponibles: [{symbol, mercado, algo}]"""
    results = []
    base = Path(BACKTEST_RESULTS_DIR)
    if not base.exists():
        return {"data": results}
    for symbol_dir in sorted(base.iterdir()):
        if not symbol_dir.is_dir():
            continue
        for combo_dir in sorted(symbol_dir.iterdir()):
            if not combo_dir.is_dir():
                continue
            csv_path = combo_dir / "results.csv"
            if not csv_path.exists():
                continue
            parts = combo_dir.name.split("_")
            if len(parts) < 2:
                continue
            mercado, algo = parts[0], parts[1]
            results.append({
                "symbol": symbol_dir.name,
                "mercado": mercado,
                "algo": algo,
            })
    return {"data": results}


@router.get("/backtest/config")
def backtest_config_get():
    try:
        return {"data": _load_backtest_config()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backtest/config")
def backtest_config_post(payload: dict = Body(...)):
    try:
        date_start = payload.get("date_start")
        date_end = payload.get("date_end")
        if not date_start or not date_end:
            raise HTTPException(status_code=400, detail="date_start y date_end son requeridos")
        saved = _save_backtest_config(date_start=date_start, date_end=date_end)
        return {"status": "ok", "data": saved}
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(status_code=400, detail="Formato de fecha inválido, usa YYYY-MM-DD")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backtest/run")
def backtest_run():
    if _backtest_run_state["running"]:
        return {"status": "running", "detail": "Backtest ya está en ejecución"}

    _backtest_run_state["running"] = True
    _backtest_run_state["last_error"] = None
    _backtest_run_state["last_started_at"] = time.time()

    worker = threading.Thread(target=_run_backtest_script_worker, daemon=True, name="BacktestWorker")
    worker.start()

    return {"status": "started", "detail": "Backtest lanzado en segundo plano"}


@router.get("/backtest/run/status")
def backtest_run_status():
    return {
        "running": _backtest_run_state["running"],
        "last_error": _backtest_run_state["last_error"],
        "last_started_at": _backtest_run_state["last_started_at"],
        "last_finished_at": _backtest_run_state["last_finished_at"],
    }


@router.get("/backtest/equity")
def backtest_equity(
    symbol: str = Query(...),
    mercado: str = Query(...),
    algo: str = Query(...),
):
    """Devuelve la curva de equity acumulada (pips) para un backtest."""
    csv_path = Path(f"{BACKTEST_RESULTS_DIR}/{symbol}/{mercado}_{algo}/results.csv")
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Backtest no encontrado")
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        df["equity"] = df["pips"].cumsum()
        # Usar time_close como eje X; rellenar si falta
        if "time_close" in df.columns:
            df["time"] = pd.to_datetime(df["time_close"], errors="coerce").dt.strftime("%Y-%m-%d")
        else:
            df["time"] = range(len(df))

        total_ops = int(len(df))
        wins = int((df["pips"] > 0).sum())
        total_pips = float(round(df["pips"].sum(), 2))
        max_dd = float(round((df["equity"] - df["equity"].cummax()).min(), 2))

        points = df[["time", "equity"]].rename(columns={"equity": "equity"})
        points["equity"] = points["equity"].round(2)

        return {
            "data": points.to_dict(orient="records"),
            "stats": {
                "total_ops": total_ops,
                "wins": wins,
                "win_rate": round(wins / total_ops * 100, 1) if total_ops else 0,
                "total_pips": total_pips,
                "max_drawdown": max_dd,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backtest/equity-all")
def backtest_equity_all():
    csv_path = Path(f"{BACKTEST_RESULTS_DIR}/all_results.csv")
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="all_results.csv no encontrado")

    try:
        import pandas as pd

        df = pd.read_csv(csv_path)
        if "pips_acumulados" not in df.columns:
            df["pips_acumulados"] = df["pips"].cumsum()

        if "time_close" in df.columns:
            df["time"] = pd.to_datetime(df["time_close"], errors="coerce").dt.strftime("%Y-%m-%d")
        else:
            df["time"] = range(len(df))

        total_ops = int(len(df))
        total_pips = float(round(df["pips"].sum(), 2))
        max_dd = float(round((df["pips_acumulados"] - df["pips_acumulados"].cummax()).min(), 2))

        points = df[["time", "pips_acumulados"]].rename(columns={"pips_acumulados": "equity"})
        points["equity"] = points["equity"].round(2)

        return {
            "data": points.to_dict(orient="records"),
            "stats": {
                "total_ops": total_ops,
                "total_pips": total_pips,
                "max_drawdown": max_dd,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# NODES
# =============================================================================

@router.get("/nodes")
def nodes_list(
    principal_symbol: str = Query(None),
    symbol_cruce: str = Query(None),
    mercado: str = Query(None),
    label: str = Query(None),
    page: int = Query(1, ge=1),
    min_ops: int = Query(5, ge=0),
):
    page_size = 50
    offset = (page - 1) * page_size
    try:
        rows, total = get_top_quality_nodes(
            principal_symbol=principal_symbol or None,
            symbol_cruce=symbol_cruce or None,
            mercado=mercado or None,
            label=label or None,
            limit=page_size,
            offset=offset,
            min_total_operations=min_ops,
        )
        return {"data": rows, "total": total, "page": page, "page_size": page_size}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# BACKUP / RESTORE
# =============================================================================

@router.post("/backup/create")
def backup_create():
    try:
        file_name = create_backup()
        return {"status": "ok", "file": file_name}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"pg_dump falló: {e.stderr}")
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"pg_dump no encontrado en PATH: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backup/list")
def backup_list():
    return {"backups": list_backups()}


@router.get("/backup/download/{file_name}")
def backup_download(file_name: str):
    from fastapi.responses import FileResponse
    from src.db.backup_db import BACKUP_DIR, _SAFE_FILENAME
    if not _SAFE_FILENAME.match(file_name):
        raise HTTPException(status_code=400, detail="Nombre de archivo no válido")
    path = BACKUP_DIR / file_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    return FileResponse(str(path), media_type="application/octet-stream", filename=file_name)


@router.post("/backup/restore")
def backup_restore(file: str = Query(...)):
    try:
        restore_backup(file)
        return {"status": "ok", "restored": file}
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backup/upload")
async def backup_upload(file: UploadFile = File(...)):
    """Sube un archivo .sql y lo restaura directamente. Útil para clonar DB desde otra PC."""
    if not file.filename or not file.filename.endswith(".sql"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos .sql")
    from pathlib import Path
    from src.db.backup_db import BACKUP_DIR, _SAFE_FILENAME
    if not _SAFE_FILENAME.match(file.filename):
        raise HTTPException(status_code=400, detail="Nombre de archivo no válido")
    BACKUP_DIR.mkdir(exist_ok=True)
    dest = BACKUP_DIR / file.filename
    content = await file.read()
    dest.write_bytes(content)
    try:
        restore_backup(file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "ok", "restored": file.filename}
    
