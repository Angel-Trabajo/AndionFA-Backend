import asyncio

import MetaTrader5 as mt5
from fastapi import APIRouter, Body, HTTPException, WebSocket, WebSocketDisconnect

from src import engine_manager

router = APIRouter()


@router.get("/live-config")
def get_live_config():
    return engine_manager.get_live_config()


@router.post("/live-config")
def update_live_config(payload: dict):
    return engine_manager.update_live_filters(payload)


@router.post("/apply-filter")
def apply_live_filter():
    return engine_manager.apply_live_filter()


@router.post("/engine/{engine_id}/lot-size")
def set_engine_lot_size(engine_id: str, payload: dict):
    if "lot_size" not in payload:
        raise HTTPException(status_code=400, detail="lot_size es requerido")
    return engine_manager.set_engine_lot_size(engine_id, float(payload["lot_size"]))


@router.post("/engine/{engine_id}/stop")
def stop_single_engine(engine_id: str, payload: dict):
    mode = str(payload.get("mode", "graceful"))
    result = engine_manager.stop_engine(engine_id, mode)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("detail", "error"))
    return result


@router.get("/stats")
def get_live_stats():
    return engine_manager.get_live_stats()


@router.post("/start")
def engine_start():
    return engine_manager.start()


@router.post("/stop")
def engine_stop(payload: dict | None = Body(default=None)):
    mode = "graceful"
    if payload is not None:
        mode = str(payload.get("mode", "graceful"))
    result = engine_manager.stop(mode=mode)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("detail", "error"))
    return result


@router.get("/status")
def engine_status():
    return engine_manager.get_status()


@router.get("/mt5-status")
def mt5_status():
    info = mt5.terminal_info()
    if info is None:
        return {"connected": False}
    account = mt5.account_info()
    return {
        "connected": True,
        "company": info.company,
        "server": account.server if account else None,
        "login": account.login if account else None,
        "balance": account.balance if account else None,
        "currency": account.currency if account else None,
    }


@router.websocket("/ws")
async def engine_ws(websocket: WebSocket):
    """
    WebSocket que hace streaming de logs en tiempo real.
    Envía cada segundo los mensajes nuevos desde el cursor del cliente.
    """
    await websocket.accept()
    cursor = 0
    try:
        while True:
            logs = engine_manager.get_logs(since=cursor)
            if logs:
                cursor += len(logs)
                await websocket.send_json({"logs": logs})
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
