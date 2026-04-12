import asyncio

import MetaTrader5 as mt5
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from src import engine_manager

router = APIRouter()


@router.post("/start")
def engine_start():
    return engine_manager.start()


@router.post("/stop")
def engine_stop():
    return engine_manager.stop()


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
