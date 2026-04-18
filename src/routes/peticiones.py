import os
from datetime import datetime

import MetaTrader5 as mt5
from dotenv import load_dotenv


load_dotenv()
MT5_PATH = os.getenv("MT5_PATH")

_initialized = False


def initialize_mt5():
    global _initialized
    if _initialized and mt5.terminal_info() is not None:
        return
    _initialized = False
    mt5.shutdown()
    if not mt5.initialize(path=MT5_PATH):
        raise RuntimeError(f"MT5 failed to initialize: {mt5.last_error()}")
    _initialized = True
    print("MetaTrader5 initialized.")


def get_active_symbols():
    initialize_mt5()
    symbols = mt5.symbols_get()
    if symbols is None:
        return {"symbols": []}
    active_symbols = [symbol.name for symbol in symbols if symbol.visible]
    return {"symbols": active_symbols}

       
        
def get_timeframes():
    timeframes = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1
    }
    return {"timeframes": timeframes}
        


def get_historical_data(symbol, timeframe, start, end):
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    timeframe = int(timeframe)

    # ✅ obtener 200 velas antes del start
    past_rates = mt5.copy_rates_from(
        symbol,
        timeframe,
        start_dt,
        200
    )

    if past_rates is None or len(past_rates) == 0:
        return {"error": "Error obteniendo velas pasadas"}

    new_start = datetime.fromtimestamp(past_rates[0]["time"])

    # ✅ ahora rango completo
    rates = mt5.copy_rates_range(
        symbol,
        timeframe,
        new_start,
        end_dt
    )

    if rates is None or len(rates) == 0:
        return {"error": "Error al extraer los datos"}

    data = [
        {
            "time": int(r["time"]),
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
            "tick_volume": int(r["tick_volume"]),
            "spread": int(r["spread"]),
            "real_volume": int(r["real_volume"]),
        }
        for r in rates
    ]

    return {"data": data}

