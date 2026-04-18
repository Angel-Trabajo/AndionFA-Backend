import os
import shutil
import math
import numpy as np
import json
from datetime import datetime

def limpiar_carpeta(ruta_carpeta):
    if os.path.exists(ruta_carpeta):
        for nombre in os.listdir(ruta_carpeta):
            ruta_elemento = os.path.join(ruta_carpeta, nombre)
            if os.path.isfile(ruta_elemento) or os.path.islink(ruta_elemento):
                os.unlink(ruta_elemento)  # elimina archivo o enlace simbólico
            elif os.path.isdir(ruta_elemento):
                shutil.rmtree(ruta_elemento)  # elimina carpeta recursivamente
        print(f"Contenido de la carpeta '{ruta_carpeta}' eliminado correctamente.")
    else:
        print(f"La carpeta '{ruta_carpeta}' no existe.")

def crear_carpeta_si_no_existe(ruta_carpeta):
    if not os.path.exists(ruta_carpeta):
        os.makedirs(ruta_carpeta)
        print(f"Carpeta '{ruta_carpeta}' creada correctamente.")
    else:
        print(f"La carpeta '{ruta_carpeta}' ya existe.")

def get_previous_4_6(date_start: str, date_end: str):
   
    start = datetime.strptime(date_start, "%Y-%m-%d")
    end = datetime.strptime(date_end, "%Y-%m-%d")

    # duración del INPUT (2/6)
    duration = end - start

    # necesitamos 4/6 → doble duración
    extra_duration = duration * 2

    returned_end = start
    returned_start = start - extra_duration

    return (
        returned_start.strftime("%Y-%m-%d"),
        returned_end.strftime("%Y-%m-%d"),
    )
    
def filtro_mercado(df, mercado):
    df = df.copy()
    if 'hour' not in df.columns:
        df.loc[:, 'hour'] = df['time'].dt.hour

    h = df['hour']

    if mercado == "Asia":
        mascara = (h >= 22) | (h <= 6)

    elif mercado == "Europa":
        mascara = (h >= 6) & (h <= 13)

    elif mercado == "America":
        mascara = (h >= 13) & (h <= 22)

    else:
        return df

    return df.loc[mascara]

def hora_en_mercado(hour, mercado):

    if mercado == "Asia":
        return (hour >= 23) or (hour <= 6)

    elif mercado == "Europa":
        return 7 <= hour <= 13

    elif mercado == "America":
        return 14 <= hour <= 22

    return True


def evaluate_live_strategy_filter(metrics, live_config=None):
    if live_config is None:
        with open('config/live_config.json', 'r', encoding='utf-8') as f:
            live_config = json.load(f).get("live", {})

    winrate = metrics.get("winrate", 0)
    profit_factor = metrics.get("profit_factor", 0)
    expectancy = metrics.get("expectancy", 0)
    cantidad_operaciones = metrics.get("cantidad_operaciones", 0)
    list_pips_monthly = list(metrics.get("temporal_stats", {}).get("monthly_pips", {}).values())

    def score(fila):
        fila = np.array(fila)
        suma = np.sum(fila)
        volatilidad = np.std(fila)
        maximo = np.max(fila)
        minimo = np.min(fila)
        return (
            0.4 * suma
            - 0.3 * volatilidad
            + 0.2 * maximo
            + 0.1 * minimo
        )

    def probabilidad(s):
        return 1 / (1 + math.exp(-s / 100))

    if not list_pips_monthly:
        return {
            "passed": False,
            "probabilidad": 0,
            "score": 0,
            "reason": "no_monthly_pips",
        }

    s = score(list_pips_monthly)
    prob = probabilidad(s)

    passed = (
        winrate >= live_config.get("winrate", 0) and
        profit_factor >= live_config.get("profit_factor", 0) and
        expectancy >= live_config.get("expectancy", 0) and
        prob >= live_config.get("probabilidad", 0) and
        cantidad_operaciones >= live_config.get("cantidad_operaciones", 0)
    )

    return {
        "passed": bool(passed),
        "probabilidad": float(prob),
        "score": float(s),
        "reason": "ok" if passed else "threshold_not_met",
    }

def should_backtest_strategy(metrics, live_config=None):
    return evaluate_live_strategy_filter(metrics, live_config=live_config)["passed"]