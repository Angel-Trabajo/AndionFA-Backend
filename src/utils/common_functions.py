import os
import shutil
from datetime import datetime
import sqlite3

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

def eliminar_ruta(ruta):
    if os.path.exists(ruta):
        if os.path.isfile(ruta):
            os.remove(ruta)  # elimina archivo
            print(f"Archivo '{ruta}' eliminado correctamente.")
        elif os.path.isdir(ruta):
            shutil.rmtree(ruta)  # elimina carpeta con todo su contenido
            print(f"Carpeta '{ruta}' eliminada correctamente.")
    else:
        print(f"La ruta '{ruta}' no existe.")

def eliminar_archivo(ruta_archivo):
    if os.path.exists(ruta_archivo):
        os.remove(ruta_archivo)
        print(f"Archivo '{ruta_archivo}' eliminado correctamente.")
    else:
        print(f"El archivo '{ruta_archivo}' no existe.")

def tabla_nodes_existe(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='nodes'")
        existe = cursor.fetchone() is not None
        conn.close()
        return existe
    except:
        return False

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


def should_backtest_strategy(metrics):
    import math
    import numpy as np

    winrate = metrics.get("winrate", 0)
    profit_factor = metrics.get("profit_factor", 0)
    expectancy = metrics.get("expectancy", 0)
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
        return False

    s = score(list_pips_monthly)
    prob = probabilidad(s)

    return (
        winrate >= 0.40 and
        profit_factor >= 1 and
        expectancy >= 1 and
        prob >= 0.55
    )