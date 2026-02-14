import pandas as pd
import os
import json
import ast
import operator
import sys
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pyarrow.parquet as pq

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.db.query import get_nodes_by_label
from src.neuronal.entrenar import load_trained_model, predict_from_inputs

print("Starting backtest...")

# ------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------

with open('config/config_test/config_test_red.json') as f:
    config = json.load(f)

algorithm = config['algorithm']
other_algorithm = 'DOWN' if algorithm == 'UP' else 'UP'

with open(f'config/list_{algorithm}.json') as f:
    config_extractor = json.load(f)

with open('config/config_crossing/config_crossing.json') as f:
    config_crossing = json.load(f)

with open('config/config_node/config_node.json') as f:
    config_node = json.load(f)

list_symbols = config_extractor['list']
principal_symbol = config_crossing['principal_symbol']
list_symbols.insert(0, principal_symbol)

with open('src/neuronal/data/maping_open.json') as f:
    encoding_actions = json.load(f)

with open('src/neuronal/data/maping_close.json') as f:
    encoding_actions_close = json.load(f)

# ------------------------------------------------------------
# BASE IS / OS
# ------------------------------------------------------------

df_is = pd.read_csv('output/is_os/is.csv')
df_os = pd.read_csv('output/is_os/os.csv')

df_base = (
    pd.concat([df_is, df_os], ignore_index=True)
      .drop_duplicates(subset='time')
)

df_base['time'] = pd.to_datetime(df_base['time'])
df_base = df_base[
    df_base['time'] >= datetime.strptime(config_node['dateStart'], '%Y-%m-%d') - relativedelta(years=4)
]

df_base = df_base.sort_values('time').set_index('time')

# ------------------------------------------------------------
# CARGA NODOS
# ------------------------------------------------------------

def parsear_nodos(nodos):
    return [
        {
            "key": n[0],
            "conditions": ast.literal_eval(n[0]),
            "file": n[1]
        }
        for n in nodos
    ]

dict_nodos = {}
for i, symbol in enumerate(list_symbols):
    label = symbol if i == 0 else f'crossing_{principal_symbol}_dbs/{symbol}'
    dict_nodos[symbol] = parsear_nodos(get_nodes_by_label(label, algorithm))

nodos_close = parsear_nodos(
    get_nodes_by_label(principal_symbol, other_algorithm)
)
# ------------------------------------------------------------
# EXTRAER COLUMNAS NECESARIAS (SET REAL)
# ------------------------------------------------------------

columnas_usadas = set()

for symbol in dict_nodos:
    for nodo in dict_nodos[symbol]:
        for col, _, _ in nodo["conditions"]:
            columnas_usadas.add(col)

for nodo in nodos_close:
    for col, _, _ in nodo["conditions"]:
        columnas_usadas.add(col)

columnas_usadas.add("time")


# # ------------------------------------------------------------
# # OPERADORES
# # ------------------------------------------------------------

operadores = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne
}

def cumple_condiciones_fast(df_struct, row_idx, condiciones):
    row = df_struct["values"][row_idx]
    col_map = df_struct["col_map"]

    for col, op, valor in condiciones:
        if col not in col_map:
            return False
        v = row[col_map[col]]
        if v is None or not operadores[op](v, valor):
            return False
    return True

# # ------------------------------------------------------------
# # CACHE DATAFRAMES (PARQUET OPTIMIZADO)
# # ------------------------------------------------------------

DATA_CACHE = {}

def load_df(path):

    parquet_path = path.replace(".csv", ".parquet")

    if parquet_path in DATA_CACHE:
        return DATA_CACHE[parquet_path]

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"No existe parquet: {parquet_path}")

    # Leer solo schema
    schema = pq.read_schema(parquet_path)
    columnas_disponibles = set(schema.names)

    columnas_validas = list(columnas_usadas & columnas_disponibles)

    df = pd.read_parquet(
        parquet_path,
        columns=columnas_validas if columnas_validas else None
    )

    # 🔥 Aquí está la clave
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').set_index('time')
    else:
        # ya viene con time como índice
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

    DATA_CACHE[parquet_path] = df
    return df

# ------------------------------------------------------------
# PRELOAD COMBINED
# ------------------------------------------------------------

ini = time.time()
COMBINED = {}

def build_combined(path_is, path_os):

    df_is = load_df(path_is)
    df_os = load_df(path_os)

    df = pd.concat([df_is, df_os])

    # Más rápido que drop_duplicates()
    df = df[~df.index.duplicated(keep='last')]

    return {
        "df": df,
        "values": df.values,
        "col_map": {col: i for i, col in enumerate(df.columns)}
    }

# CLOSE

FILES_OS_CLOSE = {
    f.split('_')[0]: f
    for f in os.listdir('output/extrac_os')
}

for nodo in nodos_close:
    path_is = f'output/extrac/{nodo["file"]}'
    file_base = nodo["file"].split('_')[0]
    file_os = FILES_OS_CLOSE[file_base]
    path_os = f'output/extrac_os/{file_os}'

    COMBINED[("close", nodo["file"])] = build_combined(path_is, path_os)

# OPEN

for symbol in list_symbols:

    if symbol == principal_symbol:
        path = 'output'
        path_os_root = 'output/extrac_os'
    else:
        path = f'output/crossing_{principal_symbol}/{symbol}'
        path_os_root = f'{path}/extrac_os'

    FILES_LOCAL = {
        f.split('_')[0]: f
        for f in os.listdir(path_os_root)
    }

    for nodo in dict_nodos[symbol]:
        path_is = f'{path}/extrac/{nodo["file"]}'
        file_base = nodo["file"].split('_')[0]
        file_os = FILES_LOCAL[file_base]
        path_os = f'{path}/extrac_os/{file_os}'
        
        COMBINED[("open", symbol, nodo["file"])] = build_combined(path_is, path_os)   

print(f"Tiempo de carga: {time.time() - ini:.2f} segundos")
print("Archivos cargados:", len(DATA_CACHE))

# ------------------------------------------------------------
# RED NEURONAL
# ------------------------------------------------------------







nn = load_trained_model(
    "src/neuronal/data/nn_binary_best.json",
    input_dim=18
)





# ------------------------------------------------------------
# DEBUG RANGO BASE
# ------------------------------------------------------------

print("RANGO df_base:")
print("Inicio:", df_base.index.min())
print("Fin:", df_base.index.max())
print("Total filas:", len(df_base))
print("-" * 50)


# ------------------------------------------------------------
# CONTADORES DEBUG
# ------------------------------------------------------------

opens_condicion_por_anio = {}
closes_condicion_por_anio = {}
nn_aprueba_por_anio = {}
operaciones_reales_por_anio = {}


# ------------------------------------------------------------
# LOOP PRINCIPAL
# ------------------------------------------------------------

is_open = False
open_price_open = 0.0
sum_pips = 0.0
entry_red = ()

for row in df_base.itertuples():

    time_actual = row.Index
    open_price = row.open

    # =========================================================
    # CIERRE
    # =========================================================
    if is_open:

        for nodo in nodos_close:

            df_struct = COMBINED[("close", nodo["file"])]
            df = df_struct["df"]

            pos = df.index.searchsorted(time_actual)
            if pos == 0:
                continue

            if cumple_condiciones_fast(df_struct, pos - 1, nodo["conditions"]):

                # DEBUG condiciones close
                anio = time_actual.year
                closes_condicion_por_anio[anio] = closes_condicion_por_anio.get(anio, 0) + 1

                entry_red_close = tuple(encoding_actions_close[nodo["key"]])
                clase, _ = predict_from_inputs(nn, entry_red, entry_red_close)

                if clase == 1:

                    # DEBUG red aprueba
                    nn_aprueba_por_anio[anio] = nn_aprueba_por_anio.get(anio, 0) + 1

                    is_open = False

                    if algorithm == 'UP':
                        sum_pips += open_price - open_price_open
                    else:
                        sum_pips += open_price_open - open_price

                    # DEBUG operación real
                    operaciones_reales_por_anio[anio] = operaciones_reales_por_anio.get(anio, 0) + 1

                    print(f"SUM PIPS: {time_actual}: {sum_pips} => {open_price_open}-{open_price} = {open_price_open - open_price}")
                    break

    # =========================================================
    # APERTURA
    # =========================================================
    else:

        for symbol in list_symbols:

            cumple_alguno = False

            for nodo in dict_nodos[symbol]:

                df_struct = COMBINED[("open", symbol, nodo["file"])]
                df = df_struct["df"]

                pos = df.index.searchsorted(time_actual)
                if pos == 0:
                    continue

                if cumple_condiciones_fast(df_struct, pos - 1, nodo["conditions"]):

                    # DEBUG condiciones open
                    anio = time_actual.year
                    opens_condicion_por_anio[anio] = opens_condicion_por_anio.get(anio, 0) + 1

                    cumple_alguno = True

                    if symbol == list_symbols[-1]:
                        entry_red = tuple(encoding_actions[nodo["key"]])
                        open_price_open = open_price
                        is_open = True

                    break

            if not cumple_alguno:
                break


# ------------------------------------------------------------
# RESUMEN FINAL DEBUG
# ------------------------------------------------------------

print("\n" + "="*60)
print("DEBUG ESTADÍSTICAS POR AÑO")
print("="*60)

print("OPEN condiciones por año:")
print(opens_condicion_por_anio)

print("\nCLOSE condiciones por año:")
print(closes_condicion_por_anio)

print("\nRed devuelve clase 1 por año:")
print(nn_aprueba_por_anio)

print("\nOperaciones reales ejecutadas por año:")
print(operaciones_reales_por_anio)

print("="*60)