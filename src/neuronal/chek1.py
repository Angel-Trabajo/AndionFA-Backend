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
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.db.query import get_nodes_by_label
from src.neuronal.entrenar import load_trained_model, predict_from_inputs, BinaryNN, load_data
from src.neuronal.data_para_entrenar import data_for_neuronal, clean_majority

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


# ------------------------------------------------------------
# BASE IS / OS
# ------------------------------------------------------------

df_is = pd.read_csv('output/is_os/is.csv')
df_os = pd.read_csv('output/is_os/os.csv')

df_base1 = (
    pd.concat([df_is, df_os], ignore_index=True)
      .drop_duplicates(subset='time')
)

df_base1['time'] = pd.to_datetime(df_base1['time'])

df_base = df_base1[
    df_base1['time'] >= datetime.strptime(config_node['dateStart'], '%Y-%m-%d') - relativedelta(years=4)
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

indicadores_dict = {}

list_files_is = os.listdir('output/extrac/')
list_files_os = os.listdir('output/extrac_os/')

for name_is, name_os in zip(list_files_is, list_files_os):

    df1 = pd.read_parquet(f'output/extrac/{name_is}')
    df2 = pd.read_parquet(f'output/extrac_os/{name_os}')

    df = pd.concat([df1, df2], ignore_index=True).drop_duplicates(subset=['time'])

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # evitar leakage
    df.iloc[:, 1:] = df.iloc[:, 1:].shift(1)

    df = df.set_index("time")

    indicadores_dict[name_is] = df

print("Indicadores cargados:", len(indicadores_dict))

pips_mas_alto = 0.0
bynary_best = {}

list_intervalos = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
for interval in list_intervalos:
    

    ini1 = time.time()
    data_for_neuronal(algorithm, principal_symbol, 30, df_base1, indicadores_dict)
    print(f"Dataset para neuronal generado en {time.time() - ini1:.2f} segundos.")
    clean_majority(algorithm, principal_symbol)


    X, Y = load_data("src/neuronal/data/data_cleaned_UP_EURUSD.csv")
    print("Datos cargados:", X.shape)

    nn = BinaryNN(input_dim=X.shape[1], lr=0.01, target_loss=0.01)
    nn.fit(X, Y, epochs=20000, batch_size=32)

    preds = nn.predict(X)
    acc = (preds.reshape(-1) == Y.reshape(-1)).mean() * 100
    print(f"\nAccuracy final: {acc:.2f}%")

    # ------------ GUARDAR EL MODELO ------------
    model = {
        'W1': nn.W1.tolist(), 'b1': nn.b1.tolist(),
        'W2': nn.W2.tolist(), 'b2': nn.b2.tolist(),
        'W3': nn.W3.tolist(), 'b3': nn.b3.tolist(),
        'W4': nn.W4.tolist(), 'b4': nn.b4.tolist(),
    }

    with open("src/neuronal/data/nn_binary_best.json", "w") as f:
        json.dump(model, f, indent=2)

    time.sleep(2)  # pequeño descanso antes del backtest




    with open('src/neuronal/data/maping_open.json') as f:
        encoding_actions = json.load(f)

    with open('src/neuronal/data/maping_close.json') as f:
        encoding_actions_close = json.load(f)



    nn = load_trained_model(
        "src/neuronal/data/nn_binary_best.json",
        input_dim=18
    )
   
    # ------------------------------------------------------------
    # LOOP PRINCIPAL
    # ------------------------------------------------------------

    is_open = False
    open_price_open = 0.0
    sum_pips = 0.0
    entry_red = ()
    cierre = 0

    cantidad_operaciones = 0
    operaciones_acertadas = 0
    operaciones_perdedoras = 0

    ganancia_bruta = 0.0
    perdida_bruta = 0.0

    lista_pips = []  # para pseudo-sharpe
    
    for row in df_base.itertuples():

        time_actual = row.Index
        open_price = row.open

        # =========================================================
        # CIERRE
        # =========================================================
        if is_open:
            cierre += 1
            if cierre >= (interval if interval >=50 else 50):
                print(f"Cierre forzado por tiempo: {time_actual} - Intervalo: {interval} - Cierre count: {cierre}") 
                is_open = False
                cantidad_operaciones += 1

                if algorithm == 'UP':
                    trade_pips = open_price - open_price_open
                else:
                    trade_pips = open_price_open - open_price

                sum_pips += trade_pips
                lista_pips.append(trade_pips)

                if trade_pips > 0:
                    operaciones_acertadas += 1
                    ganancia_bruta += trade_pips
                else:
                    operaciones_perdedoras += 1
                    perdida_bruta += abs(trade_pips)

                print(f"Cierre forzado por tiempo en {time_actual}")
                continue
            for nodo in nodos_close:

                df_struct = COMBINED[("close", nodo["file"])]
                df = df_struct["df"]

                pos = df.index.searchsorted(time_actual)
                if pos == 0:
                    continue

                if cumple_condiciones_fast(df_struct, pos - 1, nodo["conditions"]):

        
                    entry_red_close = tuple(encoding_actions_close[nodo["key"]])
                    clase, _ = predict_from_inputs(nn, entry_red, entry_red_close)
                    if clase == 1: 
                        is_open = False
                        cantidad_operaciones += 1

                        if algorithm == 'UP':
                            trade_pips = open_price - open_price_open
                        else:
                            trade_pips = open_price_open - open_price

                        sum_pips += trade_pips
                        lista_pips.append(trade_pips)

                        if trade_pips > 0:
                            operaciones_acertadas += 1
                            ganancia_bruta += trade_pips
                        else:
                            operaciones_perdedoras += 1
                            perdida_bruta += abs(trade_pips)

                        print(f"SUM PIPS: {time_actual}: {sum_pips}")
                        break

        # =========================================================
        # APERTURA
        # =========================================================
        else:
            cierre = 0
            for symbol in list_symbols:

                cumple_alguno = False

                for nodo in dict_nodos[symbol]:

                    df_struct = COMBINED[("open", symbol, nodo["file"])]
                    df = df_struct["df"]

                    pos = df.index.searchsorted(time_actual)
                    if pos == 0:
                        continue

                    if cumple_condiciones_fast(df_struct, pos - 1, nodo["conditions"]):

                        cumple_alguno = True

                        if symbol == list_symbols[-1]:
                            entry_red = tuple(encoding_actions[nodo["key"]])
                            open_price_open = open_price
                            is_open = True

                        break

                if not cumple_alguno:
                    break
    winrate = (operaciones_acertadas / cantidad_operaciones) if cantidad_operaciones > 0 else 0.0

    profit_factor = (
        ganancia_bruta / perdida_bruta
        if perdida_bruta > 0 else 0
    )

    avg_win = ganancia_bruta / operaciones_acertadas if operaciones_acertadas > 0 else 0
    avg_loss = perdida_bruta / operaciones_perdedoras if operaciones_perdedoras > 0 else 0

    expectancy = (winrate * avg_win) - ((1 - winrate) * avg_loss)

    # Pseudo Sharpe sobre pips
    if len(lista_pips) > 1 and np.std(lista_pips) != 0:
        sharpe = np.mean(lista_pips) / np.std(lista_pips)
    else:
        sharpe = 0

    print(f"""
    Operaciones: {cantidad_operaciones}
    Winrate: {winrate*100:.2f}%
    Profit Factor: {profit_factor:.2f}
    Expectancy: {expectancy:.4f}
    Sharpe (pips): {sharpe:.4f}
    Pips Totales: {sum_pips:.2f}
    """)
    
    # Penalización por pocas operaciones
    penalizacion = np.log(cantidad_operaciones + 1)

    score = (
        0.35 * sharpe +
        0.25 * profit_factor +
        0.20 * winrate +
        0.20 * (sum_pips / 1000)
    ) * penalizacion
    if score > pips_mas_alto:
        pips_mas_alto = score
        bynary_best = model
        print(f"🏆 Nuevo mejor modelo en intervalo {interval}")

with open("src/neuronal/data/nn_binary_best.json", "w") as f:
        json.dump(bynary_best, f, indent=2)
# ------------------------------------------------------------
# RESUMEN FINAL DEBUG
# ------------------------------------------------------------

