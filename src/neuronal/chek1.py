import pandas as pd
import os
import json
import ast
import operator
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.db.query import get_nodes_by_label
from src.neuronal.entrenar import load_trained_model, predict_from_inputs


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

list_symbols = config_extractor['list']
principal_symbol = config_crossing['principal_symbol']
list_symbols.insert(0, principal_symbol)

with open('src/neuronal/data/maping_open.json') as f:
    encoding_actions = json.load(f)

with open('src/neuronal/data/maping_close.json') as f:
    encoding_actions_close = json.load(f)


# ------------------------------------------------------------
# CARGA BASE IS / OS
# ------------------------------------------------------------

df_is = pd.read_csv('output/is_os/is.csv')
df_os = pd.read_csv('output/is_os/os.csv')

df_base = (
    pd.concat([df_is, df_os], ignore_index=True)
      .drop_duplicates(subset='time')
)

df_base['time'] = pd.to_datetime(df_base['time'])
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
# OPERADORES
# ------------------------------------------------------------

operadores = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne
}


def cumple_condiciones(fila, condiciones):
    for col, op, valor in condiciones:
        v = fila[col]
        if v is None or not operadores[op](v, valor):
            return False
    return True


# ------------------------------------------------------------
# CACHE DE DATAFRAMES (CLAVE)
# ------------------------------------------------------------

DATA_CACHE = {}


def load_df(path):
    if path not in DATA_CACHE:
        df = pd.read_csv(path)
        df['time'] = pd.to_datetime(df['time'])
        df = (
            df.drop_duplicates(subset='time')
              .sort_values('time')
              .set_index('time')
        )
        DATA_CACHE[path] = df
    return DATA_CACHE[path]


# ------------------------------------------------------------
# RED NEURONAL
# ------------------------------------------------------------

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

for row in df_base.itertuples():

    time_actual = row.Index
    open_price = row.open

    # ========================================================
    # CIERRE
    # ========================================================
    if is_open:
        for nodo in nodos_close:

            df_is = load_df(f'output/extrac/{nodo["file"]}')
            file_base = nodo["file"].split('_')[0]
            file_os = next(f for f in os.listdir('output/extrac_os') if file_base in f)
            df_os = load_df(f'output/extrac_os/{file_os}')

            df = pd.concat([df_is, df_os]).drop_duplicates()
            df = df.sort_index()

            if time_actual not in df.index:
                continue

            pos = df.index.get_loc(time_actual)
            if pos == 0:
                continue

            fila = df.iloc[pos - 1]

            if cumple_condiciones(fila, nodo["conditions"]):
                entry_red_close = tuple(encoding_actions_close[nodo["key"]])
                clase, _ = predict_from_inputs(nn, entry_red, entry_red_close)

                if clase == 1:
                    is_open = False
                    if algorithm == 'UP':
                        sum_pips += open_price_open - open_price
                    else:
                        sum_pips += open_price - open_price_open
                    break

    # ========================================================
    # APERTURA
    # ========================================================
    else:
        state = True

        for symbol in list_symbols:
            cumple_alguno = False

            for nodo in dict_nodos[symbol]:

                path_is = f'output/crossing_EURUSD_{algorithm}/{symbol}/extrac/{nodo["file"]}'
                df_is = load_df(path_is)

                file_base = nodo["file"].split('_')[0]
                file_os = next(f for f in os.listdir(f'output/crossing_EURUSD_{algorithm}/{symbol}/extrac_os') if file_base in f)
                df_os = load_df(f'output/crossing_EURUSD_{algorithm}/{symbol}/extrac_os/{file_os}')

                df = pd.concat([df_is, df_os]).drop_duplicates()
                df = df.sort_index()

                if time_actual not in df.index:
                    continue

                pos = df.index.get_loc(time_actual)
                if pos == 0:
                    continue

                fila = df.iloc[pos - 1]

                if cumple_condiciones(fila, nodo["conditions"]):
                    cumple_alguno = True

                    if symbol == list_symbols[-1]:
                        entry_red = tuple(encoding_actions[nodo["key"]])
                        open_price_open = open_price
                        is_open = True
                    break

            if not cumple_alguno:
                state = False
                break
    
    print(f"SUM PIPS: {time_actual}: {sum_pips:.2f}")
    
# nota: tener en cuenta las 00:00 que no tienen datos de cierre
# revisar los primeros tres años del df que no se han eliminado