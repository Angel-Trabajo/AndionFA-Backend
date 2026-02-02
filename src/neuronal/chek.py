import pandas as pd 
import os 
import json
import ast
import operator
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.db.query import get_nodes_by_label
from src.neuronal.entrenar import load_trained_model, predict_from_inputs



with open('config/config_test/config_test_red.json', 'r') as file:
    config = json.load(file)
algorithm = config['algorithm']

with open(f'config/list_{algorithm}.json', 'r') as file:
    config_extractor = json.load(file)
with open('config/config_crossing/config_crossing.json', 'r') as file:
    config_crossing = json.load(file)
list_symbols = config_extractor['list'] 
principal_symbol = config_crossing['principal_symbol']
list_symbols.insert(0, principal_symbol)

with open('src/neuronal/data/maping_close.json', 'r') as file:
    encoding_actions_close = json.load(file)   
with open('src/neuronal/data/maping_open.json', 'r') as file:
    encoding_actions = json.load(file)

with open('output/is_os/is.csv', 'r') as file:
    df_is = pd.read_csv(file)

with open('output/is_os/os.csv', 'r') as file:
    df_os = pd.read_csv(file)
df = pd.concat([df_is, df_os], ignore_index=True)
df = df.drop_duplicates(subset="time")

list_files = os.listdir('output/extrac')
list_files_os = os.listdir('output/extrac_os')
file = list_files[0]

with open(f'output/extrac/{file}', 'r') as f:
    df_extra_is = pd.read_csv(f)
inicial_time = df_extra_is.head(1)['time']

index = df.index[df['time'] == inicial_time.values[0]].tolist()[0]
df = df.iloc[index:]


dict_nodos = {}
for i, symbol in enumerate(list_symbols):
    if i == 0:
        dict_nodos[symbol] = get_nodes_by_label(symbol, algorithm)
    else:
        dict_nodos[symbol] = get_nodes_by_label(f'crossing_{principal_symbol}_dbs/{symbol}', algorithm)

def parsear_nodos(dict_nodos):
    nodos_parseados = {}

    for symbol, nodos in dict_nodos.items():
        lista = []
        for nodo in nodos:
            condiciones_str = nodo[0]
            file_name = nodo[1]

            lista.append({
                "key": condiciones_str,                    # 🔑 CLAVE
                "conditions": ast.literal_eval(condiciones_str),
                "file": file_name
            })

        nodos_parseados[symbol] = lista

    return nodos_parseados

dict_nodos = parsear_nodos(dict_nodos)


if algorithm == 'UP':
    other_algorithm = 'DOWN'
else:
    other_algorithm = 'UP'
nodos_close = get_nodes_by_label(principal_symbol, other_algorithm)

nodos_close = [
    {
        "key": n[0],
        "conditions": ast.literal_eval(n[0]),
        "file": n[1]
    }
    for n in nodos_close
]

operadores = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne
}

nn = load_trained_model("src/neuronal/data/nn_binary_best.json", input_dim=18)

def cumple_condiciones(fila, condiciones):
    for col, op, valor in condiciones:
        if fila[col] is None:
            return False
        if not operadores[op](fila[col], valor):
            return False
    return True



entry_red = []
order = 'NONE'
is_open = False
open_price_open = 0
sum_pips = 0

for i, row in enumerate(df.iterrows()):
    open_price = row[1]['open']
    
    if is_open:    
        for i, nodo in enumerate(nodos_close):
            file = nodo["file"]
            with open(f'output/extrac/{file}', 'r') as f:
                df_extra_is = pd.read_csv(f)
            file_biging = file.split('_')[0]
            file_os = next((s for s in list_files_os if file_biging in s), None)    
            with open(f'output/extrac_os/{file_os}', 'r') as f:
                df_extra_os = pd.read_csv(f)
            df = pd.concat([df_extra_is, df_extra_os], ignore_index=True)
            df = df.drop_duplicates(subset="time")
            
            row_extract = df[df['time'] == row[1]['time']]
            if i > 0:
                row_extract = df.loc[row_extract.index[0] - 1]
            if cumple_condiciones(row_extract, nodo["conditions"]):
                entry_red_close = encoding_actions_close[nodo["key"]]
                entry_red = tuple(entry_red)
                entry_red_close = tuple(entry_red_close)
                clase, valor = predict_from_inputs(nn, entry_red, entry_red_close)

                if (clase == 1):
                    order = 'CLOSE'
                    is_open = False
                    if algorithm == 'UP':
                        sum_pips += (open_price_open - open_price)
                    else:
                        sum_pips += (open_price - open_price_open)
                    open_price_open = 0
                    break
            
                else:
                    order = 'NONE'
            else:
                order = 'NONE'
    else:
        state = 1  # 1 = todos cumplen, 0 = alguno falla
        for symbol in list_symbols:
            nodos = dict_nodos[symbol]
            cumple_alguno = False  # bandera para saber si este símbolo cumple al menos un nodo
            for nodo in nodos:
                file = nodo["file"]
                with open(f'output/crossing_EURUSD_{algorithm}/{symbol}/extrac/{file}', 'r') as f:
                    df_extra_is = pd.read_csv(f)
                file_biging = file.split('_')[0]
                file_os = next((s for s in list_files_os if file_biging in s), None)
                file_os = file_os.replace('EURUSD', symbol)    
                with open(f'output/crossing_EURUSD_{algorithm}/{symbol}/extrac_os/{file_os}', 'r') as f:
                    df_extra_os = pd.read_csv(f)
                df = pd.concat([df_extra_is, df_extra_os], ignore_index=True)
                df = df.drop_duplicates(subset="time")

                row_extract = df[df['time'] == row[1]['time']].iloc[0]
                if i > 0:
                    row_extract = df.loc[row_extract.index[0] - 1]
                if cumple_condiciones(row_extract, nodo["conditions"]):
                    cumple_alguno = True
                    # si es el último símbolo, cargamos la red
                    if symbol == list_symbols[-1]:
                        entry_red = encoding_actions[nodo["key"]]
                        open_price_open = open_price
                        is_open = True
                    break
            # si este símbolo no cumplió ningún nodo, se detiene todo
            if not cumple_alguno:
                state = 0
                print(f"❌ El símbolo {symbol} no cumple ningún nodo. Se detiene la búsqueda.")
                break
            
    print(sum_pips)    
    


