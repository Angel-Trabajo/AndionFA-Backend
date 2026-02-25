# pipeline_completo.py
# Contiene: Signal + Normalizar + data_for_neuronal + clean_majority
# Ensamblado listo para ejecutar como pipeline.

import sys
import ast
import os
import json
import sqlite3
import ast
import operator
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))



# ================================
#   Clase Signal
# ================================
class Signal:
    def __init__(self, algorithms, principal_symbol):
        self.algorithms = algorithms
        self.principal_symbol = principal_symbol
        _, _, other_algorithms, last_symbol = self.get_info()
        self.other_algorithms = other_algorithms
        self.last_symbol = last_symbol

    def get_info(self):
        if self.algorithms == "UP":
            other_algorithms = "DOWN"
        else:
            other_algorithms = "UP"

        with open(f'config/list_{self.algorithms}.json', 'r') as file:
            data = json.load(file)
            last_symbol = data['list'][-1]
        return self.algorithms, self.principal_symbol, other_algorithms, last_symbol

    def get_close_signals(self):
        def get_nodes():
            conn = sqlite3.connect(f'output/db/{self.principal_symbol}.db')
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM nodes WHERE label = ?', (self.other_algorithms,))
            res = cursor.fetchall()
            conn.close()
            return res if res else None
        nodes = get_nodes()
        return [(n[3]) for n in nodes] if nodes else None

    def get_open_signals(self):
        def get_nodes():
            conn = sqlite3.connect(f'output/db/crossing_{self.principal_symbol}_dbs/{self.last_symbol}.db')
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM nodes WHERE label = ?', (self.algorithms,))
            res = cursor.fetchall()
            conn.close()
            return res if res else None
        nodes = get_nodes()
        return [n[3] for n in nodes] if nodes else None

# ================================
#   Clase Normalizar
# ================================
class Normalizar:
    def __init__(self, signal):
        self.signal = signal

    def normalize_close_signals(self):
        signal = self.signal.get_close_signals()
        if signal is None:
            return None
        
        if len(signal) < 64:
           count = 6
        elif len(signal) < 128:
           count = 7
        else:
            count = 8   
        asign = {}
        for i, elem in enumerate(signal):
            # Normalizar siempre
            if isinstance(elem, str):
                elem = ast.literal_eval(elem)

            # Clave estable (NO str())
            key = json.dumps(elem, sort_keys=True)
            asign[key] = bin(i+1)[2:].zfill(count)  # Convertir a binario de 8 bits

        with open('src/neuronal/data/maping_close.json', 'w') as file:
            json.dump(asign, file, indent=4)

        return asign

    def normalize_open_signals(self):
        signal = self.signal.get_open_signals()
        if signal is None:
            return None
        if len(signal) < 64:
           count = 6
        elif len(signal) < 128:
           count = 7
        else:
            count = 8
        asign = {}

        for i, elem in enumerate(signal):
            # Normalizar siempre
            if isinstance(elem, str):
                elem = ast.literal_eval(elem)

            # Clave estable (NO str())
            key = json.dumps(elem, sort_keys=True)
            asign[key] = bin(i+1)[2:].zfill(count)  # Convertir a binario de 8 bits

        with open('src/neuronal/data/maping_open.json', 'w') as file:
            json.dump(asign, file, indent=4)

        return asign

# ================================
#   Funciones auxiliares
# ================================
operadores = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne
}

def data_for_neuronal(algorithm, principal_symbol, dict_pips_best= {}):

    signal = Signal(algorithm, principal_symbol)
    normalizar = Normalizar(signal)
    sign_close = normalizar.normalize_close_signals()
    sign_open = normalizar.normalize_open_signals()
    data = {
        'input1': [],
        'input2': [],
        'output': []
    }
    for j, open in enumerate(sign_open.values()):
        for i, close in enumerate(sign_close.values()):
            if f'{open}_{close}' in dict_pips_best:
                if dict_pips_best[f'{open}_{close}'] > 0:
                    valor = 1
                else:
                    valor = 0
            else:
                if i > 0 or j > 0:
                    continue
                valor = 0
            data['input1'].append(open)
            data['input2'].append(close)
            data['output'].append(valor)
            
    df = pd.DataFrame(data)
    df.to_csv(f'src/neuronal/data/data_for_neuronal_{algorithm}_{principal_symbol}.csv', index=False)
    
    
if __name__ == "__main__":
    with open('config/config_crossing/config_crossing.json', 'r') as f:
        config = json.load(f)
    
    with open('config/config_test/config_test_red.json', 'r') as f:
        config_test = json.load(f)
    
    data_for_neuronal(config_test['algorithm'], config['principal_symbol'])
      