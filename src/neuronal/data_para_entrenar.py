# pipeline_completo.py
# Contiene: Signal + Normalizar + data_for_neuronal + clean_majority
# Ensamblado listo para ejecutar como pipeline.

import sys
import ast
import os
import json
import operator
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.common_functions import crear_carpeta_si_no_existe
from src.db import query as db_query


# ================================
#   Clase Signal
# ================================
class Signal:
    def __init__(self, algorithm, principal_symbol, mercado, config):
        self.algorithm = algorithm
        self.principal_symbol = principal_symbol
        self.mercado = mercado
        self.config = config
        _, _, other_algorithm = self.get_info()
        self.other_algorithm = other_algorithm

    def get_info(self):
        if self.algorithm == "UP":
            other_algorithm = "DOWN"
        else:
            other_algorithm = "UP"
        return self.algorithm, self.principal_symbol, other_algorithm

    def get_close_signals(self):
        
        nodes = db_query.get_nodes(
            principal_symbol=self.principal_symbol, 
            symbol_cruce=self.principal_symbol, 
            mercado=None, 
            label=self.other_algorithm
            )
        return [(n[6]) for n in nodes] if nodes else None

    def get_open_signals(self):
        nodes = db_query.get_nodes(
            principal_symbol=self.principal_symbol, 
            symbol_cruce=self.principal_symbol, 
            mercado=self.mercado, 
            label=self.algorithm
            )
        return [n[6] for n in nodes] if nodes else None

# ================================
#   Clase Normalizar
# ================================
class Normalizar:
    def __init__(self, signal):
        self.signal = signal

    def _mapping_path(self, kind):
        return f'output/{self.signal.principal_symbol}/data_for_neuronal/maping/maping_{kind}_{self.signal.mercado}_{self.signal.algorithm}.json'

    def _write_mapping(self, kind, data):
        with open(self._mapping_path(kind), 'w') as file:
            json.dump(data, file, indent=4)

    def normalize_close_signals(self):
        signal = self.signal.get_close_signals()
        if signal is None:
            self._write_mapping('close', {})
            return {}
        
      
        count = 8   
        asign = {}
        for i, elem in enumerate(signal):
            # Normalizar siempre
            if isinstance(elem, str):
                elem = ast.literal_eval(elem)

            # Clave estable (NO str())
            key = json.dumps(elem, sort_keys=True)
            asign[key] = bin(i+1)[2:].zfill(count)  # Convertir a binario de 8 bits

        self._write_mapping('close', asign)

        return asign

    def normalize_open_signals(self):
        signal = self.signal.get_open_signals()
        if signal is None:
            self._write_mapping('open', {})
            return {}
        
      
        count = 8
        asign = {}
        for i, elem in enumerate(signal):
            # Normalizar siempre
            if isinstance(elem, str):
                elem = ast.literal_eval(elem)

            # Clave estable (NO str())
            key = json.dumps(elem, sort_keys=True)
            asign[key] = bin(i+1)[2:].zfill(count)  # Convertir a binario de 8 bits

        self._write_mapping('open', asign)

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


def data_for_neuronal(config, mercado, algorithm, dict_pips_best= None):
    
    principal_symbol = config['principal_symbol']
    signal = Signal(algorithm, principal_symbol, mercado, config)
    normalizar = Normalizar(signal)
    normalizar.normalize_close_signals()
    normalizar.normalize_open_signals()
    data = {
        'input1': [],
        'input2': [],
        'hour': [],
        'output': []
    }

    if dict_pips_best:
        for key, value in dict_pips_best.items():
            key_parts = key.split('_')
            
            if value > 5:
                acaled = 1
            elif value < -5 :
                acaled = 0
            else:
                continue
            data['input1'].append(key_parts[0])  
            data['input2'].append(key_parts[1])  
            data['hour'].append(key_parts[2])
            data['output'].append(acaled)
          # Agregar la hora correspondiente
    df = pd.DataFrame(data)
    df.to_csv(f'output/{principal_symbol}/data_for_neuronal/data/data_{mercado}_{algorithm}.csv', index=False)


def execute_data_for_neuronal(principal_symbol, mercados, list_algorithms = None, dict_pips_best= {}):
    crear_carpeta_si_no_existe(f'output/{principal_symbol}/data_for_neuronal')
    crear_carpeta_si_no_existe(f'output/{principal_symbol}/data_for_neuronal/data')
    crear_carpeta_si_no_existe(f'output/{principal_symbol}/data_for_neuronal/maping')
    crear_carpeta_si_no_existe(f'output/{principal_symbol}/data_for_neuronal/best_score')
    
    with open(f'config/divisas/{principal_symbol}/config_{principal_symbol}.json', 'r', encoding='utf-8') as f:
        config_symbol = json.load(f)
    with open(f'config/general_config.json', 'r', encoding='utf-8') as f:
        general_config = json.load(f)
        
    config = {
        "general": general_config,
        "symbol": config_symbol,
        "principal_symbol": principal_symbol
    }
    list_algorithms = ["UP", "DOWN"] if list_algorithms is None else list_algorithms
    for algorithm in list_algorithms:
        for mercado in mercados:
            data_for_neuronal(config, mercado, algorithm, dict_pips_best)
    
    
if __name__ == "__main__":
    execute_data_for_neuronal('AUDCHF', ['Asia'], list_algorithms = None, dict_pips_best= None)
      