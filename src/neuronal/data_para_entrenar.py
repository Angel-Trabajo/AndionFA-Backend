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
        _, _, other_algorithm, last_symbol = self.get_info()
        self.other_algorithm = other_algorithm
        self.last_symbol = last_symbol

    def get_info(self):
        if self.algorithm == "UP":
            other_algorithm = "DOWN"
        else:
            other_algorithm = "UP"
        last_symbol = self.config['symbol']['list_symbol'][-1]
        return self.algorithm, self.principal_symbol, other_algorithm, last_symbol

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
            symbol_cruce=self.last_symbol, 
            mercado=None, 
            label=self.algorithm
            )
        return [n[6] for n in nodes] if nodes else None

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
        
        
        count = 8   
        asign = {}
        for i, elem in enumerate(signal):
            # Normalizar siempre
            if isinstance(elem, str):
                elem = ast.literal_eval(elem)

            # Clave estable (NO str())
            key = json.dumps(elem, sort_keys=True)
            asign[key] = bin(i+1)[2:].zfill(count)  # Convertir a binario de 8 bits

        with open(f'output/{self.signal.principal_symbol}/data_for_neuronal/maping/maping_close_{self.signal.mercado}_{self.signal.algorithm}.json', 'w') as file:
            json.dump(asign, file, indent=4)

        return asign

    def normalize_open_signals(self):
        signal = self.signal.get_open_signals()
        if signal is None:
            return None
        
        count = 8
        asign = {}

        for i, elem in enumerate(signal):
            # Normalizar siempre
            if isinstance(elem, str):
                elem = ast.literal_eval(elem)

            # Clave estable (NO str())
            key = json.dumps(elem, sort_keys=True)
            asign[key] = bin(i+1)[2:].zfill(count)  # Convertir a binario de 8 bits

        with open(f'output/{self.signal.principal_symbol}/data_for_neuronal/maping/maping_open_{self.signal.mercado}_{self.signal.algorithm}.json', 'w') as file:
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


def data_for_neuronal(config, mercado, algorithm, dict_pips_best=None, trade_samples=None):
    if dict_pips_best is None:
        dict_pips_best = {}
    if trade_samples is None:
        trade_samples = []
    
    principal_symbol = config['principal_symbol']
    signal = Signal(algorithm, principal_symbol, mercado, config)
    normalizar = Normalizar(signal)
    sign_close = normalizar.normalize_close_signals()
    sign_open = normalizar.normalize_open_signals()
    data = {
        'input1': [],
        'input2': [],
        'time_open': [],
        'time_close': [],
        'atr': [],
        'adx': [],
        'rsi': [],
        'stoch': [],
        'hour': [],
        'spread': [],
        'atr_open': [],
        'adx_open': [],
        'rsi_open': [],
        'stoch_open': [],
        'output': [],
        # Nuevos campos para Cambio 2: Tracking de decisiones
        'executed': [],
        'reason': [],
    }

    if not trade_samples:
        print(f"[data_for_neuronal] Sin trade_samples para {mercado}/{algorithm}, saltando generación de dataset.")
        return

    MIN_ABS_PIPS = 2.0
    print(f"[data_for_neuronal] Samples antes de filtrar: {len(trade_samples)}")
    skipped_micro = 0

    for sample in trade_samples:
        open_signal = str(sample.get('open_signal', ''))
        close_signal = str(sample.get('close_signal', ''))

        if not open_signal or not close_signal:
            continue

        profit = float(sample.get('profit', 0.0) or 0.0)  # target futuro (MFE-|MAE|)
        profit_real = float(sample.get('profit_real', profit) or 0.0)  # pips reales ejecutados
        if abs(profit_real) < MIN_ABS_PIPS:
            skipped_micro += 1
            continue

        data['input1'].append(open_signal)
        data['input2'].append(close_signal)
        data['time_open'].append(str(sample.get('time_open', '') or ''))
        data['time_close'].append(str(sample.get('time_close', '') or ''))
        data['atr'].append(float(sample.get('atr', 0.0) or 0.0))
        data['adx'].append(float(sample.get('adx', 0.0) or 0.0))
        data['rsi'].append(float(sample.get('rsi', 0.0) or 0.0))
        data['stoch'].append(float(sample.get('stoch', 50.0) or 50.0))
        data['hour'].append(float(sample.get('hour', 0.0) or 0.0))
        data['spread'].append(float(sample.get('spread', 0.0) or 0.0))
        data['atr_open'].append(float(sample.get('atr_open', 0.0) or 0.0))
        data['adx_open'].append(float(sample.get('adx_open', 0.0) or 0.0))
        data['rsi_open'].append(float(sample.get('rsi_open', 0.0) or 0.0))
        data['stoch_open'].append(float(sample.get('stoch_open', 50.0) or 50.0))

        # Target continuo lineal (sin tanh) para preservar resolución
        profit_scaled = float(profit / 50.0)
        data['output'].append(profit_scaled)
        
        # Nuevos campos: tracking de decisión
        data['executed'].append(int(sample.get('executed', 0)))  # 1 si fue ejecutada
        data['reason'].append(str(sample.get('reason', 'UNKNOWN') or 'UNKNOWN'))  # Razón del cierre

    df = pd.DataFrame(data)
    if df.empty:
        print(f"[data_for_neuronal] ⚠️ Dataset vacío para {mercado}/{algorithm}, no se guarda")
        return

    required_cols = [
        'input1', 'input2', 'time_open', 'time_close', 'atr', 'adx', 'rsi', 'stoch', 'hour', 'spread',
        'atr_open', 'adx_open', 'rsi_open', 'stoch_open', 'output',
        # Nuevas columnas para Cambio 2
        'executed', 'reason'
    ]
    for col in required_cols:
        if col not in df.columns:
            if col in ['input1', 'input2', 'time_open', 'time_close', 'reason']:
                df[col] = ''
            else:
                df[col] = 0.0 if col not in ['executed'] else 0

    # Mantener solo decisiones realmente ejecutadas para evitar contaminación de labels
    if 'executed' in df.columns:
        df = df[df['executed'].astype(int) == 1]
        if df.empty:
            print(f"[data_for_neuronal] ⚠️ Sin filas ejecutadas para {mercado}/{algorithm}, no se guarda")
            return

    print(f"[data_for_neuronal] Micro trades descartados (<{MIN_ABS_PIPS} pip): {skipped_micro}")
    print(f"[data_for_neuronal] Samples después de filtrar: {len(df)}")
    dataset_path = f'output/{principal_symbol}/data_for_neuronal/data/data_{mercado}_{algorithm}.csv'
    df = df[required_cols]

    # Si hay histórico, hacer merge para evitar overwrite destructivo.
    if os.path.exists(dataset_path):
        try:
            old_df = pd.read_csv(dataset_path)
            for col in required_cols:
                if col not in old_df.columns:
                    old_df[col] = '' if col in ['input1', 'input2', 'time_open', 'time_close'] else 0.0
            old_df = old_df[required_cols]
            merged = pd.concat([old_df, df], ignore_index=True)
        except Exception as e:
            print(f"[data_for_neuronal] ⚠️ No se pudo leer dataset previo, se reemplaza ({e})")
            merged = df.copy()
    else:
        merged = df.copy()

    # Deduplicar preferentemente por fecha de cierre + señales.
    dedupe_keys = ['time_close', 'input1', 'input2']
    if all(k in merged.columns for k in dedupe_keys):
        merged = merged.drop_duplicates(subset=dedupe_keys, keep='last')
    else:
        merged = merged.drop_duplicates(keep='last')

    # Si es primer archivo y todavía no hay masa crítica, evitar crear dataset débil.
    if not os.path.exists(dataset_path) and len(merged) < 5:
        print(f"[data_for_neuronal] ⚠️ Muy pocos samples iniciales ({len(merged)}), no se guarda dataset aún")
        return

    merged.to_csv(dataset_path, index=False)
    print(f"[data_for_neuronal] Dataset guardado: {dataset_path} | filas={len(merged)}")


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
    execute_data_for_neuronal('AUDCAD', ['Asia'], list_algorithms = None, dict_pips_best= {})
      