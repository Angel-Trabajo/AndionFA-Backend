# pipeline_completo.py
# Contiene: Signal + Normalizar + data_for_neuronal + clean_majority
# Ensamblado listo para ejecutar como pipeline.

import sys
import os
import json
import sqlite3
import ast
import operator
import pandas as pd



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
    def __init__(self, algorithms, principal_symbol, signal):
        self.algorithms = algorithms
        self.principal_symbol = principal_symbol
        self.signal = signal

    def normalize_close_signals(self):
        signal = self.signal.get_close_signals()
        if signal is None:
            return None
        asign = {elem: [int(b) for b in bin(i)[2:].zfill(9)] for i, elem in enumerate(signal, start=1)}
        with open('src/neuronal/data/maping_close.json', 'w') as file:
            json.dump(asign, file, indent=4)
        return asign

    def normalize_open_signals(self):
        signal = self.signal.get_open_signals()
        if signal is None:
            return None
        asign = {elem: [int(b) for b in bin(i)[2:].zfill(9)] for i, elem in enumerate(signal, start=1)}
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

def cumple_condiciones(fila, condiciones):
    return all(operadores[op](fila[col], valor) for col, op, valor in condiciones if col in fila)

# ================================
#   Obtener operaciones abiertas
# ================================
def get_operation_open(principal_symbol, last_symbol, algorithm):
    def get_nodes():
        conn = sqlite3.connect(f'output/db/crossing_{principal_symbol}_dbs/{last_symbol}.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM nodes WHERE label = ?', (algorithm,))
        r = cursor.fetchall()
        conn.close()
        return r if r else None

    nodes = get_nodes()
    if not nodes:
        return []

    ids_conditions = [(n[0], n[2], n[3]) for n in nodes if n[3] is not None]
    list_orders = []

    for node_id, name_file, condition in ids_conditions:
        conn = sqlite3.connect(f'output/db/crossing_{principal_symbol}_dbs/{last_symbol}.db')
        cursor = conn.cursor()
        cursor.execute("""
            SELECT dates FROM register WHERE node_id = ?
            UNION ALL
            SELECT dates_os FROM register_os WHERE node_id = ?
        """, (node_id, node_id))
        dates = cursor.fetchall()
        conn.close()

        if dates:
            for date in dates:
                list_orders.append((date[0], name_file, condition))

    return list_orders

# ================================
#   Generación de dataset neuronal
# ================================
def data_for_neuronal(algorithm, principal_symbol):
    # Cargar config
    with open('config/list_{}.json'.format(algorithm), 'r') as f:
        last_symbol = json.load(f)['list'][-1]

    # Inicializar clases
    signals = Signal(algorithm, principal_symbol)
    _, _, _, last_symbol = signals.get_info()
    normal = Normalizar(algorithm, principal_symbol, signals)

    # Normalizaciones
    close_norm = normal.normalize_close_signals()
    open_norm = normal.normalize_open_signals()

    # Operaciones abiertas
    open_ops = get_operation_open(principal_symbol, last_symbol, algorithm)
    if not open_ops:
        print("⚠ No hay operaciones abiertas.")
        return

    # Dataset primitivo
    df_is = pd.read_csv('output/is_os/is.csv')
    df_os = pd.read_csv('output/is_os/os.csv')
    df_primitiva = (
        pd.concat([df_is, df_os], ignore_index=True)
        .drop_duplicates(subset=['time'])
        .sort_values(by='time', ignore_index=True)
    )
    df_primitiva["time"] = pd.to_datetime(df_primitiva["time"])
    # Cargar indicadores
    list_files_is = os.listdir('output/extrac/')
    list_files_os = os.listdir('output/extrac_os/')
    indicadores_dict = {}

    for name_is, name_os in zip(list_files_is, list_files_os):
        df1 = pd.read_parquet(f'output/extrac/{name_is}')
        df2 = pd.read_parquet(f'output/extrac_os/{name_os}')
        df = pd.concat([df1, df2], ignore_index=True).drop_duplicates(subset=['time'])

        # 🔽 MEJORA 2: evitar leakage de indicadores
        df = df.sort_values("time").reset_index(drop=True)
        df.iloc[:, 1:] = df.iloc[:, 1:].shift(1)  # "time" es la primera columna

        indicadores_dict[name_is] = df

    list_codes = []

    # Generar dataset neuronal
    for date, name_file, cond_str in open_ops:
        code1 = open_norm.get(cond_str)
        if code1 is None:
            continue

        key_name = name_file.replace(last_symbol, principal_symbol)
        df_ind = indicadores_dict.get(key_name)
        if df_ind is None or df_ind.empty:
            continue

        idxs = df_primitiva.index[df_primitiva['time'] == date].tolist()
        if not idxs:
            continue
        idx = idxs[0]

        rest = df_primitiva.iloc[idx + 1: idx + 20].copy()
        if rest.empty:
            continue

        rest.sort_values(by='open', ascending=False, inplace=True)
        rest = rest.head(1) if algorithm == "UP" else rest.tail(1)

        # 🔽 MEJORA 3: penalizar cierres tardíos
        close_idx = rest.index[0]
        delta = close_idx - idx           # cuántas velas tardó en cerrarse
        penalty = 1 - (delta / 20)        # 1 = inmediato, 0 = muy tardío
        merged = pd.merge(rest, df_ind, on='time', how='inner')
        if merged.empty:
            continue

        for _, fila in merged.iterrows():
            for key, code2 in close_norm.items():
                try:
                    conds = ast.literal_eval(key)
                except:
                    continue
                out = int(
                    cumple_condiciones(fila, conds)
                    and abs(fila["close"] - fila["open"]) > 0.0002
                    and penalty > 0.1      # ❌ penalización de cierre tardío
                )
                list_codes.append((code1, code2, out))

    if not list_codes:
        print("⚠ No se generaron datos neuronales.")
        return

    df = pd.DataFrame(list_codes, columns=["input1", "input2", "output"])
    path = f"src/neuronal/data/data_{algorithm}_{principal_symbol}.csv"
    df.to_csv(path, index=False)
    print(f"✅ Dataset neuronal generado: {path}")

# ================================
#   Limpieza — Opción A (mayoría)
# ================================
def clean_majority(algorithm, principal_symbol):
    path = f"src/neuronal/data/data_{algorithm}_{principal_symbol}.csv"

    data = pd.read_csv(path)
    data["input1_t"] = data["input1"].apply(ast.literal_eval).apply(tuple)
    data["input2_t"] = data["input2"].apply(ast.literal_eval).apply(tuple)

    def majority(grp):
        if len(grp) < 10:
            return None
        return grp["output"].value_counts().idxmax()

    cleaned = (
        data.groupby(["input1_t", "input2_t"])
        .apply(majority)
        .reset_index()
        .rename(columns={0: "output"})
        .dropna(subset=["output"])
    )


    out = f"src/neuronal/data/data_cleaned_{algorithm}_{principal_symbol}.csv"
    cleaned.to_csv(out, index=False)
    print(f"✅ Datos limpiados por mayoría: {out}")
    return cleaned

# ================================
#   MAIN PIPELINE
# ================================
if __name__ == "__main__":
    with open('config/config_crossing/config_crossing.json', 'r') as f:
        config_cross = json.load(f)
    principal_symbol = config_cross['principal_symbol']

    with open('config/config_test/config_test_red.json', 'r') as f:
        config_test = json.load(f)
    algorithm = config_test['algorithm']

    print("🚀 Ejecutando pipeline completo...")
    data_for_neuronal(algorithm, principal_symbol)
    clean_majority(algorithm, principal_symbol)
    print("🎉 Pipeline finalizado.")