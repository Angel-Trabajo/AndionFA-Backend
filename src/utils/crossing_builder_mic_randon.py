import json
import os
import sys
import operator
import subprocess
import struct
import uuid
import time 


from concurrent.futures import ProcessPoolExecutor
from multiprocessing import (
    Process,
    Queue,
    Manager,
    freeze_support
)

import pandas as pd
import numpy as np

# ==========================================================
# PATH
# ==========================================================

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from src.routes import peticiones
from src.db.create_db import create_db
from src.db import query as db_query
from src.utils.crossing_funtion.crear_indicadores_in_crossing import extract_indicadores
from src.utils.crossing_funtion.constructor_node import NodeGenerator



# ==========================================================
# CONSTANTES GLOBALES
# ==========================================================

MIC_DEVICES = [0, 1, 2, 3]
MIC_STATUS = None

replaceString = {
    "SMA": 0,
    "EMA": 1,
    "WMA": 2,
    "DEMA": 3,
    "TEMA": 4,
    "TRIMA": 5,
    "KAMA": 6,
    "MAMA": 7,
    "T3": 8
}

OP_MAP = {
    '>': operator.gt,
    '<': operator.lt,
    '>=': operator.ge,
    '<=': operator.le,
    '==': operator.eq,
    '!=': operator.ne
}



with open('config/config_crossing/config_crossing.json', 'r') as file:
    config = json.load(file)

with open('config/config_node/config_node.json', encoding='utf-8') as f:
    config_node = json.load(f)

principal_symbol = config['principal_symbol']
timeframe = config['timeframe']
list_symbol = config['list_symbol']
min_operaciones = config['min_operaciones']
por_direccion = config['por_direccion']
list_symbols_inversos = config['list_symbol_inversos']
dict_symbol_correl = config['dict_symbol_correl']

NumMax_Operations = config_node['NumMaxOperations']



# ==========================================================
# CACHE DE DATAFRAMES
# ==========================================================

_DF_CACHE = {}


def load_csv_cached(path):
    if path not in _DF_CACHE:
        _DF_CACHE[path] = pd.read_csv(
            path,
            parse_dates=["time"],
            memory_map=True
        )
    return _DF_CACHE[path]


# ==========================================================
# UTILIDADES GENERALES
# ==========================================================

def calcular_descuento(N0, Nf, k):
    """
    Calcula el porcentaje de descuento por iteración.
    """
    if N0 <= 0 or Nf <= 0 or k <= 0:
        raise ValueError("N0, Nf y k deben ser mayores que cero")

    return 1 - (Nf / N0) ** (1 / k)


def replaceStringOp(op):
    return {
        '>': 0,
        '<': 1,
        '>=': 2,
        '<=': 3,
        '==': 4,
        '!=': 5
    }[op]


def preparar_condiciones(conditions, col_index):
    """
    Precompila condiciones para evaluación rápida
    """
    compiled = []
    for col, op, value in conditions:
        compiled.append(
            (col_index[col], OP_MAP[op], value)
        )
    return compiled

# ==========================================================
# EVALUACIÓN DE CONDICIONES (VECTORIZADA)
# ==========================================================

def evaluar_condiciones_vectorizado(
    matrix,
    compiled_conditions
):
    if matrix.shape[0] == 0:
        return np.array([], dtype=bool)

    mask = np.ones(
        matrix.shape[0],
        dtype=bool
    )

    for idx, op_func, value in compiled_conditions:
        mask &= op_func(
            matrix[:, idx],
            value
        )

        if not mask.any():
            break

    return mask


def dataframe_to_matrix(df, condition_cols):
    """
    Convierte DataFrame a matriz numérica ordenada
    por columnas usadas en condiciones
    """
    return df[condition_cols].to_numpy(
        dtype=np.float64,
        copy=True
    )


# ==========================================================
# MIC – EVALUACIÓN REMOTA
# ==========================================================

def evaluar_batch_mic_remote(remote_matrix, list_compiled_nodes, mic_id):

    uid = uuid.uuid4().hex
    host = f"root@mic{mic_id}"

    cond = f"/tmp/cond_{uid}.bin"
    mask = f"/tmp/mask_{uid}.bin"

    # escribir condiciones
    with open("cond.bin", "wb") as f:
        f.write(struct.pack("i", len(list_compiled_nodes)))
        for compiled in list_compiled_nodes:
            f.write(struct.pack("i", len(compiled)))
            for col, op, val in compiled:
                f.write(struct.pack("iid", col, op, val))

    with open("cond.bin", "rb") as f:
        subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", host, f"cat > {cond}"],
            stdin=f, check=True
        )

    # ejecutar MIC
    cmd = [
        "micnativeloadex",
        "evaluar_batch.mic",
        "-d", str(mic_id),
        "-a", f"{remote_matrix} {cond} {mask}"
    ]

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stdout + r.stderr)

    # leer salida
    out = subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", host, f"cat {mask}"],
        capture_output=True, check=True
    ).stdout

    # limpiar
    subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", host, "rm", "-f", cond, mask],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # parsear
    buf = memoryview(out)
    offset = 0

    num_nodes = struct.unpack_from("i", buf, offset)[0]
    offset += 4

    masks = []

    for _ in range(num_nodes):
        rows = struct.unpack_from("i", buf, offset)[0]
        offset += 4

        mask_arr = np.frombuffer(
            buf[offset:offset + rows],
            dtype=np.uint8
        ).astype(bool)

        offset += rows
        masks.append(mask_arr)

    return masks



def subir_matriz_mic(
    matrix,
    mic_id,
    tag
):
    host = f"root@mic{mic_id}"
    rows, cols = matrix.shape

    local = f"matrix_{tag}.bin"
    remote = f"/tmp/{local}"

    with open(local, "wb") as f:
        f.write(
            struct.pack(
                "ii",
                rows,
                cols
            )
        )
        f.write(
            matrix.astype(
                np.float64,
                copy=False
            ).tobytes()
        )

    with open(local, "rb") as f:
        subprocess.run(
            [
                "ssh",
                "-o",
                "StrictHostKeyChecking=no",
                host,
                f"cat > {remote}"
            ],
            stdin=f,
            check=True
        )

    os.remove(local)
    return remote

# ==========================================================
# SELECCIÓN Y EVALUACIÓN DE NODOS
# ==========================================================

def selecte_nodes(
    file: str,
    symbol,
    action,
    cont,
    list_symbol,
    node_generator,
    mic_id,
    prev_os, 
    prev_is,
    porcent_aumento_os,
    porcent_aumento_is
):
    pip_mult = 100 if 'JPY' in symbol.upper() else 10_000

    # -------------------------------------------------
    # Filtrado por dirección
    # -------------------------------------------------
    ini = time.time()
    list_nodos = node_generator.generar_nodos(10000)
    print(f'10000 nodos creados en {time.time()-ini} segundos')        
    if por_direccion:
        if symbol in list_symbols_inversos:
            list_nodos = [n for n in list_nodos if n['label'] != action]
        else:
            list_nodos = [n for n in list_nodos if n['label'] == action]

    if not list_nodos:
        return
    
    # -------------------------------------------------
    # Cargar indicadores
    # -------------------------------------------------
    ext = file.split('_')[0]

    list_dire_indicators_os = os.listdir(
        f'output/crossing_{principal_symbol}/{symbol}/extrac_os'
    )
    dire = [f for f in list_dire_indicators_os if ext in f][0]

    df_indicators_os = pd.read_csv(
        f'output/crossing_{principal_symbol}/{symbol}/extrac_os/{dire}',
        parse_dates=["time"]
    )

    df_indicators_is = pd.read_csv(
        f'output/crossing_{principal_symbol}/{symbol}/extrac/{file}',
        parse_dates=["time"]
    )

    df_os = load_csv_cached('output/is_os/os.csv')
    df_is = load_csv_cached('output/is_os/is.csv')

    os_time_np = df_os['time'].to_numpy()
    is_time_np = df_is['time'].to_numpy()

    # -------------------------------------------------
    # Fechas previas
    # -------------------------------------------------
    if cont == 0:
        list_oper_os = list(db_query.get_dates_by_label(principal_symbol, action, 'os'))
        list_oper_is = list(db_query.get_dates_by_label(principal_symbol, action, 'is'))
    else:
        prev = f'crossing_{principal_symbol}_dbs/{list_symbol[cont-1]}'
        list_oper_os = list(db_query.get_dates_by_label(prev, action, 'os'))
        list_oper_is = list(db_query.get_dates_by_label(prev, action, 'is'))

    fechas_dt_os = pd.to_datetime(list_oper_os)
    fechas_dt_is = pd.to_datetime(list_oper_is)

    idx_os = df_indicators_os.index[df_indicators_os['time'].isin(fechas_dt_os)]
    idx_is = df_indicators_is.index[df_indicators_is['time'].isin(fechas_dt_is)]

    df_ind_os_fil = df_indicators_os.iloc[idx_os[idx_os > 0] - 1]
    df_ind_is_fil = df_indicators_is.iloc[idx_is[idx_is > 0] - 1]

    # -------------------------------------------------
    # Arrays OS
    # -------------------------------------------------
    os_open = df_os['open'].to_numpy()
    os_close = df_os['close'].to_numpy()
    os_spread = df_os['spread'].to_numpy()
    os_hour = df_os['time'].dt.hour.to_numpy()
    ind_os_time = df_ind_os_fil['time'].to_numpy()

    # -------------------------------------------------
    # Arrays IS
    # -------------------------------------------------
    is_open = df_is['open'].to_numpy()
    is_close = df_is['close'].to_numpy()
    is_spread = df_is['spread'].to_numpy()
    is_hour = df_is['time'].dt.hour.to_numpy()
    ind_is_time = df_ind_is_fil['time'].to_numpy()

    # -------------------------------------------------
    # Columnas usadas por condiciones
    # -------------------------------------------------
    cond_cols = []
    seen = set()
    for nodo in list_nodos:
        for col, _, _ in nodo['conditions']:
            if col not in seen:
                seen.add(col)
                cond_cols.append(col)

    matrix_os = dataframe_to_matrix(df_ind_os_fil, cond_cols)
    matrix_is = dataframe_to_matrix(df_ind_is_fil, cond_cols)

    col_index = {c: i for i, c in enumerate(cond_cols)}

    compiled_nodes = [
        preparar_condiciones_mic(n['conditions'], col_index)
        for n in list_nodos
    ]

    # -------------------------------------------------
    # Ejecutar MIC
    # -------------------------------------------------
    uid = uuid.uuid4().hex

    remote_os = subir_matriz_mic(matrix_os, mic_id, f"os_{uid}")
    remote_is = subir_matriz_mic(matrix_is, mic_id, f"is_{uid}")

    masks_os = evaluar_batch_mic_remote(remote_os, compiled_nodes, mic_id)
    masks_is = evaluar_batch_mic_remote(remote_is, compiled_nodes, mic_id)
    print(len(list_nodos), f"nodos evaluados en la MIC-{mic_id}")
    subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", f"root@mic{mic_id}",
         "rm", "-f", remote_os, remote_is],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # -------------------------------------------------
    # Evaluar nodos (LÓGICA ORIGINAL)
    # -------------------------------------------------
    for nodo, mask_os, mask_is in zip(list_nodos, masks_os, masks_is):

        # ===================== OS =====================
        list_dates_os = []
        list_beneficio_os = []

        for i in np.where(mask_os)[0]:
            date = ind_os_time[i]
            
            pos = np.searchsorted(os_time_np, date)

            if (
                pos >= len(os_time_np)
                or os_time_np[pos] != date
                or pos + 1 >= len(df_os)
                or os_hour[pos + 1] == 0
            ):
                continue


            if action == 'UP':
                beneficio = os_close[pos + 1] - os_open[pos + 1]
            else:
                beneficio = os_open[pos + 1] - os_close[pos + 1]

            bruto = beneficio * pip_mult
            spread = os_spread[pos + 1] / 10
            beneficio = bruto - spread

            list_dates_os.append(
                pd.to_datetime(os_time_np[pos + 1]).strftime("%Y-%m-%d %H:%M:%S")
            )
            list_beneficio_os.append(beneficio)

        total_os = len(list_beneficio_os)
        if total_os == 0:
            continue

        aciertos_os = sum(1 for b in list_beneficio_os if b > 0)
        porcentaje_os = aciertos_os / total_os

        if not (prev_os + (porcent_aumento_os - 0.02) <= porcentaje_os <= prev_os + (porcent_aumento_os + 0.02)):
            continue

        # ===================== IS =====================
        list_dates_is = []
        list_beneficio_is = []

        for i in np.where(mask_is)[0]:
            date = ind_is_time[i]
            pos = np.searchsorted(is_time_np, date)

            if pos + 1 >= len(df_is) or is_hour[pos + 1] == 0:
                continue

            if action == 'UP':
                beneficio = is_close[pos + 1] - is_open[pos + 1]
            else:
                beneficio = is_open[pos + 1] - is_close[pos + 1]

            bruto = beneficio * pip_mult
            spread = is_spread[pos + 1] / 10
            beneficio = bruto - spread

            list_dates_is.append(
                pd.to_datetime(is_time_np[pos + 1]).strftime("%Y-%m-%d %H:%M:%S")
            )
            list_beneficio_is.append(beneficio)

        total_is = len(list_beneficio_is)
        if total_is == 0:
            continue

        aciertos_is = sum(1 for b in list_beneficio_is if b > 0)
        porcentaje_is = aciertos_is / total_is


        if not (prev_is + (porcent_aumento_is - 0.01) <= porcentaje_is <= prev_is + (porcent_aumento_os + 0.01)):
            continue
        
        
        
        progressive_is = total_is / len(df_indicators_is)
        progressive_os = total_os / len(df_indicators_os)

        progressiveVariation = abs(progressive_os - progressive_is)

        if progressiveVariation > config_node['ProgressiveVariation']:
            continue
        
        nodo_mas_parecido = db_query.nodo_con_mas_fechas_hora_comunes(
            f'crossing_{principal_symbol}_dbs/{symbol}',
            list_dates_is
        )

        if nodo_mas_parecido:
            coincidencias = nodo_mas_parecido['coincidencias']
            total_operaciones = nodo_mas_parecido['total_operations']

            porciento_nodo_db = coincidencias / total_operaciones
            porciento_is = coincidencias / total_is
            porciento = (porciento_nodo_db + porciento_is) / 2

            if (
                porciento >= config_node['SimilarityMax']
                and nodo_mas_parecido['total_operations'] < total_is
            ):
                db_query.eliminar_nodo_y_registros(
                    f'crossing_{principal_symbol}_dbs/{symbol}',
                    nodo_mas_parecido['node_id']
                )
            elif porciento >=config_node['SimilarityMax'] and nodo_mas_parecido['total_operations'] >= total_is: 
                print('Mayor pero el de la db mejor')
                continue

        # -------------------------------------------------
        # Insertar nodo (igual que original)
        # -------------------------------------------------
        db_query.insertar_nodo_con_registros(
            name=f'crossing_{principal_symbol}_dbs/{symbol}',
            label=action,
            file_in_db=file,
            conditions=json.dumps(nodo['conditions'], sort_keys=True),
            correct_percentage=porcentaje_is,
            successful_operations=aciertos_is,
            total_operations=total_is,
            correct_percentage_os=porcentaje_os,
            successful_operations_os=aciertos_os,
            total_operations_os=total_os,
            fechas=list_dates_is,
            veneficios=list_beneficio_is,
            fechas_os=list_dates_os,
            veneficios_os=list_beneficio_os
        )



# ==========================================================
# PROCESAMIENTO DE ARCHIVOS (WORKER)
# ==========================================================

def procesar_archivo(file: str, symbol, action, NumMaxOperations, cont, list_symbol, mic_id, prev_os, prev_is, porcent_aumento_os, porcent_aumento_is):
    
    df = pd.read_csv(f'output/crossing_{principal_symbol}/{symbol}/extrac/{file}')
    node_generator = NodeGenerator(df)
    operaciones_exitosas = 0
    while (operaciones_exitosas < NumMaxOperations):
    
        try:
            selecte_nodes(
                file,
                symbol,
                action,
                cont,
                list_symbol,
                node_generator,
                mic_id,
                prev_os, 
                prev_is,
                porcent_aumento_os,
                porcent_aumento_is
            )
        except RuntimeError as e:
            if "Unable to attach" in str(e):
                print(
                    f"[MIC{mic_id}] fuera de servicio, "
                    "se desactiva temporalmente"
                )
                MIC_STATUS[mic_id] = False
                raise

        operaciones_exitosas = (
            db_query.successful_operations_by_label(
                f'crossing_{principal_symbol}_dbs/{symbol}',
                action
            )
        )

        print(
            f"Operaciones exitosas "
            f"{symbol}-{action}: "
            f"{operaciones_exitosas}"
        )


# ==========================================================
# CREACIÓN DE ÁRBOLES (ORQUESTADOR)
# ==========================================================

def create_trees(symbol, action, NumMaxOperations, cont, list_symbol, prev_os, prev_is):
    db_path = f'output/db/crossing_{principal_symbol}_dbs/{symbol}.db'

    if not os.path.exists(db_path):
        create_db(f'crossing_{principal_symbol}_dbs/{symbol}')

    list_files = os.listdir(
        f'output/crossing_{principal_symbol}/{symbol}/extrac'
    )
    
    porcent_aumento_os =calcular_porcentage(symbol, prev_os)
    porcent_aumento_is =calcular_porcentage(symbol, prev_is)
    
    MAX_PROCESOS = 20  # ajustable según CPU

    with ProcessPoolExecutor(
        max_workers=MAX_PROCESOS,
    ) as executor:

        for i, file in enumerate(list_files):
            mic_id = MIC_DEVICES[i % 4]
            executor.submit(
                procesar_archivo,
                file,
                symbol,
                action,
                NumMaxOperations,
                cont,
                list_symbol,
                mic_id, 
                prev_os,
                prev_is,
                porcent_aumento_os,
                porcent_aumento_is
            )

    # Nodo final de cierre
    db_query.insertar_nodo_con_registros(
        name=f'crossing_{principal_symbol}_dbs/{symbol}',
        label='END',
        file_in_db='001maco',
        conditions='APO > 2',
        correct_percentage=0.53,
        successful_operations=251,
        total_operations=491,
        correct_percentage_os=0.53,
        successful_operations_os=251,
        total_operations_os=491,
        fechas=[''],
        veneficios=[0],
        fechas_os=[''],
        veneficios_os=[0]
    )


# ==========================================================
# HELPERS MIC (COMPILACIÓN CONDICIONES)
# ==========================================================

def replace_op_code(op):
    return {
        '>': 0,
        '<': 1,
        '>=': 2,
        '<=': 3,
        '==': 4,
        '!=': 5
    }[op]


def preparar_condiciones_mic(conditions, col_index):
    compiled = []
    for col, op, value in conditions:
        compiled.append(
            (col_index[col], replace_op_code(op), value)
        )
    return compiled


# ==========================================================
# EJECUTOR PRINCIPAL POR DIRECCIÓN
# ==========================================================
def calcular_porcentage(symbol, prev):
    sumatoria = 0
    for key, value in dict_symbol_correl.items():
        sumatoria += value
    corre = dict_symbol_correl[symbol]
    return corre/sumatoria * (1-prev)


def execute_crossing_builder(action, mic_status):
    global MIC_STATUS
    MIC_STATUS = mic_status

    cont = 0
    prosedio = True
    NumMaxOperations = NumMax_Operations
    
    
    while cont < len(list_symbol):
        symbol = list_symbol[cont]

        with open(f'config/list_{action}.json', 'r', encoding='utf-8') as file:
            list_symbol_hechos = json.load(file)
         
        prev_is = (
            db_query.promedio_correct_percentage(principal_symbol, action, 'is')
            if cont == 0 else
            db_query.promedio_correct_percentage(
                f'crossing_{principal_symbol}_dbs/{list_symbol_hechos["list"][-1]}', action, 'is')
        )
        
        prev_os = (
            db_query.promedio_correct_percentage(principal_symbol, action, 'os')
            if cont == 0 else
            db_query.promedio_correct_percentage(
                f"crossing_{principal_symbol}_dbs/{list_symbol_hechos['list'][-1]}", action, 'os'
            )
        )
        
        print(prev_os, prev_is, '---------')
        if prosedio:
            cont_symbol = len(list_symbol) - cont
            total_dismin = calcular_descuento(
                NumMaxOperations,
                min_operaciones,
                cont_symbol
            )

            print(
                f"Se descuenta {total_dismin:.4f} "
                f" {NumMaxOperations} "
                f"por tener {cont_symbol} símbolos restantes"
            )

            NumMaxOperations -= NumMaxOperations * total_dismin

        create_trees(symbol, action, NumMaxOperations, cont, list_symbol, prev_os, prev_is)
        cont += 1
        prosedio = True

        with open(f'config/list_{action}.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        list_sym = data["list"]
        list_sym.append(symbol)

        with open(f'config/list_{action}.json', 'w', encoding='utf-8') as f:
            json.dump({"list": list_sym},f,indent=4)
            
        print("NumMaxOperations actual:", NumMaxOperations)


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    inicio = time.time()

    with open('config/state.json', 'w', encoding='utf-8') as f:
        json.dump({"state": "running"}, f, indent=4)

    with open('config/list_UP.json', 'w', encoding='utf-8') as f:
        json.dump({"list": []}, f, indent=4)

    with open('config/list_DOWN.json', 'w', encoding='utf-8') as f:
        json.dump({"list": []}, f, indent=4)

    peticiones.initialize_mt5()
    time.sleep(3)
    
    #extract_indicadores()
    
    freeze_support()

    manager = Manager()
    MIC_STATUS = manager.dict(
        {mic: True for mic in MIC_DEVICES}
    )

    p1 = Process(
        target=execute_crossing_builder,
        args=('UP', MIC_STATUS)
    )

    p2 = Process(
        target=execute_crossing_builder,
        args=('DOWN', MIC_STATUS)
    )

    p1.start()
    time.sleep(5)
    p2.start()

    p1.join()
    p2.join()

    with open('config/state.json', 'w', encoding='utf-8') as f:
        json.dump({"state": "stopped"}, f, indent=4)

    print("Ambos procesos han terminado.")
    print(
        f"Tiempo total de ejecución: "
        f"{time.time() - inicio:.4f} segundos"
    )
