import json
import os
import sys
from datetime import time
import time as tim
import operator
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


import pandas as pd 
import numpy as np

from src.routes import peticiones
from src.db.create_db import create_db
from src.db import query as db_query
from src.utils.crossing_funtion.crear_indicadores_in_crossing import extract_indicadores
from src.utils.crossing_funtion.constructor_node import NodeGenerator



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
    '!=': operator.ne,
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

_DF_CACHE = {}

def load_csv_cached(path):
    if path not in _DF_CACHE:
        _DF_CACHE[path] = pd.read_csv(
            path,
            parse_dates=["time"],
            memory_map=True
        )
    return _DF_CACHE[path]


def calcular_descuento(N0, Nf, k):
    """
    Calcula el porcentaje de descuento por iteración.

    N0 : operaciones iniciales
    Nf : operaciones finales
    k  : número de iteraciones
    """
    if N0 <= 0 or Nf <= 0 or k <= 0:
        raise ValueError("N0, Nf y k deben ser mayores que cero")

    p = 1 - (Nf / N0) ** (1 / k)   # en decimal
    return p 


def preparar_condiciones(conditions, col_index):
    """
    Precompila condiciones para evaluación rápida
    """
    compiled = []
    for col, op, value in conditions:
        compiled.append((
            col_index[col],
            OP_MAP[op],
            value
        ))
    return compiled


def evaluar_condiciones_vectorizado(matrix, compiled_conditions):
    if matrix.shape[0] == 0:
        return np.array([], dtype=bool)

    mask = np.ones(matrix.shape[0], dtype=bool)

    for idx, op_func, value in compiled_conditions:
        mask &= op_func(matrix[:, idx], value)
        if not mask.any():
            break

    return mask


#nueva funcion
def dataframe_to_matrix(df, condition_cols):
    """
    Convierte DataFrame a matriz numérica ordenada por columnas usadas en condiciones
    """
    data = df[condition_cols].to_numpy(dtype=np.float64)
    return data


def selecte_nodes(
        file,
        symbol,
        action,
        cont,
        list_symbol,
        node_generator,  # generar nodos frescos para cada archivo
        prev_os, 
        prev_is,
        porcent_aumento_os,
        porcent_aumento_is
    ):
    
    ini = tim.time()
    cant_nodos = 1000 * (cont+1)
    list_nodos = node_generator.generar_nodos(cant_nodos)
    print(f"{cant_nodos} Nodos generados en {tim.time() - ini:.4f} segundos")
    
    pip_mult = 100 if 'JPY' in symbol.upper() else 10_000
    if por_direccion:
        if symbol in list_symbols_inversos:
            list_nodos = [nodo for nodo in list_nodos if nodo['label'] != action]
        else:
            list_nodos = [nodo for nodo in list_nodos if nodo['label'] == action]

    ext = file.split('_')[0]
    dire = next(
        f for f in os.listdir(
            f'output/crossing_{principal_symbol}/{symbol}/extrac_os'
        )
        if ext in f
    )
    
    
    df_indicators_os = pd.read_parquet(
        f'output/crossing_{principal_symbol}/{symbol}/extrac_os/{dire}'
    )

    df_indicators_is = pd.read_parquet(
        f'output/crossing_{principal_symbol}/{symbol}/extrac/{file}'
    )

    # Convertir time manualmente
    if "time" in df_indicators_os.columns:
        df_indicators_os["time"] = pd.to_datetime(df_indicators_os["time"])

    if "time" in df_indicators_is.columns:
        df_indicators_is["time"] = pd.to_datetime(df_indicators_is["time"])
        
    df_os = load_csv_cached('output/is_os/os.csv')
    df_is = load_csv_cached('output/is_os/is.csv')
    
    os_time_np = df_os['time'].to_numpy()
    is_time_np = df_is['time'].to_numpy()
    
    
    def normalizar_conditions(conditions):
        return json.dumps(conditions, sort_keys=True)
    
    
    if cont == 0:
        list_oper_os =list(db_query.get_dates_by_label(principal_symbol, action, 'os'))
        list_oper_is = list(db_query.get_dates_by_label(principal_symbol, action, 'is'))
    else: 
        list_oper_os = list(db_query.get_dates_by_label(f'crossing_{principal_symbol}_dbs/{list_symbol[cont-1]}', action, 'os'))
        list_oper_is = list(db_query.get_dates_by_label(f'crossing_{principal_symbol}_dbs/{list_symbol[cont-1]}', action, 'is'))
    fechas_dt_os = pd.to_datetime(list_oper_os)
    fechas_dt_is = pd.to_datetime(list_oper_is)
    
    
    df_indicators_os_index = df_indicators_os.index[df_indicators_os['time'].isin(fechas_dt_os)]
    df_indicators_is_index = df_indicators_is.index[df_indicators_is['time'].isin(fechas_dt_is)]

    df_indicators_os_anteriores = df_indicators_os_index[df_indicators_os_index > 0] - 1
    df_indicators_is_anteriores = df_indicators_is_index[df_indicators_is_index > 0] - 1
    

    df_indicators_os_fil = df_indicators_os.iloc[df_indicators_os_anteriores]
    df_indicators_is_fil = df_indicators_is.iloc[df_indicators_is_anteriores]
    
        # Arrays OS
    os_open   = df_os['open'].to_numpy()
    os_close  = df_os['close'].to_numpy()
    os_spread = df_os['spread'].to_numpy()
    os_time   = df_os['time'].to_numpy()
    os_hour = df_os['time'].dt.hour.to_numpy()
    ind_os_time = df_indicators_os_fil['time'].to_numpy()

    # Arrays IS
    is_open   = df_is['open'].to_numpy()
    is_close  = df_is['close'].to_numpy()
    is_spread = df_is['spread'].to_numpy()
    is_time   = df_is['time'].to_numpy()
    is_hour = df_is['time'].dt.hour.to_numpy()
    ind_is_time = df_indicators_is_fil['time'].to_numpy()
    
    # nueva parte
    cond_cols = []
    seen = set()
    for nodo in list_nodos:
        for col, _, _ in nodo['conditions']:
            if col not in seen:
                seen.add(col)
                cond_cols.append(col)

    
    matrix_os = dataframe_to_matrix(df_indicators_os_fil, cond_cols)
    if matrix_os.shape[1] != len(cond_cols):
        raise RuntimeError("Desalineación matrix_os / cond_cols")
    matrix_is = dataframe_to_matrix(df_indicators_is_fil, cond_cols)

    col_index = {col:i for i,col in enumerate(cond_cols)}
    
    #
    
    for nodo in list_nodos:
        conditions = nodo['conditions']
        list_beneficio_os = []
        list_beneficio_os_buto = []
        list_dates_os = []
        list_spread_os = []
        
        compiled = preparar_condiciones(conditions, col_index)
        mask_os = evaluar_condiciones_vectorizado(matrix_os, compiled)
        
        idx_os = np.where(mask_os)[0]
        for i in idx_os:
            
            date = ind_os_time[i]
            posicion = np.searchsorted(os_time_np, date)

            if posicion >= len(os_time_np) or os_time_np[posicion] != date:
                continue


            if posicion + 1 >= len(df_os):
                continue


            idx = posicion + 1

            if os_hour[idx] == 0:
                continue

            date = pd.to_datetime(os_time[idx]).strftime("%Y-%m-%d %H:%M:%S")
            list_dates_os.append(date)

            if action == 'UP':
                beneficio = os_close[idx] - os_open[idx]
                list_beneficio_os_buto.append(beneficio)
            elif action == 'DOWN':
                beneficio = os_open[idx] - os_close[idx]
                list_beneficio_os_buto.append(beneficio)

            spread_pips = os_spread[idx] / 10
            
            list_spread_os.append(spread_pips)
            
        
        if list_beneficio_os_buto:
            bruto = np.asarray(list_beneficio_os_buto, dtype=np.float64) * pip_mult
            spread = np.asarray(list_spread_os, dtype=np.float64)
            list_beneficio_os = (bruto - spread).tolist()
        else:
            list_beneficio_os = [] 
                 
        aciertos = sum(1 for r in list_beneficio_os if r > 0)
        total = len(list_beneficio_os)
        if total == 0:
            continue
        porcentaje_os = (aciertos / total)
        
        if not (prev_os + (porcent_aumento_os - 0.02) <= porcentaje_os <= prev_os + (porcent_aumento_os + 0.02)):
            continue
            
        progressive_os = total/len(df_indicators_os)
        
        #--------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------
        list_spread_is = []
        list_beneficio_is_buto = []
        list_dates_is = []
        list_beneficio_is = []
        
        mask_is = evaluar_condiciones_vectorizado(matrix_is, compiled)
        idx_is = np.where(mask_is)[0]
        for i in idx_is:
            
            date = ind_is_time[i]
            posicion = np.searchsorted(is_time_np, date)

            if posicion >= len(is_time_np) or is_time_np[posicion] != date:
                continue


            if posicion + 1 >= len(df_is):
                continue

            idx = posicion + 1
            if is_hour[idx] == 0:
                continue
            date = pd.to_datetime(is_time[idx]).strftime("%Y-%m-%d %H:%M:%S")
            list_dates_is.append(date)  
            if action == 'UP':
                beneficio = is_close[idx] - is_open[idx]
                list_beneficio_is_buto.append(beneficio)
            elif action == 'DOWN':
                beneficio = is_open[idx] - is_close[idx]
                list_beneficio_is_buto.append(beneficio)
            spread_pips = is_spread[idx] / 10
            
            list_spread_is.append(spread_pips)
            
        if list_beneficio_is_buto:
            bruto = np.asarray(list_beneficio_is_buto, dtype=np.float64) * pip_mult
            spread = np.asarray(list_spread_is, dtype=np.float64)
            list_beneficio_is = (bruto - spread).tolist()
        else:
            list_beneficio_is = []
            
        aciertos_is = sum(1 for r in list_beneficio_is if r > 0)
        total_is = len(list_beneficio_is)
        if total_is ==0:
            continue
        porcentaje_aciertos_is = (aciertos_is / total_is)
        
        if not (prev_is + (porcent_aumento_is - 0.01) <= porcentaje_aciertos_is <= prev_is + (porcent_aumento_is + 0.01)):
            continue
        
        progressive_is = total_is/len(df_indicators_is) 
        progressiveVariation = abs(progressive_os - progressive_is)
        if progressiveVariation > config_node['ProgressiveVariation']:
            continue
       
        nodo_mas_parecido = db_query.nodo_con_mas_fechas_hora_comunes(f'crossing_{principal_symbol}_dbs/{symbol}', list_dates_is)
        if nodo_mas_parecido:
            coincidencias = nodo_mas_parecido['coincidencias']
            total_operaciones = nodo_mas_parecido['total_operations']
            porciento_nodo_db = coincidencias/total_operaciones
            porciento_is = coincidencias/total_is
            porciento = (porciento_nodo_db + porciento_is)/2
            if porciento >=config_node['SimilarityMax'] and nodo_mas_parecido['total_operations'] < total_is:
                db_query.eliminar_nodo_y_registros(f'crossing_{principal_symbol}_dbs/{symbol}', nodo_mas_parecido['node_id']) 
            elif porciento >=config_node['SimilarityMax'] and nodo_mas_parecido['total_operations'] >= total_is: 
                print('Mayor pero el de la db mejor')
                continue
               
        
        db_query.insertar_nodo_con_registros(
            name=f'crossing_{principal_symbol}_dbs/{symbol}',
            label=action,
            file_in_db = file,
            conditions= normalizar_conditions(nodo['conditions']),
            correct_percentage= porcentaje_aciertos_is,
            successful_operations= aciertos_is,
            total_operations= total_is,
            correct_percentage_os= porcentaje_os,
            successful_operations_os= aciertos,
            total_operations_os= total,
            fechas=list_dates_is,
            veneficios=list_beneficio_is,
            fechas_os=list_dates_os,
            veneficios_os=list_beneficio_os
        )
        

def procesar_archivo(file: str, symbol, action, NumMaxOperations, cont, list_symbol, prev_os, prev_is, porcent_aumento_os, porcent_aumento_is):
    
    df = pd.read_parquet(f'output/crossing_{principal_symbol}/{symbol}/extrac/{file}')
    node_generator = NodeGenerator(df)
    operaciones_exitosas = 0
    while (operaciones_exitosas < NumMaxOperations):
        
        selecte_nodes(
            file,
            symbol,
            action,
            cont,
            list_symbol,
            node_generator,  # generar nodos frescos para cada archivo
            prev_os, 
            prev_is,
            porcent_aumento_os,
            porcent_aumento_is
        )
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

   
def init_worker():
    import weka.core.jvm as jvm
    if not jvm.started:
        jvm.start(packages=True)


def calcular_porcentage(symbol, prev):
    sumatoria = 0
    for key, value in dict_symbol_correl.items():
        sumatoria += value
    corre = dict_symbol_correl[symbol]
    return corre/sumatoria * (1-prev)


def create_trees(symbol, action, NumMaxOperations, cont, list_symbol, prev_os, prev_is):
    db_path = f'output/db/crossing_{principal_symbol}_dbs/{symbol}.db'

    if not os.path.exists(db_path):
        create_db(f'crossing_{principal_symbol}_dbs/{symbol}')

    list_files = os.listdir(
        f'output/crossing_{principal_symbol}/{symbol}/extrac'
    )
    
    porcent_aumento_os =calcular_porcentage(symbol, prev_os)
    porcent_aumento_is =calcular_porcentage(symbol, prev_is)
    
    MAX_PROCESOS = len(list_files) # ajustable según CPU
    if MAX_PROCESOS > 25:
        MAX_PROCESOS = 25
    
    with ProcessPoolExecutor(
        max_workers=MAX_PROCESOS,
    ) as executor:

        for file in list_files:
            executor.submit(
                procesar_archivo,
                file,
                symbol,
                action,
                NumMaxOperations,
                cont,
                list_symbol,
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
    


def execute_crossing_builder(action):
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



if __name__ == "__main__":
    inicio =tim.time()
    with open('config/state.json', 'w', encoding='utf-8') as f:
        json.dump({"state": "running"}, f, indent=4)
        
    with open('config/list_UP.json', 'w', encoding='utf-8') as f:
        json.dump({"list": []}, f, indent=4)
    with open('config/list_DOWN.json', 'w', encoding='utf-8') as f:
        json.dump({"list": []}, f, indent=4)
                
    peticiones.initialize_mt5()
    tim.sleep(3)
    
    extract_indicadores()#------------------------------------
    
     # Crear y ejecutar procesos para 'UP' y 'DOWN'
    
    p1 = Process(target=execute_crossing_builder, args=('UP',))
    p2 = Process(target=execute_crossing_builder, args=('DOWN',))

    p1.start()
    tim.sleep(5)  # Esperar un poco antes de iniciar el segundo proceso
    p2.start()

    # Esperar a que terminen
    p1.join()
    p2.join()
    with open('config/state.json', 'w', encoding='utf-8') as f:
        json.dump({"state": "stopped"}, f, indent=4)
    print("Ambos procesos han terminado.")
    
    print(f"Tiempo de creación: {tim.time() - inicio:.4f} segundos")
       
        
  
    
    
