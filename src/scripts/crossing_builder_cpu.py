import json
import os
import sys
import logging
from datetime import time
import time as tim
import operator
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Process
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


import pandas as pd 
import numpy as np

from src.routes import peticiones
from src.db import query as db_query
from src.utils.indicadores_for_crossing import extract_indicadores
from src.utils.constructor_node import NodeGenerator
from src.utils.extrat_data_for_crossing import extract_data_crossing, select_symbols_correl
from src.utils.common_functions import filtro_mercado, hora_en_mercado




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


_DF_CACHE = {}


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s - %(message)s',
        force=True
    )



def load_csv_cached(path):
    if path not in _DF_CACHE:
        _DF_CACHE[path] = pd.read_csv(
            path,
            parse_dates=["time"],
            memory_map=True
        )
    return _DF_CACHE[path]


def _max_decimals(series, sample_size=1000):
    s = series.dropna()
    if s.empty:
        return 0
    if len(s) > sample_size:
        s = s.sample(sample_size, random_state=0)
    max_dec = 0
    for v in s:
        text = f"{v:.10f}".rstrip('0').rstrip('.')
        if '.' in text:
            dec = len(text.split('.')[1])
            if dec > max_dec:
                max_dec = dec
    return max_dec


def _pip_sizes(series, symbol):
    digits = _max_decimals(series)
    pip_size = 0.01 if 'JPY' in symbol.upper() else 0.0001
    point_size = (10 ** (-digits)) if digits > 0 else (pip_size / 10)
    return pip_size, point_size


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
        node_generator,  # generar nodos frescos para cada archivo
        prev_os, 
        prev_is,
        porcent_aumento_os,
        porcent_aumento_is,
        config,
        mercado
    ):
    
    por_direccion = config['general']['por_direccion']
    list_symbols_inversos = config['symbol']['list_symbol_inversos']
    principal_symbol = config['principal_symbol']
    list_symbol = config['symbol']['list_symbol']
    ini = tim.time()
    cant_nodos = 1000 
    list_nodos = node_generator.generar_nodos(cant_nodos)
    print(f"{cant_nodos} Nodos generados en {tim.time() - ini:.4f} segundos")
    
    # pip_size/point_size se calculan tras cargar precios
    if por_direccion:
        if symbol in list_symbols_inversos:
            list_nodos = [nodo for nodo in list_nodos if nodo['label'] != action]
        else:
            list_nodos = [nodo for nodo in list_nodos if nodo['label'] == action]

    ext = file.split('_')[0]
    dire = next(
        f for f in os.listdir(
            f'output/{principal_symbol}/crossing/{symbol}/extrac_os'
        )
        if ext in f
    )
    
    
    df_indicators_os = pd.read_parquet(
        f'output/{principal_symbol}/crossing/{symbol}/extrac_os/{dire}'
    )
   
    is_path = f'output/{principal_symbol}/crossing/{symbol}/extrac/{file}'
    df_indicators_is = pd.read_parquet(is_path)
   
    # Convertir time manualmente
    if "time" in df_indicators_os.columns:
        df_indicators_os["time"] = pd.to_datetime(df_indicators_os["time"])

    if "time" in df_indicators_is.columns:
        df_indicators_is["time"] = pd.to_datetime(df_indicators_is["time"])
        
    df_os = load_csv_cached(f'output/{principal_symbol}/is_os/os.csv')
    df_is = load_csv_cached(f'output/{principal_symbol}/is_os/is.csv')
    pip_size, point_size = _pip_sizes(df_os['open'], symbol)
    
    os_time_np = df_os['time'].to_numpy()
    is_time_np = df_is['time'].to_numpy()
    
    
    def normalizar_conditions(conditions):
        return json.dumps(conditions, sort_keys=True)
    
    
    if cont == 0:
        list_oper_os =list(db_query.get_dates_by_label(principal_symbol, principal_symbol, mercado, action, 'os'))
        list_oper_is = list(db_query.get_dates_by_label(principal_symbol, principal_symbol, mercado, action, 'is'))
    else: 
        list_oper_os = list(db_query.get_dates_by_label(principal_symbol, list_symbol[cont-1], mercado, action, 'os'))
        list_oper_is = list(db_query.get_dates_by_label(principal_symbol, list_symbol[cont-1], mercado, action, 'is'))
    list_oper_is = list(set(list_oper_is))
    list_oper_os = list(set(list_oper_os))
    fechas_dt_os = pd.to_datetime(list_oper_os)
    fechas_dt_is = pd.to_datetime(list_oper_is)
 
    df_indicators_os_index = df_indicators_os.index[df_indicators_os['time'].isin(fechas_dt_os)]
    df_indicators_is_index = df_indicators_is.index[df_indicators_is['time'].isin(fechas_dt_is)]
    
    df_indicators_os_anteriores = df_indicators_os_index[df_indicators_os_index > 0] - 1
    df_indicators_is_anteriores = df_indicators_is_index[df_indicators_is_index > 0] - 1
    

    df_indicators_os_fil = df_indicators_os.iloc[df_indicators_os_anteriores]
    df_indicators_is_fil = df_indicators_is.iloc[df_indicators_is_anteriores]
    
    df_indicators_os_fil = filtro_mercado(df_indicators_os_fil, mercado)
    df_indicators_is_fil = filtro_mercado(df_indicators_is_fil, mercado)

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

            # ---------------------------------
            # VALIDAR MERCADO EN VELA ENTRADA
            # ---------------------------------
            hour_next = os_hour[idx]

            if not hora_en_mercado(hour_next, mercado):
                continue

            if hour_next == 0:
                continue

            date = pd.to_datetime(os_time[idx]).strftime("%Y-%m-%d %H:%M:%S")
            list_dates_os.append(date)

            if action == 'UP':
                beneficio = os_close[idx] - os_open[idx]
                list_beneficio_os_buto.append(beneficio)
            elif action == 'DOWN':
                beneficio = os_open[idx] - os_close[idx]
                list_beneficio_os_buto.append(beneficio)

            spread_pips = os_spread[idx] * point_size / pip_size
            
            list_spread_os.append(spread_pips)
            
        
        if list_beneficio_os_buto:
            bruto = np.asarray(list_beneficio_os_buto, dtype=np.float64) / pip_size
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

            # ---------------------------------
            # VALIDAR MERCADO EN VELA ENTRADA
            # ---------------------------------
            hour_next = is_hour[idx]

            if not hora_en_mercado(hour_next, mercado):
                continue

            if hour_next == 0:
                continue
            date = pd.to_datetime(is_time[idx]).strftime("%Y-%m-%d %H:%M:%S")
            list_dates_is.append(date)  
            if action == 'UP':
                beneficio = is_close[idx] - is_open[idx]
                list_beneficio_is_buto.append(beneficio)
            elif action == 'DOWN':
                beneficio = is_open[idx] - is_close[idx]
                list_beneficio_is_buto.append(beneficio)
            spread_pips = is_spread[idx] * point_size / pip_size
            
            list_spread_is.append(spread_pips)
            
        if list_beneficio_is_buto:
            bruto = np.asarray(list_beneficio_is_buto, dtype=np.float64) / pip_size
            spread = np.asarray(list_spread_is, dtype=np.float64)
            list_beneficio_is = (bruto - spread).tolist()
        else:
            list_beneficio_is = []
            
        aciertos_is = sum(1 for r in list_beneficio_is if r > 0)
        total_is = len(list_beneficio_is)
        if total_is ==0:
            continue
        porcentaje_aciertos_is = (aciertos_is / total_is)
       
        if not (prev_is + (porcent_aumento_is - 0.015) <= porcentaje_aciertos_is <= prev_is + (porcent_aumento_is + 0.015)):
            continue
        
        progressive_is = total_is/len(df_indicators_is) 
        progressiveVariation = abs(progressive_os - progressive_is)
        if progressiveVariation > config['general']['ProgressiveVariation']:
            continue
       
        nodo_mas_parecido = db_query.nodo_con_mas_fechas_hora_comunes(principal_symbol, symbol, mercado, list_dates_is)
        if nodo_mas_parecido:
            coincidencias = nodo_mas_parecido['coincidencias']
            total_operaciones = nodo_mas_parecido['total_operations']
            porciento_nodo_db = coincidencias/total_operaciones
            porciento_is = coincidencias/total_is
            porciento = (porciento_nodo_db + porciento_is)/2
            if porciento >=config['general']['SimilarityMax'] and nodo_mas_parecido['total_operations'] < total_is:
                db_query.eliminar_nodo_y_registros(nodo_mas_parecido['node_id']) 
            elif porciento >=config['general']['SimilarityMax'] and nodo_mas_parecido['total_operations'] >= total_is: 
                print('Mayor pero el de la db mejor')
                continue
               
    
        db_query.insertar_nodo_con_registros(
            principal_symbol=principal_symbol,
            symbol_cruce=symbol,
            label=action,
            mercado=mercado,
            file_in_db=file,
            conditions=normalizar_conditions(nodo['conditions']),
            correct_percentage=porcentaje_aciertos_is,
            successful_operations=aciertos_is,
            total_operations=total_is,
            correct_percentage_os=porcentaje_os,
            successful_operations_os=aciertos,
            total_operations_os=total,
            fechas=list_dates_is,
            veneficios=list_beneficio_is,
            fechas_os=list_dates_os,
            veneficios_os=list_beneficio_os,    
        )
        print(f"Nodo insertado: symbol={symbol} action={action} mercado={mercado} porcentaje_is={porcentaje_aciertos_is:.4f} porcentaje_os={porcentaje_os:.4f} total_is={total_is} total_os={total}")


def procesar_archivo(file: str, symbol, action, cont, prev_os, prev_is, porcent_aumento_os, porcent_aumento_is, NumMaxOperations, config, mercado):
    principal_symbol = config['principal_symbol']
    df = pd.read_parquet(f'output/{principal_symbol}/crossing/{symbol}/extrac/{file}')
    node_generator = NodeGenerator(df)
    operaciones_exitosas = 0
    while (operaciones_exitosas < NumMaxOperations):
        
        selecte_nodes(
            file,
            symbol,
            action,
            cont,
            node_generator,  # generar nodos frescos para cada archivo
            prev_os, 
            prev_is,
            porcent_aumento_os,
            porcent_aumento_is,
            config,
            mercado
        )
        operaciones_exitosas = (
            db_query.successful_operations_by_label(
                principal_symbol=principal_symbol,
                symbol_cruce=symbol,
                label=action,
                mercado=mercado
            
            )
        )

        print(
            f"Operaciones exitosas mercado {mercado} "
            f"{symbol}-{action}: "
            f"{operaciones_exitosas}"
        )
        
   
def init_worker():
    import weka.core.jvm as jvm
    if not jvm.started:
        jvm.start(packages=True)


def calcular_porcentage(symbol, prev, config):
    sumatoria = 0
    dict_symbol_correl = config["symbol"]["dict_symbol_correl"]
    for key, value in dict_symbol_correl.items():
        sumatoria += value
    corre = dict_symbol_correl[symbol]
    return corre/sumatoria * (1-prev)


def create_trees(symbol, action, cont, prev_os, prev_is, NumMaxOperations, mercado, config):
    principal_symbol = config['principal_symbol']

    list_files = os.listdir(
        f'output/{principal_symbol}/crossing/{symbol}/extrac'
    )
    
    porcent_aumento_os =calcular_porcentage(symbol, prev_os, config)
    porcent_aumento_is =calcular_porcentage(symbol, prev_is, config)
    
    MAX_PROCESOS = len(list_files) # ajustable según CPU
    if MAX_PROCESOS > 25:
        MAX_PROCESOS = 25
    
    with ProcessPoolExecutor(
        max_workers=MAX_PROCESOS,
    ) as executor:

        futures = []
        for file in list_files:
            futures.append(
                executor.submit(
                    procesar_archivo,
                    file,
                    symbol,
                    action,
                    cont,
                    prev_os,
                    prev_is,
                    porcent_aumento_os,
                    porcent_aumento_is,
                    NumMaxOperations,
                    config,
                    mercado
                )
            )
            
        for future in as_completed(futures):
            try:
                future.result()
            except Exception:
                logger.exception("Error en proceso hijo de create_trees")

    
def _execute_crossing_builder(action, config):
     
    for mercado in config["list_mercado"]:
        cont = 0
        prosedio = True
        NumMaxOperations = config["general"]['NumMaxOperations']
        list_symbol = config["symbol"]["list_symbol"]
        principal_symbol = config["principal_symbol"]
        min_operaciones = config["general"]['min_operaciones']
        while cont < len(list_symbol):
            symbol = list_symbol[cont]
            
            last_symbol = (
                principal_symbol
                if cont == 0 
                else list_symbol[cont - 1]
            )

            prev_is = db_query.promedio_correct_percentage(
                principal_symbol, last_symbol, mercado, action, 'is'
            )

            prev_os = db_query.promedio_correct_percentage(
                principal_symbol, last_symbol, mercado, action, 'os'
            )

            if prev_is is None:
                prev_is = 0.0
            if prev_os is None:
                prev_os = 0.0
            
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

            create_trees(symbol, action, cont, prev_os, prev_is,NumMaxOperations, mercado, config)
            cont += 1
            prosedio = True    
            print("NumMaxOperations actual:", NumMaxOperations)
         

def execute_crossing_builder(principal_symbol, list_mercado):
    inicio =tim.time()           
    peticiones.initialize_mt5()
    tim.sleep(3)
    
    extract_data_crossing(principal_symbol)
    select_symbols_correl(principal_symbol)
    extract_indicadores(principal_symbol)
    
    with open('config/general_config.json', 'r', encoding='utf-8') as f:
        general_config = json.load(f)
    with open(f'config/divisas/{principal_symbol}/config_{principal_symbol}.json', 'r', encoding='utf-8') as f:
        config_symbol = json.load(f)
        
    for symbol in config_symbol['list_symbol']:
        for mercado in list_mercado:
            db_query.eliminar_nodos_y_registros(principal_symbol, symbol, mercado)
            
    config = {
        "general": general_config,
        "symbol": config_symbol,
        "principal_symbol": principal_symbol,
        "list_mercado": list_mercado
    }
     # Crear y ejecutar procesos para 'UP' y 'DOWN'
    
    p1 = Process(
    target=_execute_crossing_builder,
    args=('UP', config)
)

    p2 = Process(
        target=_execute_crossing_builder,
        args=('DOWN', config)
    )
    p1.start()
    tim.sleep(5)  # Esperar un poco antes de iniciar el segundo proceso
    p2.start()

    # Esperar a que terminen
    p1.join()
    p2.join()
    print("Ambos procesos han terminado.")
    
    print(f"Tiempo de creación: {tim.time() - inicio:.4f} segundos")
 
if __name__ == "__main__":
    execute_crossing_builder('EURCAD', ['Asia', 'Europa', 'America'])