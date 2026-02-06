import os
import random
import json
from datetime import time
from concurrent.futures import ProcessPoolExecutor
import operator

import pandas as pd
from src.utils.create_indicators import create_files
from src.db.create_db import create_db
from src.db import query as db_query
from src.routes import peticiones
from src.utils.crossing_funtion.constructor_node import NodeGenerator


with open('config/config_node/config_node.json', encoding='utf-8') as f:
    backtest_config = json.load(f)
    
operadores = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne
}




def cumple_condiciones(df, condiciones):
    """
    Evalúa condiciones sobre un DataFrame COMPLETO de forma vectorizada.
    Retorna máscara booleana para todas las filas.
    """
    if not condiciones:
        return pd.Series(True, index=df.index)
    
    mascara_final = pd.Series(True, index=df.index)
    
    for col, op, valor in condiciones:
        if col not in df.columns:
            # Si la columna no existe, todas las filas son False
            return pd.Series(False, index=df.index)
        
        # Aplicar operación vectorizada
        if op == ">":
            mascara = df[col] > valor
        elif op == "<":
            mascara = df[col] < valor
        elif op == ">=":
            mascara = df[col] >= valor
        elif op == "<=":
            mascara = df[col] <= valor
        elif op == "==":
            mascara = df[col] == valor
        elif op == "!=":
            mascara = df[col] != valor
        else:
            continue
        
        # Combinar con AND lógico
        mascara_final = mascara_final & mascara
    
    return mascara_final


def convertir_a_pips(valores, simbolo):
    
    if 'JPY' in simbolo.upper():
        multiplicador = 100  # 1 pip = 0.01
    else:
        multiplicador = 10_000  # Por defecto, 1 pip = 0.0001

    return [round(float(v) * multiplicador, 2) for v in valores]


def selecte_nodes(file: str, op_down, op_up, symbol, list_nodos):
    """
    Versión optimizada con NumPy/Pandas vectorizado
    """
    ext = file.split('_')[0]
    symbolo = file.split('_')[1]
    
    # Cargar DataFrames (una vez)
    list_dire_indicators_os = os.listdir('output/extrac_os')
    dire = next((f for f in list_dire_indicators_os if ext in f), None)
    
    if not dire:
        return
    
    df_indicators_os = pd.read_csv(f'output/extrac_os/{dire}')
    df_indicators_is = pd.read_csv(f'output/extrac/{file}')
    df_os = pd.read_csv('output/is_os/os.csv')
    df_is = pd.read_csv('output/is_os/is.csv')
    
    # Convertir columnas time a datetime (una vez)
    for df in [df_os, df_is, df_indicators_os, df_indicators_is]:
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
    
    def normalizar_conditions(conditions):
        return json.dumps(conditions, sort_keys=True)
    
    # PRE-CALCULAR todo lo constante
    total_filas_os = len(df_indicators_os)
    total_filas_is = len(df_indicators_is)
    
    # Cache para máscaras
    cache_mascaras = {}
    
    for nodo in list_nodos:
        # Validaciones rápidas primero
        if op_down >= backtest_config['NumMaxOperations'] and nodo['label'] == 'DOWN':
            continue
        if op_up >= backtest_config['NumMaxOperations'] and nodo['label'] == 'UP':
            continue
        
        conditions = nodo['conditions']
        cond_key = normalizar_conditions(conditions)
        
        # Obtener máscara del cache o calcular
        if cond_key not in cache_mascaras:
            mascara_os = cumple_condiciones(df_indicators_os, conditions)
            mascara_is = cumple_condiciones(df_indicators_is, conditions)
            cache_mascaras[cond_key] = (mascara_os, mascara_is)
        else:
            mascara_os, mascara_is = cache_mascaras[cond_key]
        
        # Aplicar máscaras
        df_filtrado_os = df_indicators_os[mascara_os]
        df_filtrado_is = df_indicators_is[mascara_is]
        
        if df_filtrado_os.empty or df_filtrado_is.empty:
            continue
        
        # PROCESAMIENTO OS (optimizado)
        # Merge con df_os
        merged_os = pd.merge(
            df_filtrado_os[['time']],
            df_os[['time', 'open', 'close', 'spread']],
            on='time',
            how='left'
        )
        
        # Filtrar hora != 00:00
        merged_os = merged_os[merged_os['time'].dt.time != time(0, 0)]
        
        if merged_os.empty:
            continue
        
        # Calcular beneficios OS
        if nodo['label'] == 'UP':
            beneficios_brutos_os = merged_os['close'] - merged_os['open']
        else:  # 'DOWN'
            beneficios_brutos_os = merged_os['open'] - merged_os['close']
        
        # Convertir a pips
        if 'JPY' in symbolo.upper():
            multiplicador = 100
        else:
            multiplicador = 10000
        
        beneficios_pips_os = beneficios_brutos_os * multiplicador
        spreads_pips_os = merged_os['spread'] / 10
        beneficios_netos_os = beneficios_pips_os - spreads_pips_os
        
        # Estadísticas OS
        aciertos_os = (beneficios_netos_os > 0).sum()
        total_os = len(beneficios_netos_os)
        
        if (aciertos_os < backtest_config['MinOperationsOS'] or 
            total_os == 0):
            continue
        
        porcentaje_aciertos_os = aciertos_os / total_os
        if (porcentaje_aciertos_os < backtest_config['MinSuccessRate'] or 
            porcentaje_aciertos_os > backtest_config['MaxSuccessRate']):
            continue
        
        progressive_os = total_os / total_filas_os
        
        # PROCESAMIENTO IS (similar a OS)
        merged_is = pd.merge(
            df_filtrado_is[['time']],
            df_is[['time', 'open', 'close', 'spread']],
            on='time',
            how='left'
        )
        merged_is = merged_is[merged_is['time'].dt.time != time(0, 0)]
        
        if merged_is.empty:
            continue
        
        if nodo['label'] == 'UP':
            beneficios_brutos_is = merged_is['close'] - merged_is['open']
        else:
            beneficios_brutos_is = merged_is['open'] - merged_is['close']
        
        beneficios_pips_is = beneficios_brutos_is * multiplicador
        spreads_pips_is = merged_is['spread'] / 10
        beneficios_netos_is = beneficios_pips_is - spreads_pips_is
        
        aciertos_is = (beneficios_netos_is > 0).sum()
        total_is = len(beneficios_netos_is)
        
        if (aciertos_is < backtest_config['MinOperationsIS'] or 
            total_is == 0):
            continue
        
        porcentaje_aciertos_is = aciertos_is / total_is
        if (porcentaje_aciertos_is < backtest_config['MinSuccessRate'] or 
            porcentaje_aciertos_is > backtest_config['MaxSuccessRate']):
            continue
        
        progressive_is = total_is / total_filas_is
        progressiveVariation = abs(progressive_os - progressive_is)
        
        if progressiveVariation > backtest_config['ProgressiveVariation']:
            continue
        
        # Consulta DB (esta parte ya es rápida)
        list_dates_is = merged_is['time'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
       
        
        nodo_mas_parecido = db_query.nodo_con_mas_fechas_hora_comunes(symbol, list_dates_is)
        if nodo_mas_parecido:
            coincidencias = nodo_mas_parecido['coincidencias']
            total_operaciones = nodo_mas_parecido['total_operations']
            porciento_nodo_db = coincidencias/total_operaciones
            porciento_is = coincidencias/total_is
            porciento = (porciento_nodo_db + porciento_is)/2
            if porciento >=backtest_config['SimilarityMax'] and nodo_mas_parecido['total_operations'] < total_is:
                db_query.eliminar_nodo_y_registros(symbol, nodo_mas_parecido['node_id']) 
            elif porciento >=backtest_config['SimilarityMax'] and nodo_mas_parecido['total_operations'] >= total_is: 
                print('Mayor pero el de la db mejor')
                continue
        
        # Insertar en DB
        db_query.insertar_nodo_con_registros(
            name=symbol,
            label=nodo['label'],
            file_in_db=file,
            conditions=cond_key,
            correct_percentage=float(porcentaje_aciertos_is),
            successful_operations=int(aciertos_is),
            total_operations=int(total_is),
            correct_percentage_os=float(porcentaje_aciertos_os),
            successful_operations_os=int(aciertos_os),
            total_operations_os=int(total_os),
            fechas=list_dates_is,
            veneficios=beneficios_netos_is.round(2).tolist(),
            fechas_os=merged_os['time'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            veneficios_os=beneficios_netos_os.round(2).tolist()
        )
        
   
        
def procesar_archivo(file:str, symbol):
    df = pd.read_csv(f"output/extrac/{file}")
    node_generator = NodeGenerator(df)
    operaciones_exitosas_UP = 0
    operaciones_exitosas_DOWN = 0
    while operaciones_exitosas_UP < backtest_config['NumMaxOperations'] or operaciones_exitosas_DOWN < backtest_config['NumMaxOperations']:
        list_nodos = node_generator.generar_nodos(1000)
        selecte_nodes(file, operaciones_exitosas_DOWN, operaciones_exitosas_UP, symbol, list_nodos)
        operaciones_exitosas_UP = db_query.successful_operations_by_label(symbol, 'UP')
        operaciones_exitosas_DOWN = db_query.successful_operations_by_label(symbol,'DOWN')
                    

  


def create_trees(symbol, timeframe):  
    create_db(symbol)
    peticiones.initialize_mt5()
    list_files = os.listdir('output/extrac')
    
    indicators_files = [file.split('_')[0]+'.csv'  for file in list_files]
    create_files(symbol, timeframe, backtest_config['dateStart'], backtest_config['dateEnd'], indicators_files, 'extrac_os')
    
    MAX_PROCESOS = 25  # Puedes ajustar este número según tu CPU
    futures = []
    
    with ProcessPoolExecutor(max_workers=MAX_PROCESOS) as executor:
        for file in list_files:
            future = executor.submit(procesar_archivo, file, symbol)
            futures.append(future)

    db_query.insertar_nodo_con_registros(
        name=symbol,
        label='END',
        file_in_db= '001maco',
        conditions= 'APO > 2',
        correct_percentage= 0.53,
        successful_operations= 251,
        total_operations= 491,
        correct_percentage_os= 0.53,
        successful_operations_os= 251,
        total_operations_os= 491,
        fechas=[''],
        veneficios=[0],
        fechas_os=[''],
        veneficios_os=[0]
    )
    

    
    
