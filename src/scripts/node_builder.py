import os
import json
import logging
from datetime import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import operator

import pandas as pd
from src.db import query as db_query
from src.routes import peticiones
from src.utils.constructor_node import NodeGenerator
from src.utils.common_functions import filtro_mercado, hora_en_mercado


with open('config/general_config.json', encoding='utf-8') as f:
    config = json.load(f)


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s - %(message)s'
    )

    
operadores = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne
}


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
        mascara_final &= mascara
    return mascara_final


def selecte_nodes(file: str, op_down, op_up, symbol, list_nodos, mercado):
    """
    Versión optimizada con NumPy/Pandas vectorizado
    """
    ext = file.split('_')[0]
    symbolo = file.split('_')[1]
    
    # Cargar DataFrames (una vez)
    list_dire_indicators_os = os.listdir(f'output/{symbol}/extrac_os')
    dire = next((f for f in list_dire_indicators_os if ext in f), None)
    
    if not dire:
        return
    
    df_indicators_os = pd.read_parquet(f'output/{symbol}/extrac_os/{dire}')
    df_indicators_is = pd.read_parquet(f'output/{symbol}/extrac/{file}')
    df_os = pd.read_csv(f'output/{symbol}/is_os/os.csv')
    df_is = pd.read_csv(f'output/{symbol}/is_os/is.csv')
    
    # Convertir columnas time a datetime (una vez)
    for df in [df_os, df_is, df_indicators_os, df_indicators_is]:
        if 'time' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['time']):
                df['time'] = pd.to_datetime(df['time'])

            # ⭐ PRECALCULO CRÍTICO
            df['hour'] = df['time'].dt.hour
            
    # ===== FILTRO MERCADO SEGURO =====

    df_os = filtro_mercado(df_os, mercado)
    df_is = filtro_mercado(df_is, mercado)

    # sincronizar indicators con precios
    df_indicators_os = df_indicators_os[
        df_indicators_os['time'].isin(df_os['time'])
    ]

    df_indicators_is = df_indicators_is[
        df_indicators_is['time'].isin(df_is['time'])
    ]
            
    def normalizar_conditions(conditions):
        return json.dumps(conditions, sort_keys=True)
    
    # PRE-CALCULAR todo lo constante
    total_filas_os = len(df_indicators_os)
    total_filas_is = len(df_indicators_is)
    
    # Cache para máscaras
    cache_mascaras = {}
    
    for nodo in list_nodos:
        # Validaciones rápidas primero
        if op_down >= config['NumMaxOperations'] and nodo['label'] == 'DOWN':
            continue
        if op_up >= config['NumMaxOperations'] and nodo['label'] == 'UP':
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
        # ===============================
        # ENTRADA EN VELA SIGUIENTE (+1)
        # ===============================
        merged_os[['open', 'close', 'spread']] = (
            merged_os[['open', 'close', 'spread']].shift(-1)
        )
        merged_os['hour_next'] = merged_os['time'].shift(-1).dt.hour

        merged_os = merged_os[
            merged_os['hour_next'].apply(
                lambda h: hora_en_mercado(h, mercado)
            )
        ]

        merged_os = merged_os.dropna(subset=['open', 'close'])
        # Filtrar hora != 00:00
        merged_os = merged_os[merged_os['time'].dt.time != time(0, 0)]
        
        if merged_os.empty:
            continue
        
        # Calcular beneficios OS
        if nodo['label'] == 'UP':
            beneficios_brutos_os = merged_os['close'] - merged_os['open']
        else:  # 'DOWN'
            beneficios_brutos_os = merged_os['open'] - merged_os['close']
        
        pip_size, point_size = _pip_sizes(df_os['open'], symbolo)
        beneficios_pips_os = beneficios_brutos_os / pip_size
        spreads_pips_os = merged_os['spread'] * point_size / pip_size
        beneficios_netos_os = beneficios_pips_os - spreads_pips_os
        
        # Estadísticas OS
        aciertos_os = (beneficios_netos_os > 0).sum()
        total_os = len(beneficios_netos_os)
        
        if (aciertos_os < config['MinOperationsOS'] or 
            total_os == 0):
            continue
        
        porcentaje_aciertos_os = aciertos_os / total_os
        if (porcentaje_aciertos_os < config['MinSuccessRate'] or 
            porcentaje_aciertos_os > config['MaxSuccessRate']):
            continue
        
        progressive_os = total_os / total_filas_os
        
        # PROCESAMIENTO IS (similar a OS)
        merged_is = pd.merge(
            df_filtrado_is[['time']],
            df_is[['time', 'open', 'close', 'spread']],
            on='time',
            how='left'
        )
        # ===============================
        # ENTRADA EN VELA SIGUIENTE (+1)
        # ===============================
        merged_is[['open', 'close', 'spread']] = (
            merged_is[['open', 'close', 'spread']].shift(-1)
        )
        
        merged_is['hour_next'] = merged_is['time'].shift(-1).dt.hour

        merged_is = merged_is[
            merged_is['hour_next'].apply(
                lambda h: hora_en_mercado(h, mercado)
            )
        ]

        merged_is = merged_is.dropna(subset=['open', 'close'])
        
        merged_is = merged_is[merged_is['time'].dt.time != time(0, 0)]
        
        if merged_is.empty:
            continue
        
        if nodo['label'] == 'UP':
            beneficios_brutos_is = merged_is['close'] - merged_is['open']
        else:
            beneficios_brutos_is = merged_is['open'] - merged_is['close']
        
        beneficios_pips_is = beneficios_brutos_is / pip_size
        spreads_pips_is = merged_is['spread'] * point_size / pip_size
        beneficios_netos_is = beneficios_pips_is - spreads_pips_is
        
        aciertos_is = (beneficios_netos_is > 0).sum()
        total_is = len(beneficios_netos_is)
        
        if (aciertos_is < config['MinOperationsIS'] or 
            total_is == 0):
            continue
        
        porcentaje_aciertos_is = aciertos_is / total_is
        if (porcentaje_aciertos_is < config['MinSuccessRate'] or 
            porcentaje_aciertos_is > config['MaxSuccessRate']):
            continue
        
        progressive_is = total_is / total_filas_is
        progressiveVariation = abs(progressive_os - progressive_is)
        
        if progressiveVariation > config['ProgressiveVariation']:
            continue
        
        # Consulta DB (esta parte ya es rápida)
        list_dates_is = merged_is['time'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
       
        
        nodo_mas_parecido = db_query.nodo_con_mas_fechas_hora_comunes(symbol, symbol, mercado, list_dates_is)
        if nodo_mas_parecido:
            coincidencias = nodo_mas_parecido['coincidencias']
            total_operaciones = nodo_mas_parecido['total_operations']
            porciento_nodo_db = coincidencias/total_operaciones
            porciento_is = coincidencias/total_is
            porciento = (porciento_nodo_db + porciento_is)/2
            if porciento >=config['SimilarityMax'] and nodo_mas_parecido['total_operations'] < total_is:
                db_query.eliminar_nodo_y_registros(nodo_mas_parecido['node_id']) 
            elif porciento >=config['SimilarityMax'] and nodo_mas_parecido['total_operations'] >= total_is: 
                print('Mayor pero el de la db mejor')
                continue
        
        # Insertar en DB
        db_query.insertar_nodo_con_registros(
            principal_symbol=symbol,
            symbol_cruce=symbol,
            label=nodo['label'],
            mercado= mercado,
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
        print(f"Nodo insertado: {nodo['label']}  - Aciertos IS: {aciertos_is}/{total_is} ({porcentaje_aciertos_is:.2%}) - Aciertos OS: {aciertos_os}/{total_os} ({porcentaje_aciertos_os:.2%})")
        
           
def procesar_archivo(file: str, symbol, mercado):
    try:
        df = pd.read_parquet(f"output/{symbol}/extrac/{file}")
        node_generator = NodeGenerator(df)
        operaciones_exitosas_UP = 0
        operaciones_exitosas_DOWN = 0
        while operaciones_exitosas_UP < config['NumMaxOperations'] or operaciones_exitosas_DOWN < config['NumMaxOperations']:
            list_nodos = node_generator.generar_nodos(100)
            selecte_nodes(file, operaciones_exitosas_DOWN, operaciones_exitosas_UP, symbol, list_nodos, mercado)
            operaciones_exitosas_UP = db_query.successful_operations_by_label(principal_symbol=symbol, symbol_cruce=symbol, label='UP', mercado=mercado)
            operaciones_exitosas_DOWN = db_query.successful_operations_by_label(principal_symbol=symbol, symbol_cruce=symbol, label='DOWN', mercado=mercado)
    except Exception:
        logger.exception(
            "Error procesando archivo %s (symbol=%s, mercado=%s)",
            file,
            symbol,
            mercado
        )
        raise
                    

def execute_node_builder(symbol, mercados):  
    peticiones.initialize_mt5()
    
    for mercado in mercados:
        db_query.eliminar_nodos_y_registros(principal_symbol=symbol, symbol_cruce=symbol, mercado=mercado)
        
    list_files = os.listdir(f'output/{symbol}/extrac')
    
    
    MAX_PROCESOS = len(list_files)
    if MAX_PROCESOS > 25:
        MAX_PROCESOS = 25
    # Puedes ajustar este número según tu CPU
    for mercado in mercados:
        futures = []

        with ProcessPoolExecutor(max_workers=MAX_PROCESOS) as executor:
            for file in list_files:
                future = executor.submit(procesar_archivo, file, symbol, mercado)
                futures.append(future)

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    logger.exception("Error en proceso hijo durante execute_node_builder")


        
        
