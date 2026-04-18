import os
import json
import logging
from datetime import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import operator

import pandas as pd
import numpy as np
from src.db import query as db_query
from src.routes import peticiones
from src.signals.event_generator import add_event_features
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


def max_losing_streak(values):
    streak = 0
    max_streak = 0
    for value in values:
        if value < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def calculate_node_quality_stats(pips_values, num_conditions=3):
    values = pd.Series(pips_values, dtype='float64').dropna().to_numpy()
    if values.size == 0:
        return None

    expectancy = float(values.mean())
    gross_profit = float(values[values > 0].sum())
    gross_loss = float(abs(values[values < 0].sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (2.0 if gross_profit > 0 else 0.0)
    std_dev = float(values.std())
    sharpe_like = float(expectancy / std_dev) if std_dev > 1e-8 else (2.0 if expectancy > 0 else 0.0)
    equity_curve = values.cumsum()
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = equity_curve - running_max
    max_drawdown = float(abs(drawdowns.min())) if drawdowns.size else 0.0
    drawdown_ratio = max_drawdown / max(abs(float(values.sum())), 1.0)
    num_trades = int(values.size)
    winrate = float((values > 0).mean()) if num_trades else 0.0

    quality_base = float(
        (expectancy * 1.5) +
        (min(profit_factor, 3.0) * 2.0) +
        (min(sharpe_like, 2.5) * 1.5) -
        (drawdown_ratio * 2.0) -
        (max(max_losing_streak(values) - 3, 0) * 0.25)
        + (winrate * 0.5)
    )

    # Penaliza edges con poca muestra y nodos excesivamente complejos.
    trade_weight = float(num_trades / (num_trades + 50.0))
    complexity_factor = float(1.0 / (1.0 + 0.15 * max(int(num_conditions) - 3, 0)))
    quality_score = float(quality_base * trade_weight * complexity_factor)

    return {
        'expectancy': expectancy,
        'profit_factor': float(profit_factor),
        'sharpe_like': sharpe_like,
        'drawdown_ratio': float(drawdown_ratio),
        'winrate': winrate,
        'num_trades': num_trades,
        'trade_weight': trade_weight,
        'complexity_factor': complexity_factor,
        'quality_score': quality_score,
    }


def passes_quality_filters(stats_is, stats_os):
    min_quality_score_is = float(config.get('MinNodeQualityScoreIS', 0.5))
    min_quality_score_os = float(config.get('MinNodeQualityScoreOS', 0.25))
    min_expectancy_is = float(config.get('MinNodeExpectancyIS', 0.0))
    min_expectancy_os = float(config.get('MinNodeExpectancyOS', -0.25))
    min_profit_factor_is = float(config.get('MinNodeProfitFactorIS', 1.02))
    min_profit_factor_os = float(config.get('MinNodeProfitFactorOS', 0.98))
    max_quality_gap = float(config.get('MaxNodeQualityGap', 2.0))

    if stats_is is None or stats_os is None:
        return False

    if stats_is['quality_score'] < min_quality_score_is:
        return False
    if stats_os['quality_score'] < min_quality_score_os:
        return False
    if stats_is['expectancy'] < min_expectancy_is:
        return False
    if stats_os['expectancy'] < min_expectancy_os:
        return False
    if stats_is['profit_factor'] < min_profit_factor_is:
        return False
    if stats_os['profit_factor'] < min_profit_factor_os:
        return False
    if abs(stats_is['quality_score'] - stats_os['quality_score']) > max_quality_gap:
        return False

    return True


def enrich_with_event_features(indicators_df, prices_df):
    enriched = indicators_df.copy()
    if 'time' in enriched.columns and not pd.api.types.is_datetime64_any_dtype(enriched['time']):
        enriched['time'] = pd.to_datetime(enriched['time'])

    price_frame = prices_df.copy()
    if 'time' in price_frame.columns and not pd.api.types.is_datetime64_any_dtype(price_frame['time']):
        price_frame['time'] = pd.to_datetime(price_frame['time'])

    price_columns = ['open', 'high', 'low', 'close']
    missing_columns = [column for column in price_columns if column not in enriched.columns]

    if missing_columns:
        merge_columns = ['time'] + [column for column in missing_columns if column in price_frame.columns]
        if len(merge_columns) > 1:
            enriched = enriched.merge(
                price_frame[merge_columns].drop_duplicates(subset='time'),
                on='time',
                how='left'
            )

    return add_event_features(enriched)



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


def selecte_nodes(file: str, op_down, op_up, symbol, list_nodos, mercado, log_q=None):
    """
    Versión optimizada con NumPy/Pandas vectorizado
    """
    ext = file.split('_')[0]
    symbolo = file.split('_')[1]
    
    # Cargar DataFrames (una vez)
    list_dire_indicators_os = os.listdir(f'output/symbol_data/{symbol}/extrac_os')
    dire = next((f for f in list_dire_indicators_os if ext in f), None)
    
    if not dire:
        return
    
    
    indicators_is = pd.read_parquet(f'output/symbol_data/{symbol}/extrac/{file}')
    # Convertir time a datetime ANTES de slice/copy
    if 'time' in indicators_is.columns and not pd.api.types.is_datetime64_any_dtype(indicators_is['time']):
        indicators_is['time'] = pd.to_datetime(indicators_is['time'])
    
    df_bas = pd.read_csv(f'output/symbol_data/{symbol}/is_os/is.csv')
    # Convertir time a datetime ANTES de slice/copy
    if 'time' in df_bas.columns and not pd.api.types.is_datetime64_any_dtype(df_bas['time']):
        df_bas['time'] = pd.to_datetime(df_bas['time'])

    indicators_is = enrich_with_event_features(indicators_is, df_bas)

    df_indicators_os = indicators_is.iloc[int(len(indicators_is)*0.8):].copy()  # Para no modificar el original
    df_indicators_is = indicators_is.iloc[:int(len(indicators_is)*0.8)].copy()  # Para no modificar el original
    
    df_os = df_bas.iloc[int(len(df_bas)*0.8):].copy()  # Para no modificar el original
    df_is = df_bas.iloc[:int(len(df_bas)*0.8)].copy()
    
    # Calcular hora para todos los DataFrames (ahora time es datetime garantizado)
    for df in [df_os, df_is, df_indicators_os, df_indicators_is]:
        if 'time' in df.columns and pd.api.types.is_datetime64_any_dtype(df['time']):
            # ⭐ PRECALCULO CRÍTICO
            df.loc[:, 'hour'] = df['time'].dt.hour
            
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
        stats_os = calculate_node_quality_stats(
            beneficios_netos_os,
            num_conditions=nodo.get('num_conditions', len(conditions)),
        )
        
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
        stats_is = calculate_node_quality_stats(
            beneficios_netos_is,
            num_conditions=nodo.get('num_conditions', len(conditions)),
        )
        if not passes_quality_filters(stats_is, stats_os):
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
                if log_q is not None:
                    log_q.put('Mayor pero el de la db mejor')
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
            stats_is=stats_is,
            stats_os=stats_os,
            fechas=list_dates_is,
            veneficios=beneficios_netos_is.round(2).tolist(),
            fechas_os=merged_os['time'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            veneficios_os=beneficios_netos_os.round(2).tolist()
        )
        msg = (
            f"Nodo insertado: {nodo['label']} - Aciertos IS: {aciertos_is}/{total_is} ({porcentaje_aciertos_is:.2%}) "
            f"- Aciertos OS: {aciertos_os}/{total_os} ({porcentaje_aciertos_os:.2%}) "
            f"- score_is={stats_is['quality_score']:.3f} score_os={stats_os['quality_score']:.3f}"
        )
        print(msg)
        if log_q is not None:
            log_q.put(msg)
        
           
def procesar_archivo(file: str, symbol, mercado, log_q=None):
    try:
        df = pd.read_parquet(f"output/symbol_data/{symbol}/extrac/{file}")
        df_bas = pd.read_csv(f'output/symbol_data/{symbol}/is_os/is.csv')
        if 'time' in df_bas.columns and not pd.api.types.is_datetime64_any_dtype(df_bas['time']):
            df_bas['time'] = pd.to_datetime(df_bas['time'])
        df = enrich_with_event_features(df, df_bas)
        df_generator = df.iloc[:int(len(df)*0.2)].copy()  # Para no modificar el original
        node_generator = NodeGenerator(df_generator)
        operaciones_exitosas_UP = 0
        operaciones_exitosas_DOWN = 0
        while operaciones_exitosas_UP < config['NumMaxOperations'] or operaciones_exitosas_DOWN < config['NumMaxOperations']:
            list_nodos = node_generator.generar_nodos(100)
            selecte_nodes(file, operaciones_exitosas_DOWN, operaciones_exitosas_UP, symbol, list_nodos, mercado, log_q=log_q)
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
                    

def execute_node_builder(symbol, mercados, log_q=None):  
    peticiones.initialize_mt5()
         
    list_files = os.listdir(f'output/symbol_data/{symbol}/extrac')

    for mercado in mercados:
        MAX_PROCESOS = int(config.get('use_proces', 25))
        futures = []
        with ProcessPoolExecutor(max_workers=MAX_PROCESOS) as executor:
            for i in range(MAX_PROCESOS):
                indice = i % len(list_files)
                file = list_files[indice]
                future = executor.submit(procesar_archivo, file, symbol, mercado, log_q)
                futures.append(future)

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print(f"Error en proceso hijo durante execute_node_builder: {exc}")
                    logger.exception("Error en proceso hijo durante execute_node_builder")


        
        