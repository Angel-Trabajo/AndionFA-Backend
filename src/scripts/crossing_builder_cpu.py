import json
import os
import sys
import logging
import threading
import time as tim
import operator
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


import pandas as pd 
import numpy as np

from src.routes import peticiones
from src.db import query as db_query
from src.signals.event_generator import add_event_features
from src.utils.constructor_node import NodeGenerator
from src.utils.extrat_data_for_crossing import select_symbols_correl
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


def calculate_node_quality_stats(pips_list, num_conditions=3):
    values = np.asarray(pips_list, dtype=np.float64)
    if values.size == 0:
        return None

    expectancy = float(np.mean(values))
    gross_profit = float(values[values > 0].sum())
    gross_loss = float(abs(values[values < 0].sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (2.0 if gross_profit > 0 else 0.0)
    std_dev = float(np.std(values))
    sharpe_like = float(expectancy / std_dev) if std_dev > 1e-8 else (2.0 if expectancy > 0 else 0.0)
    equity_curve = np.cumsum(values)
    running_max = np.maximum.accumulate(equity_curve) if values.size else np.array([], dtype=np.float64)
    drawdowns = equity_curve - running_max if values.size else np.array([], dtype=np.float64)
    max_drawdown = float(abs(drawdowns.min())) if drawdowns.size else 0.0
    drawdown_ratio = max_drawdown / max(abs(float(values.sum())), 1.0)
    loss_streak = max_losing_streak(values)
    num_trades = int(values.size)
    winrate = float((values > 0).mean()) if num_trades else 0.0

    quality_base = float(
        (expectancy * 1.5) +
        (min(profit_factor, 3.0) * 2.0) +
        (min(sharpe_like, 2.5) * 1.5) -
        (drawdown_ratio * 2.0) -
        (max(loss_streak - 3, 0) * 0.25)
        + (winrate * 0.5)
    )
    trade_weight = float(num_trades / (num_trades + 50.0))
    complexity_factor = float(1.0 / (1.0 + 0.15 * max(int(num_conditions) - 3, 0)))
    quality_score = float(quality_base * trade_weight * complexity_factor)

    return {
        'expectancy': expectancy,
        'profit_factor': float(profit_factor),
        'sharpe_like': sharpe_like,
        'max_drawdown': max_drawdown,
        'drawdown_ratio': float(drawdown_ratio),
        'max_losing_streak': int(loss_streak),
        'winrate': winrate,
        'num_trades': num_trades,
        'trade_weight': trade_weight,
        'complexity_factor': complexity_factor,
        'quality_score': quality_score,
    }


def passes_quality_filters(stats_is, stats_os, config):
    general = config.get('general', {})
    min_quality_score_is = float(general.get('MinNodeQualityScoreIS', 0.5))
    min_quality_score_os = float(general.get('MinNodeQualityScoreOS', 0.25))
    min_expectancy_is = float(general.get('MinNodeExpectancyIS', 0.0))
    min_expectancy_os = float(general.get('MinNodeExpectancyOS', -0.25))
    min_profit_factor_is = float(general.get('MinNodeProfitFactorIS', 1.02))
    min_profit_factor_os = float(general.get('MinNodeProfitFactorOS', 0.98))
    max_quality_gap = float(general.get('MaxNodeQualityGap', 2.0))

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
    working_df = df.copy()
    missing_columns = [column for column in condition_cols if column not in working_df.columns]
    for column in missing_columns:
        working_df[column] = 0.0
    data = working_df[condition_cols].to_numpy(dtype=np.float64)
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
        mercado,
        log_q=None
    ):
    
    por_direccion = config['general']['por_direccion']
    list_symbols_inversos = config['symbol']['list_symbol_inversos']
    principal_symbol = config['principal_symbol']
    list_symbol = config['symbol']['list_symbol']
    ini = tim.time()
    cant_nodos = int(config['general'].get('cant_nodos', 1000))
    cant_nodos = max(50, cant_nodos)
    list_nodos = node_generator.generar_nodos(cant_nodos)
    _msg_nodos = f"{cant_nodos} Nodos generados en {tim.time() - ini:.4f} segundos"
    print(_msg_nodos)
    if log_q is not None:
        log_q.put(_msg_nodos)
    
    # pip_size/point_size se calculan tras cargar precios
    if por_direccion:
        if symbol in list_symbols_inversos:
            list_nodos = [nodo for nodo in list_nodos if nodo['label'] != action]
        else:
            list_nodos = [nodo for nodo in list_nodos if nodo['label'] == action]

    is_path = f'output/symbol_data/{symbol}/extrac/{file}'
    indicators_is = pd.read_parquet(is_path)
    # Convertir time a datetime ANTES de slice/copy
    if 'time' in indicators_is.columns and not pd.api.types.is_datetime64_any_dtype(indicators_is['time']):
        indicators_is['time'] = pd.to_datetime(indicators_is['time'])
     
    df_bas = load_csv_cached(f'output/symbol_data/{principal_symbol}/is_os/is.csv')
    # Convertir time a datetime ANTES de slice/copy
    if 'time' in df_bas.columns and not pd.api.types.is_datetime64_any_dtype(df_bas['time']):
        df_bas['time'] = pd.to_datetime(df_bas['time'])

    indicators_is = enrich_with_event_features(indicators_is, df_bas)

    df_indicators_os = indicators_is.iloc[int(len(indicators_is)*0.8):].copy()  # Para no modificar el original
    df_indicators_is = indicators_is.iloc[:int(len(indicators_is)*0.8)].copy()  # Para no modificar el original
    
    df_os = df_bas.iloc[int(len(df_bas)*0.8):].copy()  # Para no modificar el original
    df_is = df_bas.iloc[:int(len(df_bas)*0.8)].copy()  # Para no modificar el original
    # Beneficios y spreads se miden sobre precio del principal_symbol.
    pip_size, point_size = _pip_sizes(df_os['open'], principal_symbol)
    
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

    # Importante: usar posiciones (0..n-1), no labels de índice.
    # Con labels heredados del DataFrame original, .iloc puede romper con out-of-bounds.
    os_match_pos = np.flatnonzero(df_indicators_os['time'].isin(fechas_dt_os).to_numpy())
    is_match_pos = np.flatnonzero(df_indicators_is['time'].isin(fechas_dt_is).to_numpy())

    df_indicators_os_anteriores = os_match_pos[os_match_pos > 0] - 1
    df_indicators_is_anteriores = is_match_pos[is_match_pos > 0] - 1

    df_indicators_os_fil = df_indicators_os.iloc[df_indicators_os_anteriores].copy()
    df_indicators_is_fil = df_indicators_is.iloc[df_indicators_is_anteriores].copy()
    
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
        if not (prev_os + (porcent_aumento_os - 0.025) <= porcentaje_os <= prev_os + (porcent_aumento_os + 0.025)):
            continue        
        stats_os = calculate_node_quality_stats(
            list_beneficio_os,
            num_conditions=nodo.get('num_conditions', len(conditions)),
        )
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
       
        if not (prev_is + (porcent_aumento_is - 0.025) <= porcentaje_aciertos_is <= prev_is + (porcent_aumento_is + 0.025)):
            continue
        stats_is = calculate_node_quality_stats(
            list_beneficio_is,
            num_conditions=nodo.get('num_conditions', len(conditions)),
        )
        if not passes_quality_filters(stats_is, stats_os, config):
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
                if log_q is not None:
                    log_q.put('Mayor pero el de la db mejor')
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
            stats_is=stats_is,
            stats_os=stats_os,
            fechas=list_dates_is,
            veneficios=list_beneficio_is,
            fechas_os=list_dates_os,
            veneficios_os=list_beneficio_os,    
        )
        _msg_nodo = (
            f"Nodo insertado: symbol={symbol} action={action} mercado={mercado} "
            f"porcentaje_is={porcentaje_aciertos_is:.4f} porcentaje_os={porcentaje_os:.4f} "
            f"score_is={stats_is['quality_score']:.3f} score_os={stats_os['quality_score']:.3f} "
            f"pf_is={stats_is['profit_factor']:.2f} pf_os={stats_os['profit_factor']:.2f} "
            f"total_is={total_is} total_os={total}"
        )
        print(_msg_nodo)
        if log_q is not None:
            log_q.put(_msg_nodo)


def procesar_archivo(file: str, symbol, action, cont, prev_os, prev_is, porcent_aumento_os, porcent_aumento_is, NumMaxOperations, config, mercado, log_q=None):
    principal_symbol = config['principal_symbol']
    df = pd.read_parquet(f'output/symbol_data/{symbol}/extrac/{file}')
    df_bas = load_csv_cached(f'output/symbol_data/{principal_symbol}/is_os/is.csv')
    df = enrich_with_event_features(df, df_bas)
    df_generator = df.iloc[:int(len(df)*0.2)].copy()  # Para no modificar el original
    node_generator = NodeGenerator(df_generator)
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
            mercado,
            log_q=log_q,
        )
        operaciones_exitosas = (
            db_query.successful_operations_by_label(
                principal_symbol=principal_symbol,
                symbol_cruce=symbol,
                label=action,
                mercado=mercado
            
            )
        )

        _msg_ops = (
            f"Operaciones exitosas mercado {mercado} "
            f"{symbol}-{action}: "
            f"{operaciones_exitosas}"
        )
        print(_msg_ops)
        if log_q is not None:
            log_q.put(_msg_ops)
        
   

def calcular_porcentage(symbol, prev, config):
    sumatoria = 0
    dict_symbol_correl = config["symbol"]["dict_symbol_correl"]
    for value in dict_symbol_correl.values():
        sumatoria += value
    corre = dict_symbol_correl[symbol]
    return corre/sumatoria * (1-prev)


def create_trees(symbol, action, cont, prev_os, prev_is, NumMaxOperations, mercado, config, log_q=None):
   
    list_files = os.listdir(
        f'output/symbol_data/{symbol}/extrac'
    )
    
    porcent_aumento_os =calcular_porcentage(symbol, prev_os, config)
    porcent_aumento_is =calcular_porcentage(symbol, prev_is, config)
    
    MAX_PROCESOS =  int(config['general'].get('use_proces', 25))//2# ajustable según CPU
    
    
    with ProcessPoolExecutor(
        max_workers=MAX_PROCESOS,
    ) as executor:

        futures = []
        for i in range(MAX_PROCESOS):
            indice = i % len(list_files)  # Para asegurar que el índice no exceda el número de archivos
            file = list_files[indice]
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
                    mercado,
                    log_q,
                )
            )
            
        for future in as_completed(futures):
            try:
                future.result()
            except Exception:
                logger.exception("Error en proceso hijo de create_trees")

    
def _execute_crossing_builder(action, config, log_q=None):
     
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
            
            print(f"{prev_os} {prev_is} ---------")
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

            create_trees(symbol, action, cont, prev_os, prev_is, NumMaxOperations, mercado, config, log_q=log_q)
            cont += 1
            prosedio = True
            print(f"NumMaxOperations actual: {NumMaxOperations}")
         

def execute_crossing_builder(principal_symbol, list_mercado, log_q=None):
    inicio =tim.time()           
    peticiones.initialize_mt5()
    tim.sleep(3)
    
    select_symbols_correl(principal_symbol)
    
    with open('config/general_config.json', 'r', encoding='utf-8') as f:
        general_config = json.load(f)
    with open(f'config/divisas/{principal_symbol}/config_{principal_symbol}.json', 'r', encoding='utf-8') as f:
        config_symbol = json.load(f)
            
    config = {
        "general": general_config,
        "symbol": config_symbol,
        "principal_symbol": principal_symbol,
        "list_mercado": list_mercado
    }

    # Usamos hilos para UP y DOWN: comparten sys.stdout con el proceso padre,
    # por lo que todos los print() llegan automáticamente al frontend.
    t1 = threading.Thread(target=_execute_crossing_builder, args=('UP', config, log_q), daemon=False)
    t2 = threading.Thread(target=_execute_crossing_builder, args=('DOWN', config, log_q), daemon=False)
    t1.start()
    tim.sleep(5)  # Esperar un poco antes de iniciar el segundo hilo
    t2.start()

    t1.join()
    t2.join()
    print("Ambos hilos han terminado.")
    print(f"Tiempo de creación: {tim.time() - inicio:.4f} segundos")
 
if __name__ == "__main__":
    execute_crossing_builder('EURCAD', ['Asia', 'Europa', 'America'])