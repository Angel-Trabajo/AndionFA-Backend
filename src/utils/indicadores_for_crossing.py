import pandas as pd
import json
import numpy as np
import talib
import concurrent.futures
import os
from datetime import datetime
from src.utils.common_functions import get_previous_4_6


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


def extract_indicadores(principal_symbol, data=None):
    with open('config/general_config.json', 'r', encoding='utf8') as file:
        general_config = json.load(file)
    with open(f'config/divisas/{principal_symbol}/config_{principal_symbol}.json', 'r', encoding='utf8') as file:
        config_symbol = json.load(file)
        
    list_symbol = config_symbol['list_symbol']
    list_files = general_config['indicators_files']
    timeframe = general_config['timeframe']
    str_start, end = get_previous_4_6(general_config['dateStart'], general_config['dateEnd'])
    
    if data is not None:
        list_symbol.append(principal_symbol)
             
    for sym in list_symbol:
        print(f"Procesando divisa {principal_symbol} cruce :{sym}...")
        # Procesar EN PARALELO ambos folders
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(_create_files_fast, sym, timeframe, 
                                     general_config['dateStart'], general_config['dateEnd'], 
                                     list_files, 'extrac_os', principal_symbol, data)
            future2 = executor.submit(_create_files_fast, sym, timeframe, 
                                     str_start, end, list_files, 'extrac',principal_symbol, data)
            concurrent.futures.wait([future1, future2])


def _create_files_fast(symbol, timeframe, str_start, end, indicators_files, folder, principal_symbol, data):
    
     # Leer CSV inmediatamente
    if data is not None:
        df = data
    else:
        if symbol == principal_symbol:
            path = f'output/{principal_symbol}'
        else:
            path = f'output/{principal_symbol}/crossing/{symbol}'
        
        if folder == 'extrac':
            df= pd.read_csv(f'{path}/is_os/is.csv')
        else:
            df= pd.read_csv(f'{path}/is_os/os.csv')
    
        start = datetime.strptime(str_start, '%Y-%m-%d')
        pos, _ = _buscar_fecha_o_siguiente_fast(df, start)    
    # PRECALCULAR arrays numpy UNA sola vez - CRÍTICO para velocidad
    high = df['high'].to_numpy(np.float64)
    low = df['low'].to_numpy(np.float64)
    close = df['close'].to_numpy(np.float64)
    open_arr = df['open'].to_numpy(np.float64)
    time_arr = df['time'].to_numpy('datetime64[ns]')
     
    # Procesar TODOS los indicadores en paralelo
    if data is None:
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(30, len(indicators_files), os.cpu_count())) as executor:
            futures = {}
            
            for file in indicators_files:
                future = executor.submit(
                    _generate_files_fast, 
                    file, pos, symbol, str_start, end, timeframe, folder,
                    high, low, close, open_arr, time_arr, df, principal_symbol
                )
                futures[future] = file
            
            # Esperar resultados
            for future in concurrent.futures.as_completed(futures):
                file = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Error {file}: {e}")
    else:
        for file in indicators_files:
            _generate_files_fast(file, 1, symbol, str_start, end, timeframe,'extrac_os' ,high, low, close, open_arr, time_arr, df, principal_symbol, True)


def _generate_files_fast(indicator_file, pos, symbol, start, end, timeframe, folder,
                        high, low, close, open_arr, time_arr, df_ref, principal_symbol, is_data=False):
    """Versión MEGA optimizada de _generate_files"""
    
    # Leer archivo de configuración SIN caché (más rápido para pocos archivos)
    with open(f'config/extractor/{indicator_file}', 'r') as f:
        raw_lines = f.readlines()
    
    # Preparar datos base UNA sola vez
    data = {'time': time_arr[pos:]}
    
    # Diccionario local para máximo rendimiento
    price_map = {'[[CLOSE_PRICE]]': 'close', '[[HIGH_PRICE]]': 'high', '[[LOW_PRICE]]': 'low'}
    
    # Procesar cada línea
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split(';')
        if len(parts) < 2:
            continue
            
        indicator_name = parts[0]
        params = parts[1].split(',')
        
        # SWITCH optimizado usando if-elif
        if indicator_name == 'ADX':
            tp = int(params[-1])
            data[f'ADX_{tp}'] = talib.ADX(high, low, close, timeperiod=tp)[pos:]
            
        elif indicator_name == 'ADXR':
            tp = int(params[-1])
            data[f'ADXR_{tp}'] = talib.ADXR(high, low, close, timeperiod=tp)[pos:]
            
        elif indicator_name == 'APO':
            col = price_map.get(params[0], 'close')
            real = df_ref[col].to_numpy(np.float64)
            fast, slow = int(params[1]), int(params[2])
            matype_str = params[3].strip('[]')
            matype = replaceString.get(matype_str, 0)
            data[f'APO_{fast}_{slow}_{matype_str}'] = talib.APO(real, fastperiod=fast, slowperiod=slow, matype=matype)[pos:]
            
        elif indicator_name == 'AROON':
            tp = int(params[2])
            pos_out = 1 if params[3] == '[[OUTPUT2]]' else 0
            data[f'AROON_{tp}_pos{pos_out}'] = talib.AROON(high, low, timeperiod=tp)[pos_out][pos:]
            
        elif indicator_name == 'ATR':
            tp = int(params[-1])
            data[f'ATR_{tp}'] = talib.ATR(high, low, close, timeperiod=tp)[pos:]
            
        elif indicator_name == 'BOP':
            data['BOP'] = talib.BOP(open_arr, high, low, close)[pos:]
            
        elif indicator_name == 'CCI':
            tp = int(params[-1])
            data[f'CCI_{tp}'] = talib.CCI(high, low, close, timeperiod=tp)[pos:]
            
        elif indicator_name == 'CMO':
            col = price_map.get(params[0], 'close')
            real = df_ref[col].to_numpy(np.float64)
            tp = int(params[-1])
            data[f'CMO_{tp}'] = talib.CMO(real, timeperiod=tp)[pos:]
            
        elif indicator_name == 'DX':
            tp = int(params[-1])
            data[f'DX_{tp}'] = talib.DX(high, low, close, timeperiod=tp)[pos:]
            
        elif indicator_name == 'MACD':
            col = price_map.get(params[0], 'close')
            real = df_ref[col].to_numpy(np.float64)
            fast, slow, signal = int(params[1]), int(params[2]), int(params[3])
            pos_map = {'[[OUTPUT1]]': 0, '[[OUTPUT2]]': 1, '[[OUTPUT3]]': 2}
            pos_out = pos_map.get(params[4], 0)
            data[f'MACD_{fast}_{slow}_{signal}_pos{pos_out}'] = talib.MACD(real, fastperiod=fast, slowperiod=slow, signalperiod=signal)[pos_out][pos:]
            
        elif indicator_name == 'MINUS_DI':
            tp = int(params[-1])
            data[f'MINUS_DI_{tp}'] = talib.MINUS_DI(high, low, close, timeperiod=tp)[pos:]
            
        elif indicator_name == 'MINUS_DM':
            tp = int(params[-1])
            data[f'MINUS_DM_{tp}'] = talib.MINUS_DM(high, low, timeperiod=tp)[pos:]
            
        elif indicator_name == 'MOM':
            col = price_map.get(params[0], 'close')
            real = df_ref[col].to_numpy(np.float64)
            tp = int(params[-1])
            data[f'MOM_{tp}'] = talib.MOM(real, timeperiod=tp)[pos:]
            
        elif indicator_name == 'NATR':
            tp = int(params[-1])
            data[f'NATR_{tp}'] = talib.NATR(high, low, close, timeperiod=tp)[pos:]
            
        elif indicator_name == 'PLUS_DI':
            tp = int(params[-1])
            data[f'PLUS_DI_{tp}'] = talib.PLUS_DI(high, low, close, timeperiod=tp)[pos:]
            
        elif indicator_name == 'PLUS_DM':
            tp = int(params[-1])
            data[f'PLUS_DM_{tp}'] = talib.PLUS_DM(high, low, timeperiod=tp)[pos:]
            
        elif indicator_name == 'PPO':
            col = price_map.get(params[0], 'close')
            real = df_ref[col].to_numpy(np.float64)
            fast, slow = int(params[1]), int(params[2])
            matype_str = params[3].strip('[]')
            matype = replaceString.get(matype_str, 0)
            data[f'PPO_{fast}_{slow}_{matype_str}'] = talib.PPO(real, fastperiod=fast, slowperiod=slow, matype=matype)[pos:]
            
        elif indicator_name == 'ROC':
            col = price_map.get(params[0], 'close')
            real = df_ref[col].to_numpy(np.float64)
            tp = int(params[1])
            data[f'ROC_{tp}'] = talib.ROC(real, timeperiod=tp)[pos:]
            
        elif indicator_name == 'RSI':
            col = price_map.get(params[0], 'close')
            real = df_ref[col].to_numpy(np.float64)
            tp = int(params[-1])
            data[f'RSI_{tp}'] = talib.RSI(real, timeperiod=tp)[pos:]
            
        elif indicator_name == 'STDDEV':
            col = price_map.get(params[0], 'close')
            real = df_ref[col].to_numpy(np.float64)
            tp, nbdev = int(params[1]), float(params[2])
            data[f"STDDEV_{tp}_{str(nbdev).replace('.', '')}"] = talib.STDDEV(real, timeperiod=tp, nbdev=nbdev)[pos:]
            
        elif indicator_name == 'STOCHF':
            fastk, fastd = int(params[3]), int(params[4])
            matype_str = params[5].strip('[]')
            matype = replaceString.get(matype_str, 0)
            pos_out = 1 if params[6] == '[[OUTPUT2]]' else 0
            data[f'STOCHF_{fastk}_{fastd}_{matype_str}_pos{pos_out}'] = talib.STOCHF(high, low, close, fastk_period=fastk, fastd_period=fastd, fastd_matype=matype)[pos_out][pos:]
            
        elif indicator_name == 'STOCH':
            fastk, slowk = int(params[3]), int(params[4])
            matype1 = params[5].strip('[]')
            slowk_matype = replaceString.get(matype1, 0)
            slowd = int(params[6])
            matype2 = params[7].strip('[]')
            slowd_matype = replaceString.get(matype2, 0)
            stoch_result = talib.STOCH(high, low, close, fastk_period=fastk, slowk_period=slowk, slowk_matype=slowk_matype, slowd_period=slowd, slowd_matype=slowd_matype)
            data[f'STOCH_{fastk}_{slowk}_{matype1}_{slowd}_{matype2}_pos0'] = stoch_result[0][pos:]
            data[f'STOCH_{fastk}_{slowk}_{matype1}_{slowd}_{matype2}_pos1'] = stoch_result[1][pos:]
            
        elif indicator_name == 'STOCHRSI':
            col = price_map.get(params[0], 'close')
            real = df_ref[col].to_numpy(np.float64)
            tp, fastk, fastd = int(params[1]), int(params[2]), int(params[3])
            matype_str = params[4].strip('[]')
            matype = replaceString.get(matype_str, 0)
            pos_out = 1 if params[5] == '[[OUTPUT2]]' else 0
            data[f'STOCHRSI_{tp}_{fastk}_{fastd}_{matype_str}_pos{pos_out}'] = talib.STOCHRSI(real, timeperiod=tp, fastk_period=fastk, fastd_period=fastd, fastd_matype=matype)[pos_out][pos:]
            
        elif indicator_name == 'TRANGE':
            data['TRANGE'] = talib.TRANGE(high, low, close)[pos:]
            
        elif indicator_name == 'ULTOSC':
            tp1, tp2, tp3 = int(params[3]), int(params[4]), int(params[5])
            data[f'ULTOSC_{tp1}_{tp2}_{tp3}'] = talib.ULTOSC(high, low, close, timeperiod1=tp1, timeperiod2=tp2, timeperiod3=tp3)[pos:]
            
        elif indicator_name == 'VAR':
            col = price_map.get(params[0], 'close')
            real = df_ref[col].to_numpy(np.float64)
            tp, nbdev = int(params[1]), float(params[2])
            data[f"VAR_{tp}_{str(nbdev).replace('.','')}"] = talib.VAR(real, timeperiod=tp, nbdev=nbdev)[pos:]
            
        elif indicator_name == 'WILLR':
            tp = int(params[-1])
            data[f'WILLR_{tp}'] = talib.WILLR(high, low, close, timeperiod=tp)[pos:]
    
    # Crear DataFrame final - OPTIMIZADO
    output_df = pd.DataFrame(data)
    
    # Añadir open/close para labels
    output_df['open'] = open_arr[pos:]
    output_df['close'] = close[pos:]
    
    # Crear labels VECTORIZADO
    output_df['label'] = np.where(
        output_df['close'] > output_df['open'], 'UP',
        np.where(output_df['close'] < output_df['open'], 'DOWN', None)
    )
    
    # Filtrar y limpiar
    output_df = output_df[output_df['label'].notna()].copy()
    output_df = output_df.drop(columns=['open', 'close'])
    output_df = output_df.reset_index(drop=True)
    
    # Guardar archivo
    name_f = f'_{symbol}_{start.replace("-", "")}_{end.replace("-", "")}_timeframe{timeframe}.parquet'
    
    if symbol == principal_symbol:
         output_path = f'output/{principal_symbol}/{folder}/{indicator_file.replace(".csv", name_f)}'
    else:
        output_path = f'output/{principal_symbol}/crossing/{symbol}/{folder}/{indicator_file.replace(".csv", name_f)}'
    if is_data:
        df_in = pd.read_parquet(output_path)
        output_df = pd.concat([df_in, output_df], ignore_index=True)
        output_df = output_df.drop_duplicates(subset=['time'])
        output_df.to_parquet(output_path, index=False, engine='pyarrow', compression='snappy')
    else:
        output_df.to_parquet(output_path, index=False, engine='pyarrow', compression='snappy')


def _buscar_fecha_o_siguiente_fast(df, fecha_objetivo):
    """Versión RÁPIDA de búsqueda de fecha"""
    # Convertir solo si es necesario
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        df = df.copy()
        df['time'] = pd.to_datetime(df['time'])
    
    # Búsqueda directa - más rápido para DataFrames no enormes
    for idx, fecha in enumerate(df['time']):
        if fecha >= fecha_objetivo:
            return idx, fecha
    
    return None, None


