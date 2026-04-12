from datetime import datetime, timedelta
import concurrent.futures
import sys
import os
import json

import pandas as pd
import talib
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.common_functions import limpiar_carpeta, crear_carpeta_si_no_existe
from src.routes import peticiones

PATH_BASE = 'output/symbol_data/{}'
PATH_DATA_IS = 'output/symbol_data/{}/is_os/is.csv'
PATH_DATA_OS = 'output/symbol_data/{}/is_os/os.csv'


_mapping_time = peticiones.get_timeframes()['timeframes']


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


def _get_data_mt5(symbol, timeframe, start, end, folder):
    
    timeframe = _mapping_time.get(timeframe)
    
    response = peticiones.get_historical_data(symbol, timeframe, start, end)
    if 'error' in response:
        raise RuntimeError(f"MT5 data error for {symbol} [{start} - {end}]: {response['error']}")
    rates = response['data']
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    if folder == 'extrac':
        df.to_csv(PATH_DATA_IS.format(symbol), index=False)
    else:
        df.to_csv(PATH_DATA_OS.format(symbol), index=False)
        
    return df


def _buscar_fecha_o_siguiente(df, fecha_objetivo):
   
    df['time'] = pd.to_datetime(df['time'])  # asegurar tipo datetime

    while True:
        posiciones = df.index[df['time'] == fecha_objetivo].tolist()
        if posiciones:
            return posiciones[0], fecha_objetivo
        fecha_objetivo += timedelta(days=1) 


def _generate_files(indicator_file, pos, symbol, start, end, timeframe, folder):
    with open(f'config/extractor/{indicator_file}', 'r') as f:
        raw_lines = f.readlines()
    
    price_column_map = {
        '[[CLOSE_PRICE]]': 'close',
        '[[HIGH_PRICE]]': 'high',
        '[[LOW_PRICE]]': 'low'
    }
    if folder == 'extrac':
        df = pd.read_csv(PATH_DATA_IS.format(symbol))
    else:
        df = pd.read_csv(PATH_DATA_OS.format(symbol))
    
    data = {
        'time': df['time'][pos:].to_numpy(dtype='datetime64[ns]'),
    }    
    
    high = df['high'].to_numpy(dtype=np.float64)
    low = df['low'].to_numpy(dtype=np.float64)
    close = df['close'].to_numpy(dtype=np.float64)
    
    for line in raw_lines:
        line = line.strip().split(';')
        indicator_name = line[0]
        
        if indicator_name == 'ADX':
            timeperiod = int(line[1].split(',')[-1])
            result = talib.ADX(high, low, close, timeperiod=timeperiod)[pos:]
            data[f'ADX_{timeperiod}'] = result
            
        elif indicator_name == 'ADXR':
            timeperiod = int(line[1].split(',')[-1])
            result = talib.ADXR(high, low, close, timeperiod=timeperiod)[pos:]
            data[f'ADXR_{timeperiod}'] = result
            
        elif indicator_name == 'APO':
            params = line[1].split(',')
            colum = price_column_map[params[0]]
            real = df[colum].to_numpy(dtype=np.float64)
            fast = int(params[1])
            slow = int(params[2])
            value3 = params[3].replace('[', '').replace(']', '').strip()
            matype = replaceString[value3]
            result = talib.APO(real, fast, slow, matype)[pos:]
            data[f'APO_{fast}_{slow}_{value3}'] = result

        elif indicator_name == 'AROON':
            params = line[1].split(',')
            timeperiod = int(params[2])
            pos_output = 0
            if params[3] == '[[OUTPUT2]]':
                pos_output = 1
            result = talib.AROON(high, low, timeperiod=timeperiod)[pos_output][pos:]
            data[f'AROON_{timeperiod}_pos{pos_output}'] = result

        elif indicator_name == 'ATR':
            timeperiod = int(line[1].split(',')[-1])
            result = talib.ATR(high, low, close, timeperiod=timeperiod)[pos:]
            data[f'ATR_{timeperiod}'] = result
            
        elif indicator_name == 'BOP':
            colum_open = df['open'].to_numpy(dtype=np.float64)
            result = talib.BOP(colum_open, high, low, close)[pos:]
            data['BOP'] = result
            
        elif indicator_name == 'CCI':
            timeperiod = int(line[1].split(',')[-1])
            result = talib.CCI(high, low, close, timeperiod=timeperiod)[pos:]
            data[f'CCI_{timeperiod}'] = result
            
        elif indicator_name == 'CMO':
            colum = price_column_map[line[1].split(',')[0]]
            real = df[colum].to_numpy(dtype=np.float64)
            timeperiod = int(line[1].split(',')[-1])
            result = talib.CMO(real, timeperiod=timeperiod)[pos:]
            data[f'CMO_{timeperiod}'] = result
        
        elif indicator_name == 'DX':
            timeperiod = int(line[1].split(',')[-1])
            result = talib.DX(high, low, close, timeperiod=timeperiod)[pos:]
            data[f'DX_{timeperiod}'] = result
            
        elif indicator_name == 'MACD':
            params = line[1].split(',')
            colum = price_column_map[params[0]]
            real = df[colum].to_numpy(dtype=np.float64)
            fastperiod = int(params[1])
            slowperiod = int(params[2])
            signalperiod = int(params[3])
            result_pos = 0
            if params[4] == '[[OUTPUT2]]':
                result_pos= 1
            elif params[4] == '[[OUTPUT3]]':
                result_pos = 2
            result = talib.MACD(real, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)[result_pos][pos:]
            data[f'MACD_{fastperiod}_{slowperiod}_{signalperiod}_pos{result_pos}'] = result
            
        elif indicator_name == 'MINUS_DI':
            timeperiod = int(line[1].split(',')[-1])
            result = talib.MINUS_DI(high, low, close, timeperiod=timeperiod)[pos:]
            data[f'MINUS_DI_{timeperiod}'] = result
            
        elif indicator_name == 'MINUS_DM':
            timeperiod = int(line[1].split(',')[-1])
            result = talib.MINUS_DM(high, low, timeperiod=timeperiod)[pos:]
            data[f'MINUS_DM_{timeperiod}'] = result
            
        elif indicator_name == 'MOM':
            colum = price_column_map[line[1].split(',')[0]]
            real = df[colum].to_numpy(dtype=np.float64)
            timeperiod = int(line[1].split(',')[-1])
            result = talib.MOM(real, timeperiod=timeperiod)[pos:]
            data[f'MOM_{timeperiod}'] = result
            
        elif indicator_name == 'NATR':
            timeperiod = int(line[1].split(',')[-1])
            result = talib.NATR(high, low, close, timeperiod=timeperiod)[pos:]
            data[f'NATR_{timeperiod}'] = result
        
        elif indicator_name == 'PLUS_DI':
            timeperiod = int(line[1].split(',')[-1])
            result = talib.PLUS_DI(high, low, close, timeperiod=timeperiod)[pos:]
            data[f'PLUS_DI_{timeperiod}'] = result
        
        elif indicator_name == 'PLUS_DM':
            timeperiod = int(line[1].split(',')[-1])
            result = talib.PLUS_DM(high, low, timeperiod=timeperiod)[pos:]
            data[f'PLUS_DM_{timeperiod}'] = result
            
        elif indicator_name == 'PPO':
            params = line[1].split(',')
            colum = price_column_map[params[0]]
            real = df[colum].to_numpy(dtype=np.float64)
            fastperiod = int(params[1])
            slowperiod = int(params[2])
            value3 = params[3].replace('[', '').replace(']', '').strip()
            matype = replaceString[value3]
            result = talib.PPO(real, fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)[pos:]
            data[f'PPO_{fastperiod}_{slowperiod}_{value3}'] = result
            
        elif indicator_name == 'ROC':
            params = line[1].split(',')
            colum = price_column_map[params[0]]
            real = df[colum].to_numpy(dtype=np.float64)
            timeperiod = int(params[1])
            result = talib.ROC(real, timeperiod=timeperiod)[pos:]
            data[f'ROC_{timeperiod}'] = result
            
        elif indicator_name == 'RSI':
            colum = price_column_map[line[1].split(',')[0]]
            real = df[colum].to_numpy(dtype=np.float64)
            timeperiod = int(line[1].split(',')[-1])
            result = talib.RSI(real, timeperiod=timeperiod)[pos:]
            data[f'RSI_{timeperiod}'] = result
         
        elif indicator_name == 'STDDEV':
            colum = price_column_map[line[1].split(',')[0]]
            real = df[colum].to_numpy(dtype=np.float64)
            timeperiod = int(line[1].split(',')[1])
            nbdev = float(line[1].split(',')[2])
            result = talib.STDDEV(real, timeperiod=timeperiod, nbdev=nbdev)[pos:]
            data[f"STDDEV_{timeperiod}_{str(nbdev).replace('.', '')}"] = result
            
        elif indicator_name == 'STOCHF':
            params = line[1].split(',')
            fastk_period = int(params[3])
            fastd_period = int(params[4])
            matype = params[5].replace('[', '').replace(']', '').strip()
            fastd_matype = replaceString[matype]
            result_pos = 0
            if params[6] == '[[OUTPUT2]]':
                result_pos = 1
            result = talib.STOCHF(high, low, close, fastk_period=fastk_period, fastd_period=fastd_period, fastd_matype=fastd_matype)[result_pos][pos:]
            data[f'STOCHF_{fastk_period}_{fastd_period}_{matype}_pos{result_pos}'] = result
        
        elif indicator_name == 'STOCH':
            params = line[1].split(',')
            fastk_period = int(params[3])
            slowk_period = int(params[4])
            matype1 = params[5].replace('[', '').replace(']', '').strip()
            slowk_matype = replaceString[matype1]
            slowd_period = int(params[6])
            matype2 = params[7].replace('[', '').replace(']', '').strip()
            slowd_matype = replaceString[matype2]
            result = talib.STOCH(high, low, close, fastk_period=fastk_period, slowk_period=slowk_period, slowk_matype=slowk_matype, slowd_period=slowd_period, slowd_matype=slowd_matype)[0][pos:]
            data[f'STOCH_{fastk_period}_{slowk_period}_{matype1}_{slowd_period}_{matype2}_pos{0}'] = result
            result2 = talib.STOCH(high, low, close, fastk_period=fastk_period, slowk_period=slowk_period, slowk_matype=slowk_matype, slowd_period=slowd_period, slowd_matype=slowd_matype)[1][pos:]
            data[f'STOCH_{fastk_period}_{slowk_period}_{matype1}_{slowd_period}_{matype2}_pos{1}'] = result2
            
        elif indicator_name == 'STOCHRSI':
            params = line[1].split(',')
            colum = price_column_map[params[0]]
            real = df[colum].to_numpy(dtype=np.float64)
            timeperiod = int(params[1])
            fastk_period = int(params[2])
            fastd_period = int(params[3])
            matype = params[4].replace('[', '').replace(']', '').strip()
            fastd_matype = replaceString[matype]
            result_pos = 0
            if params[5] == '[[OUTPUT2]]':
                result_pos = 1
            result = talib.STOCHRSI(real, timeperiod=timeperiod, fastk_period=fastk_period, fastd_period=fastd_period, fastd_matype=fastd_matype)[result_pos][pos:]
            data[f'STOCHRSI_{timeperiod}_{fastk_period}_{fastd_period}_{matype}_pos{result_pos}'] = result

        elif indicator_name == 'TRANGE':
            result = talib.TRANGE(high, low, close)[pos:]
            data['TRANGE'] = result
            
        elif indicator_name == 'ULTOSC':
            timeperiod1 = int(line[1].split(',')[3])
            timeperiod2 = int(line[1].split(',')[4])
            timeperiod3 = int(line[1].split(',')[5])
            result = talib.ULTOSC(high, low, close, timeperiod1=timeperiod1, timeperiod2=timeperiod2, timeperiod3=timeperiod3)[pos:]
            data[f'ULTOSC_{timeperiod1}_{timeperiod2}_{timeperiod3}'] = result
            
        elif indicator_name == 'VAR':
            colum = price_column_map[line[1].split(',')[0]]
            real = df[colum].to_numpy(dtype=np.float64)
            timeperiod = int(line[1].split(',')[1])
            nbdev = float(line[1].split(',')[2])
            result = talib.VAR(real, timeperiod=timeperiod, nbdev=nbdev)[pos:]
            data[f"VAR_{timeperiod}_{str(nbdev).replace('.','')}"] = result
            
        elif indicator_name == 'WILLR':
            timeperiod = int(line[1].split(',')[-1])
            result = talib.WILLR(high, low, close, timeperiod=timeperiod)[pos:]
            data[f"WILLR_{timeperiod}"] = result

    data['open'] = df['open'][pos:].to_numpy(dtype=np.float64)
    data['close'] = df['close'][pos:].to_numpy(dtype=np.float64)
    output_df = pd.DataFrame(data)

    output_df['label'] = np.where(
        output_df['close'] > output_df['open'], 'UP',
        np.where(output_df['close'] < output_df['open'], 'DOWN', None)
    )
    # Eliminar filas donde label es None (close == open)
    output_df = output_df[output_df['label'].notna()]
    # Eliminar columnas 'open' y 'close'
    output_df = output_df.drop(columns=['open', 'close'])
    # (Opcional) Resetear el índice si lo deseas
    output_df = output_df.reset_index(drop=True)
    
    start_clean = start.replace("-", "")
    end_clean = end.replace("-", "")

    filename = indicator_file.replace(
        ".csv",
        f"_{symbol}_{start_clean}_{end_clean}_timeframe{timeframe}.parquet"
    )

    output_df.to_parquet(
        f"{PATH_BASE.format(symbol)}/{folder}/{filename}",
        index=False,
        engine="pyarrow",
        compression="snappy"# recomendado
)

       
def create_files(symbol, timeframe, str_start, end, indicators_files, folder, max_workers):
    
    crear_carpeta_si_no_existe(f'{PATH_BASE.format(symbol)}/is_os')
    crear_carpeta_si_no_existe(f'{PATH_BASE.format(symbol)}/extrac')
    crear_carpeta_si_no_existe(f'{PATH_BASE.format(symbol)}/extrac_os')
    peticiones.initialize_mt5()
    df = _get_data_mt5(symbol, timeframe, str_start, end, folder)
    start = datetime.strptime(str_start, '%Y-%m-%d')
    pos, _ = _buscar_fecha_o_siguiente(df, start)
    
    proces = min(max_workers//2, len(indicators_files))
    with concurrent.futures.ProcessPoolExecutor(max_workers=proces) as executor:
        # Diccionario para hacer seguimiento de los futuros
        futures = {}
        
        # Enviamos todos los trabajos al executor
        for i in range(proces):
            indice = i % len(indicators_files)
            file = indicators_files[indice]
            future = executor.submit(
                _generate_files, 
                file, pos, symbol, str_start, end, timeframe, folder
            )
            futures[future] = file
        
        # Esperamos a que todos los futuros completen
        for future in concurrent.futures.as_completed(futures):
            file = futures[future]
            try:
                # Obtenemos el resultado (por si hay retorno)
                future.result()
            except Exception as e:
                print(f"Error al procesar el archivo {file}: {e}")       


if __name__ == "__main__":
    with open(f'config/general_config.json', 'r', encoding='utf8') as file:
        config = json.load(file)
    for symbol in config['list_principal_symbols']:
        limpiar_carpeta(PATH_BASE.format(symbol) + '/extrac')
        create_files(
            symbol= symbol, 
            timeframe=config['timeframe'], 
            str_start=config['dateStart'], 
            end=config['dateEnd'], 
            indicators_files=config['indicators_files'], 
            folder='extrac'
            )

