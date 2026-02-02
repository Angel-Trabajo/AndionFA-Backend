import zmq
import ast
from datetime import datetime, time, timedelta
import pandas as pd 
import numpy as np
import operator
import talib
import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import MetaTrader5 as mt5
from src.db.query import get_node_by_id, get_dates_by_label

operadores = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne
}

orden = {
    "DOWN": 'SELL',
    "UP"  : 'BUY',
    "UPDOWN": 'BUYSELL'
}

TIMEFRAMES = {
    "timeframeH1": mt5.TIMEFRAME_H1,
    "timeframeH4": mt5.TIMEFRAME_H4,
    "timeframeD1": mt5.TIMEFRAME_D1,
    "timeframeW1": mt5.TIMEFRAME_W1,
    "timeframeMN1": mt5.TIMEFRAME_MN1,
}

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


def obtener_historicos_ultimo_anio(fecha_dt: datetime, symbol: str, timeframe_str: str) -> pd.DataFrame:
    """
    Obtiene datos históricos desde un año antes de la fecha proporcionada hasta la fecha exacta,
    ordenados desde la fecha más reciente hacia atrás.
    """
    if timeframe_str not in TIMEFRAMES:
        raise ValueError(f"❌ Timeframe '{timeframe_str}' no reconocido.")

    timeframe = TIMEFRAMES[timeframe_str]
    desde = fecha_dt - timedelta(days=730)
    
    final = fecha_dt + timedelta(days=1500)

    if not mt5.initialize():
        raise RuntimeError(f"❌ Error al inicializar MetaTrader5: {mt5.last_error()}")

    rates = mt5.copy_rates_range(symbol, timeframe, desde, final)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise ValueError(f"⚠️ No se encontraron datos para {symbol} desde {desde} hasta {fecha_dt}")

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


def generate_files(indicator_file, df):
    with open(f'config/extractor/{indicator_file}', 'r') as f:
        raw_lines = f.readlines()
    
    price_column_map = {
        '[[CLOSE_PRICE]]': 'close',
        '[[HIGH_PRICE]]': 'high',
        '[[LOW_PRICE]]': 'low'
    }
    
    data = {
        'time': df['time'].to_numpy(dtype='datetime64[ns]'),
    }    
    
    high = df['high'].to_numpy(dtype=np.float64)
    low = df['low'].to_numpy(dtype=np.float64)
    close = df['close'].to_numpy(dtype=np.float64)
    
    for line in raw_lines:
        line = line.strip().split(';')
        indicator_name = line[0]
        
        if indicator_name == 'ADX':
            timeperiod = int(line[1].split(',')[-1])
            result = talib.ADX(high, low, close, timeperiod=timeperiod)
            data[f'ADX_{timeperiod}'] = result
            
        elif indicator_name == 'ADXR':
            timeperiod = int(line[1].split(',')[-1])
            result = talib.ADXR(high, low, close, timeperiod=timeperiod)
            data[f'ADXR_{timeperiod}'] = result
            
        elif indicator_name == 'APO':
            params = line[1].split(',')
            colum = price_column_map[params[0]]
            real = df[colum].to_numpy(dtype=np.float64)
            fast = int(params[1])
            slow = int(params[2])
            value3 = params[3].replace('[', '').replace(']', '').strip()
            matype = replaceString[value3]
            result = talib.APO(real, fast, slow, matype)
            data[f'APO_{fast}_{slow}_{value3}'] = result

        elif indicator_name == 'AROON':
            params = line[1].split(',')
            timeperiod = int(params[2])
            pos_output = 0
            if params[3] == '[[OUTPUT2]]':
                pos_output = 1
            result = talib.AROON(high, low, timeperiod=timeperiod)[pos_output]
            data[f'AROON_{timeperiod}_pos{pos_output}'] = result

        elif indicator_name == 'ATR':
            timeperiod = int(line[1].split(',')[-1])
            result = talib.ATR(high, low, close, timeperiod=timeperiod)
            data[f'ATR_{timeperiod}'] = result
            
        elif indicator_name == 'BOP':
            colum_open = df['open'].to_numpy(dtype=np.float64)
            result = talib.BOP(colum_open, high, low, close)
            data['BOP'] = result
            
        elif indicator_name == 'CCI':
            timeperiod = int(line[1].split(',')[-1])
            result = talib.CCI(high, low, close, timeperiod=timeperiod)
            data[f'CCI_{timeperiod}'] = result
            
        elif indicator_name == 'CMO':
            colum = price_column_map[line[1].split(',')[0]]
            real = df[colum].to_numpy(dtype=np.float64)
            timeperiod = int(line[1].split(',')[-1])
            result = talib.CMO(real, timeperiod=timeperiod)
            data[f'CMO_{timeperiod}'] = result
        
        elif indicator_name == 'DX':
            timeperiod = int(line[1].split(',')[-1])
            result = talib.DX(high, low, close, timeperiod=timeperiod)
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
            result = talib.MACD(real, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)[result_pos]
            data[f'MACD_{fastperiod}_{slowperiod}_{signalperiod}_pos{result_pos}'] = result
            
        elif indicator_name == 'MINUS_DI':
            timeperiod = int(line[1].split(',')[-1])
            result = talib.MINUS_DI(high, low, close, timeperiod=timeperiod)
            data[f'MINUS_DI_{timeperiod}'] = result
            
        elif indicator_name == 'MINUS_DM':
            timeperiod = int(line[1].split(',')[-1])
            result = talib.MINUS_DM(high, low, timeperiod=timeperiod)
            data[f'MINUS_DM_{timeperiod}'] = result
            
        elif indicator_name == 'MOM':
            colum = price_column_map[line[1].split(',')[0]]
            real = df[colum].to_numpy(dtype=np.float64)
            timeperiod = int(line[1].split(',')[-1])
            result = talib.MOM(real, timeperiod=timeperiod)
            data[f'MOM_{timeperiod}'] = result
            
        elif indicator_name == 'NATR':
            timeperiod = int(line[1].split(',')[-1])
            result = talib.NATR(high, low, close, timeperiod=timeperiod)
            data[f'NATR_{timeperiod}'] = result
        
        elif indicator_name == 'PLUS_DI':
            timeperiod = int(line[1].split(',')[-1])
            result = talib.PLUS_DI(high, low, close, timeperiod=timeperiod)
            data[f'PLUS_DI_{timeperiod}'] = result
        
        elif indicator_name == 'PLUS_DM':
            timeperiod = int(line[1].split(',')[-1])
            result = talib.PLUS_DM(high, low, timeperiod=timeperiod)
            data[f'PLUS_DM_{timeperiod}'] = result
            
        elif indicator_name == 'PPO':
            params = line[1].split(',')
            colum = price_column_map[params[0]]
            real = df[colum].to_numpy(dtype=np.float64)
            fastperiod = int(params[1])
            slowperiod = int(params[2])
            value3 = params[3].replace('[', '').replace(']', '').strip()
            matype = replaceString[value3]
            result = talib.PPO(real, fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)
            data[f'PPO_{fastperiod}_{slowperiod}_{value3}'] = result
            
        elif indicator_name == 'ROC':
            params = line[1].split(',')
            colum = price_column_map[params[0]]
            real = df[colum].to_numpy(dtype=np.float64)
            timeperiod = int(params[1])
            result = talib.ROC(real, timeperiod=timeperiod)
            data[f'ROC_{timeperiod}'] = result
            
        elif indicator_name == 'RSI':
            colum = price_column_map[line[1].split(',')[0]]
            real = df[colum].to_numpy(dtype=np.float64)
            timeperiod = int(line[1].split(',')[-1])
            result = talib.RSI(real, timeperiod=timeperiod)
            data[f'RSI_{timeperiod}'] = result
         
        elif indicator_name == 'STDDEV':
            colum = price_column_map[line[1].split(',')[0]]
            real = df[colum].to_numpy(dtype=np.float64)
            timeperiod = int(line[1].split(',')[1])
            nbdev = float(line[1].split(',')[2])
            result = talib.STDDEV(real, timeperiod=timeperiod, nbdev=nbdev)
            data[f'STDDEV_{timeperiod}_{str(nbdev).replace('.', '')}'] = result
            
        elif indicator_name == 'STOCHF':
            params = line[1].split(',')
            fastk_period = int(params[3])
            fastd_period = int(params[4])
            matype = params[5].replace('[', '').replace(']', '').strip()
            fastd_matype = replaceString[matype]
            result_pos = 0
            if params[6] == '[[OUTPUT2]]':
                result_pos = 1
            result = talib.STOCHF(high, low, close, fastk_period=fastk_period, fastd_period=fastd_period, fastd_matype=fastd_matype)[result_pos]
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
            result = talib.STOCH(high, low, close, fastk_period=fastk_period, slowk_period=slowk_period, slowk_matype=slowk_matype, slowd_period=slowd_period, slowd_matype=slowd_matype)[0]
            data[f'STOCH_{fastk_period}_{slowk_period}_{matype1}_{slowd_period}_{matype2}_pos{0}'] = result
            result2 = talib.STOCH(high, low, close, fastk_period=fastk_period, slowk_period=slowk_period, slowk_matype=slowk_matype, slowd_period=slowd_period, slowd_matype=slowd_matype)[1]
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
            result = talib.STOCHRSI(real, timeperiod=timeperiod, fastk_period=fastk_period, fastd_period=fastd_period, fastd_matype=fastd_matype)[result_pos]
            data[f'STOCHRSI_{timeperiod}_{fastk_period}_{fastd_period}_{matype}_pos{result_pos}'] = result

        elif indicator_name == 'TRANGE':
            result = talib.TRANGE(high, low, close)
            data['TRANGE'] = result
            
        elif indicator_name == 'ULTOSC':
            timeperiod1 = int(line[1].split(',')[3])
            timeperiod2 = int(line[1].split(',')[4])
            timeperiod3 = int(line[1].split(',')[5])
            result = talib.ULTOSC(high, low, close, timeperiod1=timeperiod1, timeperiod2=timeperiod2, timeperiod3=timeperiod3)
            data[f'ULTOSC_{timeperiod1}_{timeperiod2}_{timeperiod3}'] = result
            
        elif indicator_name == 'VAR':
            colum = price_column_map[line[1].split(',')[0]]
            real = df[colum].to_numpy(dtype=np.float64)
            timeperiod = int(line[1].split(',')[1])
            nbdev = float(line[1].split(',')[2])
            result = talib.VAR(real, timeperiod=timeperiod, nbdev=nbdev)
            data[f'VAR_{timeperiod}_{str(nbdev).replace('.','')}'] = result
            
        elif indicator_name == 'WILLR':
            timeperiod = int(line[1].split(',')[-1])
            result = talib.WILLR(high, low, close, timeperiod=timeperiod)
            data[f'WILLR_{timeperiod}'] = result

    data['open'] = df['open'].to_numpy(dtype=np.float64)
    data['close'] = df['close'].to_numpy(dtype=np.float64)
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
    return output_df
   
   
def cumple_condiciones(fila, condiciones):
    return all(
        operadores[op](fila[col], valor)
        for col, op, valor in condiciones
        if col in fila
    )
    


def make_test(prin_symbol, symbol, list_id):
    """
    Función para ejecutar el script de prueba con un símbolo y un ID específicos.
    """
    list_nodes = []
    list_files = []
    list_timeframes = []
    list_dfs = []
    back_symbol = None
    list_dates_UP = []
    list_dates_DOWN = []


    for id in list_id:
        node = get_node_by_id(f'crossing_{prin_symbol}_dbs/{symbol}', id)
        list_nodes.append(node)
        file = node[2].split('_')[0]
        timeframe_str = node[2].split('_')[-1].replace('.csv', '')
        list_files.append(file)
        list_timeframes.append(timeframe_str)

    #Crear contexto y socket ZMQ tipo REP (respuesta)
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://127.0.0.1:5555")  # Escucha en el puerto 5555
    print("Servidor Python listo y esperando conexiones...")
    cont = 0
    try:
        while True:
            cont +=1
            # Espera mensaje del EA
            message = socket.recv_string()
            if message == "FIN":
                print(f"[EA] Vela recibida: {message}")
                print("EA finalizó. Cerrando servidor.")
                break
            fecha_dt = datetime.strptime(message, "%Y.%m.%d %H:%M")
            print(f"[EA] Vela recibida: {fecha_dt}")
            
            if cont == 1:
                for node in list_nodes:
                    with open(f'config/list_{node[1]}.json', encoding='utf8') as file:
                        data = json.load(file)
                    list_symbol = data['list']
                    list_symbol.insert(0, prin_symbol)
                    back_symbol= list_symbol[list_symbol.index(symbol)-1]
                    if node[1] == 'UP':
                        if prin_symbol == back_symbol:
                            list_oper_os = list(get_dates_by_label(back_symbol, node[1], 'os'))
                            list_oper_is = list(get_dates_by_label(back_symbol, node[1], 'is'))
                        else:
                            list_oper_os = list(get_dates_by_label(f'crossing_{prin_symbol}_dbs/{back_symbol}', node[1], 'os'))
                            list_oper_is = list(get_dates_by_label(f'crossing_{prin_symbol}_dbs/{back_symbol}', node[1], 'is'))
                        fechas_dt_os = pd.to_datetime(list_oper_os)
                        fechas_dt_is = pd.to_datetime(list_oper_is)
                        list_dates_UP = [*fechas_dt_os, *fechas_dt_is]
                    else:
                        if prin_symbol == back_symbol:
                            list_oper_os = list(get_dates_by_label(back_symbol, node[1], 'os'))
                            list_oper_is = list(get_dates_by_label(back_symbol, node[1], 'is'))
                        else:
                            list_oper_os = list(get_dates_by_label(f'crossing_{prin_symbol}_dbs/{back_symbol}', node[1], 'os'))
                            list_oper_is = list(get_dates_by_label(f'crossing_{prin_symbol}_dbs/{back_symbol}', node[1], 'is'))
                        fechas_dt_os = pd.to_datetime(list_oper_os)
                        fechas_dt_is = pd.to_datetime(list_oper_is)
                        list_dates_DOWN = [*fechas_dt_os, *fechas_dt_is]

                for file, timeframe_str in zip(list_files, list_timeframes):
                    df = obtener_historicos_ultimo_anio(fecha_dt, symbol, timeframe_str,)
                    df_indicadores = generate_files(f'{file}.csv', df) 
                    df_indicadores['time'] = pd.to_datetime(df_indicadores['time'])
                    list_dfs.append(df_indicadores)
                  
                        
            hora = fecha_dt.time()
            if hora == time(0, 0):
                print("[Python] Hora de cierre del día, enviando mensaje vacío.")
                socket.send_string('')
                continue
            fecha_dt = pd.to_datetime(fecha_dt)
            
            
            decision = ''
            for node, df_indicadores in zip(list_nodes, list_dfs):
                if node[1] == 'DOWN' and fecha_dt not in list_dates_DOWN:
                    continue
                if node[1] == 'UP' and fecha_dt not in list_dates_UP:
                    continue
                try:
                    index_row = df_indicadores[df_indicadores['time'] == fecha_dt].index[0]
                    indicador_row = df_indicadores.iloc[index_row-1]
                except Exception:
                    print(f"[Python] No se encontró la fecha {fecha_dt} en los datos de indicadores.")
                    break
                conditions = node[3]
                conditions = ast.literal_eval(conditions)
                if cumple_condiciones(indicador_row, conditions):
                    decision = orden[node[1]]
                    break
                    
            print(f"[Python] Enviando decisión: {decision}")
            socket.send_string(decision)

    except KeyboardInterrupt:
        print("Interrumpido por el usuario.")

    finally:
        socket.close()
        context.term()
        
        
if __name__ == "__main__":
    with open('config/config_test/config_test.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    make_test(data['prin_symbol'], data["symbol"], data["list_id"])
    