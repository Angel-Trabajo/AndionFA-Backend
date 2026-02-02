import zmq
from datetime import datetime
import pandas as pd
import os
import sys
import operator
import json
import ast



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
 
from src.routes import peticiones
import numpy as np
import talib
from src.db.query import get_nodes_by_label
from src.neuronal.entrenar import load_trained_model, predict_from_inputs

dict_files = {}

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


operadores = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne
}

nn = load_trained_model("src/neuronal/data/nn_binary_best.json", input_dim=18)


def obtener_ultimas_velas(symbol: str, fecha_str: str, timeframe: str, cont):
    """
    Obtiene los últimos 300 datos OHLCV (open, high, low, close, volume)
    antes de la fecha dada usando MetaTrader5.

    Parámetros:
        symbol (str): símbolo del activo, ej. "EURUSD"
        fecha_str (str): fecha en formato "YYYY.MM.DD HH:MM"
        timeframe (str): tamaño de vela ("M1", "M5", "H1", "D1", etc.)

    Retorna:
        pandas.DataFrame con las columnas [time, open, high, low, close, tick_volume]
    """

    
    # --- Conversión de fecha ---
    fecha_final = datetime.strptime(fecha_str, "%Y.%m.%d %H:%M")

    # --- Conversión de timeframe a constante de MT5 ---
    timeframes = peticiones.get_timeframes()

    if timeframe not in timeframes:
        raise ValueError(f"⛔ Timeframe no válido: {timeframe}")

    tf = timeframes[timeframe]

    # --- Descargar las últimas 830 velas antes de la fecha final ---
    cantidad = 1
    if cont == 1:
        cantidad = 830
        
    rates = peticiones.get_data_by_days(symbol, tf, fecha_final, cantidad)
    if rates is None or len(rates) == 0:
        raise ValueError("⚠️ No se obtuvieron datos del terminal MT5")
    
    # --- Convertir a DataFrame ---
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]
    # Cerrar conexión
    return df



def cumple_condiciones(fila, condiciones):
    for col, op, valor in condiciones:
        if fila[col] is None:
            return False
        if not operadores[op](fila[col], valor):
            return False
    return True


with open('config/config_test/config_test_red.json', 'r') as file:
    config = json.load(file)
algorithm = config['algorithm']

list_files_name = os.listdir('output/data_arff/')
list_files_name = [f'{file.split('_')[0]}.csv' for file in list_files_name ]

with open(f'config/list_{algorithm}.json', 'r') as file:
    config_extractor = json.load(file)
with open('config/config_crossing/config_crossing.json', 'r') as file:
    config_crossing = json.load(file)
list_symbols = config_extractor['list'] 
principal_symbol = config_crossing['principal_symbol']
list_symbols.insert(0, principal_symbol)

with open('config/config_crossing/config_crossing.json', 'r') as file:
    config_crossing = json.load(file)
    timeframe = config_crossing['timeframe']

dict_nodos = {}
for i, symbol in enumerate(list_symbols):
    if i == 0:
        dict_nodos[symbol] = get_nodes_by_label(symbol, algorithm)
    else:
        dict_nodos[symbol] = get_nodes_by_label(f'crossing_{principal_symbol}_dbs/{symbol}', algorithm)

def parsear_nodos(dict_nodos):
    nodos_parseados = {}

    for symbol, nodos in dict_nodos.items():
        lista = []
        for nodo in nodos:
            condiciones_str = nodo[0]
            file_name = nodo[1]

            lista.append({
                "key": condiciones_str,                    # 🔑 CLAVE
                "conditions": ast.literal_eval(condiciones_str),
                "file": f"{file_name.split('_')[0]}.csv"
            })

        nodos_parseados[symbol] = lista

    return nodos_parseados


dict_nodos = parsear_nodos(dict_nodos)


if algorithm == 'UP':
    other_algorithm = 'DOWN'
else:
    other_algorithm = 'UP'
nodos_close = get_nodes_by_label(principal_symbol, other_algorithm)

nodos_close = [
    {
        "key": n[0],
        "conditions": ast.literal_eval(n[0]),
        "file": f"{n[1].split('_')[0]}.csv"
    }
    for n in nodos_close
]

entry_red = []
order = 'NONE'
is_open = False
open_price_open = 0


def generar_indicador(file, df_data, symbol):
    """Genera y guarda un indicador."""
    df_indicators = generate_files(file, df_data)
    dict_files[f'{symbol}_{file}'] = df_indicators


def procesar_symbol(symbol, date, list_files_name, cont):
    os.makedirs(f'output/test_neuronal/{symbol}', exist_ok=True)

   
    # Obtener datos
    df_data = obtener_ultimas_velas(symbol, date, timeframe, cont)

    # Concatenar histórico si aplica
    if cont > 1:
        old_path = f'output/test_neuronal/{symbol}/{symbol}_data_test.csv'
        if os.path.exists(old_path):
            old_data = pd.read_csv(old_path)
            df_data = pd.concat([old_data, df_data], ignore_index=True).drop(index=0)

    # Guardar datos actualizados
    df_data.to_csv(
        f'output/test_neuronal/{symbol}/{symbol}_data_test.csv',
        index=False
    )

    # 🔁 Generación secuencial de indicadores 
    for file in list_files_name:
        try:
            generar_indicador(file, df_data, symbol)
        except Exception as e:
            print(f"❌ [{symbol}] Error procesando {file}: {e}")

    return symbol



#--------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    peticiones.initialize_mt5()
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://0.0.0.0:5555")

    print("Servidor Python listo...")
    cont = 0
    with open('src/neuronal/data/maping_close.json', 'r') as file:
        encoding_actions_close = json.load(file)   
    with open('src/neuronal/data/maping_open.json', 'r') as file:
        encoding_actions = json.load(file)
     
    while True:
        cont += 1
        dict_files.clear() 
        message = socket.recv_string()
        print("Recibido de MT5:", message)

        if message == "FIN":
            print("EA finalizó. Cerrando servidor.")
            break

        date = message.split(",")[0]
        open_price = float(message.split(",")[1])

        # ------------------------------------------------------
        # 🔁 PROCESAMIENTO SECUENCIAL DE SÍMBOLOS
        # ------------------------------------------------------
        if cont == 1:
            for symbol in list_symbols:
                try:
                    procesar_symbol(symbol, date, list_files_name, cont)#
                except Exception as e:
                    print(f"❌ Error procesando {symbol}: {e}")
               
        if is_open:
            for i, nodo in enumerate(nodos_close):
                file = nodo["file"]
                df = dict_files[f'{principal_symbol}_{file}']
                row = df.iloc[-1]
                if cumple_condiciones(row, nodo["conditions"]):
                    entry_red_close = encoding_actions_close[nodo["key"]]
                    entry_red = tuple(entry_red)
                    entry_red_close = tuple(entry_red_close)
                    clase, valor = predict_from_inputs(nn, entry_red, entry_red_close)
                    
                    if (clase == 1) or (i > 10 and open_price > open_price_open):
                        order = 'CLOSE'
                        is_open = False
                        break
                
                    else:
                        order = 'NONE'
                else:
                    order = 'NONE'
        else:
            state = 1  # 1 = todos cumplen, 0 = alguno falla

            for symbol in list_symbols:
                nodos = dict_nodos[symbol]
                cumple_alguno = False  # bandera para saber si este símbolo cumple al menos un nodo

                for nodo in nodos:
                    file = f'{nodo["file"]}'

                    try:
                        df = dict_files[f'{symbol}_{file}']
                    except KeyError:
                        print(f"⚠️ df no encontrado: {symbol}_{file}")
                        state = 0
                        break  # rompe este símbolo
                    
                    if cont == 1:
                        row = df.iloc[-2]
                    else:
                        row = df.iloc[-1]
                    if cumple_condiciones(row, nodo["conditions"]):
                        cumple_alguno = True

                        # si es el último símbolo, cargamos la red
                        if symbol == list_symbols[-1]:
                            
                            entry_red = encoding_actions[nodo["key"]]
                            open_price_open = open_price
                            is_open = True

                        # no hace falta seguir revisando nodos de este símbolo
                        break

                # si este símbolo no cumplió ningún nodo, se detiene todo
                if not cumple_alguno:
                    state = 0
                    print(f"❌ El símbolo {symbol} no cumple ningún nodo. Se detiene la búsqueda.")
                    break

            # ---------------------------
            # Decisión final de orden
            # ---------------------------
            if state == 1:
                order = 'BUY' if algorithm == 'UP' else 'SELL'
            else:
                order = 'NONE'

        print("→ Enviando orden:", order)

        socket.send_string(order)
        if cont > 1:
            for symbol in list_symbols:
                try:
                    procesar_symbol(symbol, date, list_files_name, cont)#
                except Exception as e:
                    print(f"❌ Error procesando {symbol}: {e}")