import zmq
from datetime import datetime
import pandas as pd
import os
import sys
import operator
import json
import ast
import shutil



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
 
from src.routes import peticiones
from src.db.query import get_nodes_by_label
from src.neuronal.entrenar import load_trained_model, load_data, predict_from_inputs, BinaryNN
from src.neuronal.backtester import Backtester
from src.neuronal.data_para_entrenar import data_for_neuronal
from src.neuronal.generar_indicadores_for_neuro_test import generate_files


dict_files = {}

operadores = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne
}




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

list_files_name = os.listdir('output/extrac/')
list_files_name = [f"{file.split('_')[0]}.csv" for file in list_files_name ]

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
        old_path = f'output/test_neuronal/{symbol}/{symbol}_data_test.parquet'
        if os.path.exists(old_path):
            old_data = pd.read_parquet(old_path)
            df_data = pd.concat([old_data, df_data], ignore_index=True).drop(index=0)

    # Guardar datos actualizados
    df_data.to_parquet(
        f'output/test_neuronal/{symbol}/{symbol}_data_test.parquet',
        index=False
    )

    # 🔁 Generación secuencial de indicadores 
    for file in list_files_name:
        try:
            generar_indicador(file, df_data, symbol)
        except Exception as e:
            print(f"❌ [{symbol}] Error procesando {file}: {e}")

    return symbol

def limpiar_carpeta(ruta_carpeta):
    if os.path.exists(ruta_carpeta):
        for nombre in os.listdir(ruta_carpeta):
            ruta_elemento = os.path.join(ruta_carpeta, nombre)
            if os.path.isfile(ruta_elemento) or os.path.islink(ruta_elemento):
                os.unlink(ruta_elemento)  # elimina archivo o enlace simbólico
            elif os.path.isdir(ruta_elemento):
                shutil.rmtree(ruta_elemento)  # elimina carpeta recursivamente
        print(f"Contenido de la carpeta '{ruta_carpeta}' eliminado correctamente.")
    else:
        print(f"La carpeta '{ruta_carpeta}' no existe.")

def actualizar_dict(principal, nuevo_dict):
    """Actualiza dict_pips_best con nuevos datos."""
    for k, v in nuevo_dict.items():
        if k not in principal:
            principal[k] = v  # Guardar valor real
        else:
            # Promedio ponderado (como en TradingEngine.record_trade)
            principal[k] = (
                principal[k] * 0.9 + v * 0.1
            )
    
    return principal 

#--------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    
    limpiar_carpeta('output/test_neuronal/')
    path_for_neuronal = f'src/neuronal/data/data_for_neuronal_{algorithm}_{principal_symbol}.csv'
    X, Y = load_data(path_for_neuronal)
    nn = load_trained_model(
        "src/neuronal/data/model_trained.json",
        input_dim=X.shape[1]
    )
    
    peticiones.initialize_mt5()
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://0.0.0.0:5555")

    print("Servidor Python listo...")
    cont = 0
    moved_to_be = False

    with open('src/neuronal/data/maping_open.json', 'r') as file:
        maping_open = json.load(file)

    with open('src/neuronal/data/maping_close.json', 'r') as file:
        maping_close = json.load(file)
        
    with open('src/neuronal/data/best_score.json', 'r') as file:
        best_score = json.load(file)
    
    dict_pips_best = best_score['dict_pips_best']
    dict_pips = {}
    neuro_evaluation = False
    numero_iteracion_for_evalution = best_score['mas_perdidas_seguidas'] * 4  
    cont_iteracion_for_evalution = 0
    perdidas_seguidas = 0  
    nodo_open = ''
    nodo_close = ''

    while True:
        cont += 1 
        message = socket.recv_string()
        print("Recibido de MT5:", message)

        if message == "FIN":
            print("EA finalizó. Cerrando servidor.")
            break

        date = message.split(",")[0]
        open_price = float(message.split(",")[1])
        dict_files.clear()
        # ------------------------------------------------------
        # 🔁 PROCESAMIENTO SECUENCIAL DE SÍMBOLOS
        # ------------------------------------------------------
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
                cerrar = False
                if cumple_condiciones(row, nodo["conditions"]):
                    nodo_close = maping_close[nodo["key"]]
                    clase, prob = predict_from_inputs(nn, nodo_open, nodo_close)
                    
                    if algorithm == 'UP':
                        trade_pips = open_price - open_price_open
                    else:
                        trade_pips = open_price_open - open_price
                        
                    key = f'{nodo_open}_{nodo_close}'                    
                    if key not in dict_pips:
                            dict_pips[key] = trade_pips
                    else:
                        dict_pips[key] = (
                            dict_pips[key] * 0.9 + trade_pips * 0.1
                        )
                        
                    if clase == 1:  # Si la red predice que se debe cerrar
                        cerrar = True
                        
                if cerrar:                   
                    if algorithm == 'UP':
                        trade_pips = open_price - open_price_open
                    else:
                        trade_pips = open_price_open - open_price
                        
                    if trade_pips < 0:
                        perdidas_seguidas += 1
                    else:
                        perdidas_seguidas = 0
                    
                    if neuro_evaluation:
                        cont_iteracion_for_evalution += 1
                        if cont_iteracion_for_evalution >= numero_iteracion_for_evalution:
                            neuro_evaluation = False
                            cont_iteracion_for_evalution = 0
                            print(f"Finalizó fase de evaluación neuronal después de {numero_iteracion_for_evalution} iteraciones.")
                            
                    order = 'CLOSE'
                    is_open = False 
                    break   
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
                            nodo_open = maping_open[nodo["key"]]
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

        
        if perdidas_seguidas > best_score['mas_perdidas_seguidas'] and neuro_evaluation:
            date_red_neuronal = date.split()[0].replace('.', '-')
            neuro_evaluation = False
            backtester = Backtester(date_red_neuronal)
            backtester.run()
            
        if perdidas_seguidas >= best_score['mas_perdidas_seguidas']:
            interv = int(len(dict_pips)/8)
            top = dict(
                sorted(dict_pips.items(), key=lambda x: x[1], reverse=True)[:interv]
            )

            bottom = dict(
                sorted(dict_pips.items(), key=lambda x: x[1])[:interv]
            )
            dict_pips_best = actualizar_dict(dict_pips_best, top)
            dict_pips_best = actualizar_dict(dict_pips_best, bottom)
            
            perdidas_seguidas = 0
            data_for_neuronal(algorithm, principal_symbol, dict_pips_best)
            X, Y = load_data(path_for_neuronal)
            print("Datos cargados:")
            print("X shape:", X.shape)
            print("Y shape:", Y.shape)
            nn = BinaryNN(input_dim=X.shape[1], lr=0.01, target_loss=0.10)
            nn.fit(X, Y, epochs=20000, batch_size=32)
            # Guardar modelo entrenado
            model_data = {
                'W1': nn.W1.tolist(),
                'b1': nn.b1.tolist(),
                'W2': nn.W2.tolist(),
                'b2': nn.b2.tolist(),
                'W3': nn.W3.tolist(),
                'b3': nn.b3.tolist(),
                'W4': nn.W4.tolist(),
                'b4': nn.b4.tolist()
            }
            with open('src/neuronal/data/model_trained.json', 'w') as f:
                json.dump(model_data, f, indent=4)
            print("Modelo entrenado guardado en 'src/neuronal/data/model_trained.json'")
            nn = load_trained_model(
                "src/neuronal/data/model_trained.json",
                input_dim=X.shape[1]
            )
            neuro_evaluation = True
            
            
        print("→ Enviando orden:", order)
        socket.send_string(order)
        
            