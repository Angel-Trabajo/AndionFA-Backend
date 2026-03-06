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
from src.utils.indicadores_for_principal_script import generate_files

operadores = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne
}

RUTA_CONFIG_TEST_RED = 'config/config_test/config_test_red.json'
PATH_FILES_NAME = 'output/extrac/'
PATH_LIST_SYMBOLS = 'config/list_{}.json'
PATH_CONFIG_CROSSING = 'config/config_crossing/config_crossing.json'
PATH_DATA_TEST = 'output/test_neuronal/'
PATH_DATA_FOR_NEURONAL = 'src/neuronal/data/data_for_neuronal_{}_{}.csv'
PATH_MODEL_TRAINED = "src/neuronal/data/model_trained.json"
PATH_MAPPING_OPEN = 'src/neuronal/data/maping_open.json'
PATH_MAPPING_CLOSE = 'src/neuronal/data/maping_close.json'
PATH_BEST_SCORE = 'src/neuronal/data/best_score.json'
PATH_CONT_EVOLUTION = 'src/neuronal/data/cont_evolution.json'
MULTIPLICADOR_OPERACIONES_PARA_EVALUACION = 4

def obtener_ultimas_velas(symbol: str, fecha_str: str, timeframe: str, cont):
    fecha_final = datetime.strptime(fecha_str, "%Y.%m.%d %H:%M")

    timeframes = peticiones.get_timeframes()

    if timeframe not in timeframes:
        raise ValueError(f"⛔ Timeframe no válido: {timeframe}")

    tf = timeframes[timeframe]

    cantidad = 1
    if cont == 1:
        cantidad = 830

    rates = peticiones.get_data_by_days(symbol, tf, fecha_final, cantidad)

    if rates is None or len(rates) == 0:
        raise ValueError("⚠️ No se obtuvieron datos del terminal MT5")

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]

    return df


def cumple_condiciones(fila, condiciones):
    for col, op, valor in condiciones:
        if fila[col] is None:
            return False
        if not operadores[op](fila[col], valor):
            return False
    return True


def limpiar_carpeta(ruta_carpeta):
    if os.path.exists(ruta_carpeta):
        for nombre in os.listdir(ruta_carpeta):
            ruta_elemento = os.path.join(ruta_carpeta, nombre)
            if os.path.isfile(ruta_elemento) or os.path.islink(ruta_elemento):
                os.unlink(ruta_elemento)
            elif os.path.isdir(ruta_elemento):
                shutil.rmtree(ruta_elemento)
        print(f"Contenido de la carpeta '{ruta_carpeta}' eliminado correctamente.")
    else:
        print(f"La carpeta '{ruta_carpeta}' no existe.")
        
        
def actualizar_dict(principal, nuevo_dict):
    for k, v in nuevo_dict.items():
        if k not in principal:
            principal[k] = v
        else:
            principal[k] = (
                principal[k] * 0.9 + v * 0.1
            )
    return principal


class TradingServer:

    def __init__(self):

        # =========================
        # CARGA CONFIGURACIÓN EXACTA
        # =========================

        with open(RUTA_CONFIG_TEST_RED, 'r') as file:
            config = json.load(file)

        self.algorithm = config['algorithm']

        list_files_name = os.listdir(PATH_FILES_NAME)
        self.list_files_name = [f"{file.split('_')[0]}.csv" for file in list_files_name]

        with open(PATH_LIST_SYMBOLS.format(self.algorithm), 'r') as file:
            config_extractor = json.load(file)

        with open(PATH_CONFIG_CROSSING, 'r') as file:
            config_crossing = json.load(file)

        self.list_symbols = config_extractor['list']
        self.principal_symbol = config_crossing['principal_symbol']
        self.list_symbols.insert(0, self.principal_symbol)

        with open(PATH_CONFIG_CROSSING, 'r') as file:
            config_crossing = json.load(file)
            self.timeframe = config_crossing['timeframe']

        # =========================
        # NODOS EXACTAMENTE IGUAL
        # =========================

        dict_nodos = {}

        for i, symbol in enumerate(self.list_symbols):
            if i == 0:
                dict_nodos[symbol] = get_nodes_by_label(symbol, self.algorithm)
            else:
                dict_nodos[symbol] = get_nodes_by_label(
                    f'crossing_{self.principal_symbol}_dbs/{symbol}',
                    self.algorithm
                )

        self.dict_nodos = self.parsear_nodos(dict_nodos)

        if self.algorithm == 'UP':
            other_algorithm = 'DOWN'
        else:
            other_algorithm = 'UP'

        nodos_close = get_nodes_by_label(self.principal_symbol, other_algorithm)

        self.nodos_close = [
            {
                "key": n[0],
                "conditions": ast.literal_eval(n[0]),
                "file": f"{n[1].split('_')[0]}.csv"
            }
            for n in nodos_close
        ]

        # =========================
        # ESTADO GLOBAL ORIGINAL
        # =========================

        self.dict_files = {}
        self.order = 'NONE'
        self.is_open = False
        self.open_price_open = 0
        self.cont = 0
        self.moved_to_be = False
        
        self.neuro_evaluation = False
        self.cont_iteracion_for_evalution = 0
        self.perdidas_seguidas = 0
        self.nodo_open = ''
        self.nodo_close = ''
        self.dict_pips = {}

    def parsear_nodos(self, dict_nodos):
        nodos_parseados = {}

        for symbol, nodos in dict_nodos.items():
            lista = []
            for nodo in nodos:
                condiciones_str = nodo[0]
                file_name = nodo[1]

                lista.append({
                    "key": condiciones_str,
                    "conditions": ast.literal_eval(condiciones_str),
                    "file": f"{file_name.split('_')[0]}.csv"
                })

            nodos_parseados[symbol] = lista

        return nodos_parseados
    
    
    def generar_indicador(self, file, df_data, symbol):
        df_indicators = generate_files(file, df_data)
        self.dict_files[f'{symbol}_{file}'] = df_indicators
        
    
    def procesar_symbol(self, symbol, date, cont):

        os.makedirs(f'output/test_neuronal/{symbol}', exist_ok=True)

        df_data = obtener_ultimas_velas(symbol, date, self.timeframe, cont)

        if cont > 1:
            old_path = f'output/test_neuronal/{symbol}/{symbol}_data_test.parquet'
            if os.path.exists(old_path):
                old_data = pd.read_parquet(old_path)
                df_data = pd.concat([old_data, df_data], ignore_index=True).drop(index=0)

        df_data.to_parquet(
            f'output/test_neuronal/{symbol}/{symbol}_data_test.parquet',
            index=False
        )

        for file in self.list_files_name:
            try:
                self.generar_indicador(file, df_data, symbol)
            except Exception as e:
                print(f"❌ [{symbol}] Error procesando {file}: {e}")

        return symbol
    
    
    def initialize_neuronal(self):

        limpiar_carpeta(PATH_DATA_TEST)
        
        with open(PATH_CONT_EVOLUTION, 'w') as file:
            json.dump({"trained_cont": 0, "neuronal_evolution_cont": 0}, file, indent=4)

        self.path_for_neuronal = (
            PATH_DATA_FOR_NEURONAL.format(self.algorithm, self.principal_symbol)
        )

        X, Y = load_data(self.path_for_neuronal)

        self.nn = load_trained_model(
            PATH_MODEL_TRAINED,
            input_dim=X.shape[1]
        )

        with open(PATH_MAPPING_OPEN, 'r') as file:
            self.maping_open = json.load(file)

        with open(PATH_MAPPING_CLOSE, 'r') as file:
            self.maping_close = json.load(file)

        with open(PATH_BEST_SCORE, 'r') as file:
            self.best_score = json.load(file)

        self.dict_pips_best = self.best_score['dict_pips_best']

        self.numero_iteracion_for_evalution = (
            self.best_score['mas_perdidas_seguidas'] * MULTIPLICADOR_OPERACIONES_PARA_EVALUACION
        )
        
        
    def handle_close_logic(self, open_price):

        for i, nodo in enumerate(self.nodos_close):

            file = nodo["file"]
            df = self.dict_files[f'{self.principal_symbol}_{file}']
            row = df.iloc[-1]

            cerrar = False

            if cumple_condiciones(row, nodo["conditions"]):

                self.nodo_close = self.maping_close[nodo["key"]]
                clase, prob = predict_from_inputs(
                    self.nn,
                    self.nodo_open,
                    self.nodo_close
                )

                if self.algorithm == 'UP':
                    trade_pips = open_price - self.open_price_open
                else:
                    trade_pips = self.open_price_open - open_price

                key = f'{self.nodo_open}_{self.nodo_close}'

                if key not in self.dict_pips:
                    self.dict_pips[key] = trade_pips
                else:
                    self.dict_pips[key] = (
                        self.dict_pips[key] * 0.9 + trade_pips * 0.1
                    )

                if clase == 1:
                    cerrar = True

            if cerrar:
                if trade_pips < 0:
                    self.perdidas_seguidas += 1
                else:
                    self.perdidas_seguidas = 0

                if self.neuro_evaluation:
                    self.cont_iteracion_for_evalution += 1
                    if (
                        self.cont_iteracion_for_evalution
                        >= self.numero_iteracion_for_evalution
                    ):
                        self.neuro_evaluation = False
                        self.cont_iteracion_for_evalution = 0
                        print(
                            f"Finalizó fase de evaluación neuronal después de "
                            f"{self.numero_iteracion_for_evalution} iteraciones."
                        )

                self.order = 'CLOSE'
                self.is_open = False
                break

            else:
                self.order = 'NONE'
                
                
    def handle_open_logic(self, open_price):

        state = 1

        for symbol in self.list_symbols:

            nodos = self.dict_nodos[symbol]
            cumple_alguno = False

            for nodo in nodos:

                file = f'{nodo["file"]}'

                try:
                    df = self.dict_files[f'{symbol}_{file}']
                except KeyError:
                    print(f"⚠️ df no encontrado: {symbol}_{file}")
                    state = 0
                    break

                if self.cont == 1:
                    row = df.iloc[-2]
                else:
                    row = df.iloc[-1]

                if cumple_condiciones(row, nodo["conditions"]):

                    cumple_alguno = True

                    if symbol == self.list_symbols[-1]:
                        self.nodo_open = self.maping_open[nodo["key"]]
                        self.open_price_open = open_price
                        self.is_open = True

                    break

            if not cumple_alguno:
                state = 0
                print(
                    f"❌ El símbolo {symbol} no cumple ningún nodo. "
                    f"Se detiene la búsqueda."
                )
                break

        if state == 1:
            self.order = 'BUY' if self.algorithm == 'UP' else 'SELL'
        else:
            self.order = 'NONE'
      
            
    def retrain_if_needed(self, date):

        if (
            self.perdidas_seguidas
            > self.best_score['mas_perdidas_seguidas']
            and self.neuro_evaluation
        ):

            date_red_neuronal = date.split()[0].replace('.', '-')
            self.neuro_evaluation = False

            backtester = Backtester(date_red_neuronal)
            backtester.run()
            
            
            with open(PATH_CONT_EVOLUTION, 'r') as file:
                cont_evolution = json.load(file)
            cont_evolution['neuronal_evolution_cont'] += 1
            with open(PATH_CONT_EVOLUTION, 'w') as file:
                json.dump(cont_evolution, file, indent=4)
                

        if (
            self.perdidas_seguidas
            >= self.best_score['mas_perdidas_seguidas']
        ):

            interv = int(len(self.dict_pips) / 8)

            top = dict(
                sorted(
                    self.dict_pips.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:interv]
            )

            bottom = dict(
                sorted(
                    self.dict_pips.items(),
                    key=lambda x: x[1]
                )[:interv]
            )

            self.dict_pips_best = actualizar_dict(
                self.dict_pips_best,
                top
            )

            self.dict_pips_best = actualizar_dict(
                self.dict_pips_best,
                bottom
            )

            self.perdidas_seguidas = 0

            data_for_neuronal(
                self.algorithm,
                self.principal_symbol,
                self.dict_pips_best
            )

            X, Y = load_data(self.path_for_neuronal)

            print("Datos cargados:")
            print("X shape:", X.shape)
            print("Y shape:", Y.shape)

            nn = BinaryNN(
                input_dim=X.shape[1],
                lr=0.01,
                target_loss=0.10
            )

            nn.fit(X, Y, epochs=20000, batch_size=32)

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

            with open(
                'src/neuronal/data/model_trained.json',
                'w'
            ) as f:
                json.dump(model_data, f, indent=4)

            print(
                "Modelo entrenado guardado en "
                "'src/neuronal/data/model_trained.json'"
            )

            self.nn = load_trained_model(
                "src/neuronal/data/model_trained.json",
                input_dim=X.shape[1]
            )

            self.neuro_evaluation = True
            with open(PATH_CONT_EVOLUTION, 'r') as file:
                cont_evolution = json.load(file)
            cont_evolution['trained_cont'] += 1
            with open(PATH_CONT_EVOLUTION, 'w') as file:
                json.dump(cont_evolution, file, indent=4)
    
    
    def run(self):

        self.initialize_neuronal()

        peticiones.initialize_mt5()

        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://0.0.0.0:5555")

        print("Servidor Python listo...")

        while True:

            self.cont += 1

            message = socket.recv_string()
            print("Recibido de MT5:", message)

            if message == "FIN":
                print("EA finalizó. Cerrando servidor.")
                break

            date = message.split(",")[0]
            open_price = float(message.split(",")[1])

            self.dict_files.clear()

            for symbol in self.list_symbols:
                try:
                    self.procesar_symbol(symbol, date, self.cont)
                except Exception as e:
                    print(f"❌ Error procesando {symbol}: {e}")

            if self.is_open:
                self.handle_close_logic(open_price)
            else:
                self.handle_open_logic(open_price)

            self.retrain_if_needed(date)

            print("→ Enviando orden:", self.order)
            socket.send_string(self.order)
    
    
if __name__ == "__main__":
    trading_server = TradingServer()
    trading_server.run()