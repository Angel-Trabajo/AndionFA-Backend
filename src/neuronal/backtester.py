# ============================================================
# IMPORTS
# ============================================================

import os
import sys
import json
import time
import ast
import operator
import random

import pandas as pd
import numpy as np
import pyarrow.parquet as pq

from datetime import datetime
from dateutil.relativedelta import relativedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.routes.peticiones import get_historical_data, get_timeframes
from src.db.query import get_nodes_by_label
from src.neuronal.data_para_entrenar import data_for_neuronal
from src.utils.crossing_funtion.crear_indicadores_in_crossing import extract_indicadores
from src.neuronal.entrenar import (
    load_trained_model,
    predict_from_inputs,
    BinaryNN,
    load_data
)

class Backtester:

    def __init__(self, date_end=None):

        print("Starting backtest...")
        self.comienzo = time.time()

        # Estado global del sistema
        self.index = 0
        self.best_score = 0.0
        self.best_model_data = None
        self.peor_trade = 0.0
        self.dict_pips_best = {}
        self.data_end = date_end

        self.DATA_CACHE = {}
        self.COMBINED = {}

        # Cargar sistema
        self.load_config()
        self.prepare_base_data()

        self.setup_operators_and_mappings()
        self.load_nodes()
        self.extract_required_columns()
        self.preload_data()
    
    
    def load_config(self):
        with open('config/config_test/config_test_red.json') as f:
            config = json.load(f)

        self.algorithm = config['algorithm']
        self.other_algorithm = 'DOWN' if self.algorithm == 'UP' else 'UP'

        with open(f'config/list_{self.algorithm}.json') as f:
            config_extractor = json.load(f)

        with open('config/config_crossing/config_crossing.json') as f:
            config_crossing = json.load(f)

        with open('config/config_node/config_node.json') as f:
            config_node = json.load(f)

        self.list_symbols = config_extractor['list']
        self.principal_symbol = config_crossing['principal_symbol']
        self.list_symbols.insert(0, self.principal_symbol)
        self.config_crossing = config_crossing
        self.config_node = config_node   
        
        
    def prepare_base_data(self):

        if self.data_end is None:
            df_is = pd.read_csv('output/is_os/is.csv')
            df_os = pd.read_csv('output/is_os/os.csv')

            df_base1 = (
                pd.concat([df_is, df_os], ignore_index=True)
                .drop_duplicates(subset='time')
            )

            df_base1['time'] = pd.to_datetime(df_base1['time'])

            df_base = df_base1[
                df_base1['time'] >= datetime.strptime(
                    self.config_node['dateStart'], '%Y-%m-%d'
                ) - relativedelta(years=4)
            ]
        else:
            timeframes = get_timeframes()
            fecha_str = self.data_end
            fecha_dt = datetime.strptime(fecha_str, "%Y-%m-%d")

            # Restar 6 años
            fecha_6_atras = fecha_dt - relativedelta(years=3)

            # Volver a formato string
            fecha_6_atras_str = fecha_6_atras.strftime("%Y-%m-%d")
            timeframe = timeframes.get(self.config_crossing['timeframe']) 
            rates = get_historical_data(self.principal_symbol, timeframe, fecha_6_atras_str, self.data_end)
            df_base = pd.DataFrame(rates)
            df_base['time'] = pd.to_datetime(df_base['time'], unit='s')
            df_base_for_indicators = df_base.copy()
            list_files_ = os.listdir('output/extrac_os')
            indicadores_m =pd.read_parquet(f'output/extrac_os/{list_files_[0]}', columns=['time'])
            time_ultimo_indicador = indicadores_m['time'].iloc[-1]
            index_coincidencia = df_base_for_indicators[df_base_for_indicators['time'] == time_ultimo_indicador].index[0]
            if index_coincidencia >= 830:
                df_base_for_indicators = df_base_for_indicators.iloc[index_coincidencia-830:]
            print(len(df_base_for_indicators))
            if len(df_base_for_indicators) - 830 > 20:
                extract_indicadores(df_base_for_indicators)
            
            
        df_base = df_base.sort_values('time').set_index('time')
        corte = int(0.8 * len(df_base))
        self.df_train = df_base.iloc[:corte]
        self.df_valid = df_base.iloc[corte:]   
       
    
    
    # ============================================================
    # FUNCIONES AUXILIARES
    # ============================================================

    def actualizar_dict(self, principal, nuevo_dict):
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


    def parsear_nodos(self, nodos):
        return [
            {
                "key": n[0],
                "conditions": ast.literal_eval(n[0]),
                "file": n[1]
            }
            for n in nodos
        ]


    def cumple_condiciones_fast(self, df_struct, row_idx, condiciones):
        row = df_struct["values"][row_idx]
        col_map = df_struct["col_map"]

        for col, op, valor in condiciones:
            if col not in col_map:
                return False
            v = row[col_map[col]]
            if v is None or not self.operadores[op](v, valor):
                return False
        return True   
       
    # ============================================================
    # OPERADORES Y MAPEINGS
    # ============================================================

    def setup_operators_and_mappings(self):

        self.operadores = {
            "<": operator.lt,
            "<=": operator.le,
            ">": operator.gt,
            ">=": operator.ge,
            "==": operator.eq,
            "!=": operator.ne
        }

        with open('src/neuronal/data/maping_open.json', 'r') as file:
            self.maping_open = json.load(file)

        with open('src/neuronal/data/maping_close.json', 'r') as file:
            self.maping_close = json.load(file)  
    
       
    # ============================================================
    # CARGA DE NODOS
    # ============================================================

    def load_nodes(self):

        self.dict_nodos = {}

        for i, symbol in enumerate(self.list_symbols):
            label = symbol if i == 0 else f'crossing_{self.principal_symbol}_dbs/{symbol}'
            self.dict_nodos[symbol] = self.parsear_nodos(
                get_nodes_by_label(label, self.algorithm)
            )

        self.nodos_close = self.parsear_nodos(
            get_nodes_by_label(self.principal_symbol, self.other_algorithm)
        )  
       
    
    # ============================================================
    # EXTRAER COLUMNAS NECESARIAS
    # ============================================================

    def extract_required_columns(self):

        self.columnas_usadas = set()

        for symbol in self.dict_nodos:
            for nodo in self.dict_nodos[symbol]:
                for col, _, _ in nodo["conditions"]:
                    self.columnas_usadas.add(col)

        for nodo in self.nodos_close:
            for col, _, _ in nodo["conditions"]:
                self.columnas_usadas.add(col)

        self.columnas_usadas.add("time")


    def load_df(self, path):

        parquet_path = path.replace(".csv", ".parquet")

        if parquet_path in self.DATA_CACHE:
            return self.DATA_CACHE[parquet_path]

        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"No existe parquet: {parquet_path}")
        schema = pq.read_schema(parquet_path)
        columnas_disponibles = set(schema.names)

        columnas_validas = list(self.columnas_usadas & columnas_disponibles)

        df = pd.read_parquet(
            parquet_path,
            columns=columnas_validas if columnas_validas else None
        )

        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time').set_index('time')
        else:
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

        self.DATA_CACHE[parquet_path] = df
        return df
      
    
    def build_combined(self, path_is, path_os):

        df_is = self.load_df(path_is)
        df_os = self.load_df(path_os)

        df = pd.concat([df_is, df_os])
        df = df[~df.index.duplicated(keep='last')]

        return {
            "df": df,
            "values": df.values,
            "col_map": {col: i for i, col in enumerate(df.columns)}
        }
        
        
    # ============================================================
    # PRELOAD DATA
    # ============================================================

    def preload_data(self):

        ini = time.time()

        # CLOSE
        FILES_OS_CLOSE = {
            f.split('_')[0]: f
            for f in os.listdir('output/extrac_os')
        }

        for nodo in self.nodos_close:

            path_is = f'output/extrac/{nodo["file"]}'
            file_base = nodo["file"].split('_')[0]
            file_os = FILES_OS_CLOSE[file_base]
            path_os = f'output/extrac_os/{file_os}'

            self.COMBINED[("close", nodo["file"])] = self.build_combined(path_is, path_os)

        # OPEN
        for symbol in self.list_symbols:

            if symbol == self.principal_symbol:
                path = 'output'
                path_os_root = 'output/extrac_os'
            else:
                path = f'output/crossing_{self.principal_symbol}/{symbol}'
                path_os_root = f'{path}/extrac_os'

            FILES_LOCAL = {
                f.split('_')[0]: f
                for f in os.listdir(path_os_root)
            }

            for nodo in self.dict_nodos[symbol]:

                path_is = f'{path}/extrac/{nodo["file"]}'
                file_base = nodo["file"].split('_')[0]
                file_os = FILES_LOCAL[file_base]
                path_os = f'{path}/extrac_os/{file_os}'

                self.COMBINED[("open", symbol, nodo["file"])] = self.build_combined(path_is, path_os)

        print(f"Tiempo de carga: {time.time() - ini:.2f} segundos")
        print("Archivos cargados:", len(self.DATA_CACHE))
     
    
    # ============================================================
    # TRAIN ITERATION (IS)
    # ============================================================

    def train_iteration(self):

        path_data_red = f'src/neuronal/data/data_for_neuronal_{self.algorithm}_{self.principal_symbol}.csv'
        X, Y = load_data(path_data_red)

        nn = load_trained_model(
            "src/neuronal/data/model_trained.json",
            input_dim=X.shape[1]
        )

        is_open = False
        open_price_open = 0.0
        sum_pips = 0.0
        entry_red_open = ''
        cierre = 0

        dict_pips = {}

        for row in self.df_train.itertuples():

            time_actual = row.Index
            open_price = row.open

            # =========================
            # CIERRE
            # =========================
            if is_open:
                cierre += 1

                for nodo in self.nodos_close:

                    df_struct = self.COMBINED[("close", nodo["file"])]
                    df = df_struct["df"]

                    pos = df.index.searchsorted(time_actual)
                    if pos == 0:
                        continue

                    cerrar = False

                    if self.cumple_condiciones_fast(df_struct, pos - 1, nodo["conditions"]):

                        nodo_close = self.maping_close[nodo["key"]]

                        if self.algorithm == 'UP':
                            trade_pips = open_price - open_price_open
                        else:
                            trade_pips = open_price_open - open_price

                        clase, prob = predict_from_inputs(nn, entry_red_open, nodo_close)

                        key = f'{entry_red_open}_{nodo_close}'

                        if key not in dict_pips:
                            dict_pips[key] = trade_pips
                        else:
                            dict_pips[key] = (
                                dict_pips[key] * 0.9 + trade_pips * 0.1
                            )
                        if self.index == 1:
                            clase = random.randint(0, 1)
                        if clase == 1:
                            cerrar = True

                    if cerrar:
                        is_open = False
                        print(f"SUM PIPS: {time_actual}: {sum_pips}", cierre)
                        break

            # =========================
            # APERTURA
            # =========================
            else:

                cierre = 0

                for symbol in self.list_symbols:

                    cumple_alguno = False
                    nodo_open_list = self.dict_nodos[symbol]

                    for nodo in nodo_open_list:

                        df_struct = self.COMBINED[("open", symbol, nodo["file"])]
                        df = df_struct["df"]

                        pos = df.index.searchsorted(time_actual)
                        if pos == 0:
                            continue

                        if self.cumple_condiciones_fast(df_struct, pos - 1, nodo["conditions"]):

                            cumple_alguno = True

                            if symbol == self.list_symbols[-1]:
                                open_price_open = open_price
                                is_open = True
                                nodo_open = self.maping_open[nodo["key"]]
                                entry_red_open = nodo_open

                            break

                    if not cumple_alguno:
                        break

        # =========================
        # SELECCIÓN TOP/BOTTOM
        # =========================
        if self.index == 1:
            self.dict_pips_best = dict_pips
        else:
            top = dict(
                sorted(dict_pips.items(), key=lambda x: x[1], reverse=True)[:40]
            )

            bottom = dict(
                sorted(dict_pips.items(), key=lambda x: x[1])[:40]
            )

            self.dict_pips_best = self.actualizar_dict(self.dict_pips_best, top)
            self.dict_pips_best = self.actualizar_dict(self.dict_pips_best, bottom)

        print(len(self.dict_pips_best), "pips para actualizar en la próxima iteración")

        # =========================
        # REENTRENAMIENTO
        # =========================

        data_for_neuronal(self.algorithm, self.principal_symbol, self.dict_pips_best)

        X, Y = load_data(path_data_red)

        print("Datos cargados:")
        print("X shape:", X.shape)
        print("Y shape:", Y.shape)

        nn = BinaryNN(input_dim=X.shape[1], lr=0.01, target_loss=0.10)
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

        with open('src/neuronal/data/model_trained.json', 'w') as f:
            json.dump(model_data, f, indent=4)

        print("Modelo entrenado guardado en 'src/neuronal/data/model_trained.json'")

        return model_data
    
    
    # ============================================================
    # VALIDATION (OS)
    # ============================================================

    def validate_iteration(self):

        path_data_red = f'src/neuronal/data/data_for_neuronal_{self.algorithm}_{self.principal_symbol}.csv'
        X, Y = load_data(path_data_red)

        nn = load_trained_model(
            "src/neuronal/data/model_trained.json",
            input_dim=X.shape[1]
        )

        is_open = False
        open_price_open = 0.0
        entry_red_open = ''
        cierre = 0

        cantidad_operaciones = 0
        operaciones_acertadas = 0
        operaciones_perdedoras = 0

        ganancia_bruta = 0.0
        perdida_bruta = 0.0
        sum_pips = 0.0

        lista_pips = []
        perdidas_seguidas = 0
        mas_perdidas_seguidas = 0

        for row in self.df_valid.itertuples():

            time_actual = row.Index
            open_price = row.open

            # =========================
            # CIERRE
            # =========================
            if is_open:
                cierre += 1
                for nodo in self.nodos_close:
                    df_struct = self.COMBINED[("close", nodo["file"])]
                    df = df_struct["df"]

                    pos = df.index.searchsorted(time_actual)
                    if pos == 0:
                        continue

                    cerrar = False

                    if self.cumple_condiciones_fast(df_struct, pos - 1, nodo["conditions"]):

                        nodo_close = self.maping_close[nodo["key"]]
                        clase, prob = predict_from_inputs(nn, entry_red_open, nodo_close)
                        if clase == 1:
                            cerrar = True

                    if cerrar:

                        if self.algorithm == 'UP':
                            trade_pips = open_price - open_price_open
                        else:
                            trade_pips = open_price_open - open_price

                        if trade_pips < 0:
                            perdidas_seguidas += 1
                        else:
                            perdidas_seguidas = 0

                        is_open = False
                        cantidad_operaciones += 1
                        sum_pips += trade_pips
                        lista_pips.append(trade_pips)

                        if trade_pips > 0:
                            operaciones_acertadas += 1
                            ganancia_bruta += trade_pips
                        else:
                            operaciones_perdedoras += 1
                            perdida_bruta += abs(trade_pips)

                        print(f"SUM PIPS: {time_actual}: {sum_pips}", cierre)
                        break

            # =========================
            # APERTURA
            # =========================
            else:

                cierre = 0

                for symbol in self.list_symbols:

                    cumple_alguno = False
                    nodo_open_list = self.dict_nodos[symbol]

                    for nodo in nodo_open_list:

                        df_struct = self.COMBINED[("open", symbol, nodo["file"])]
                        df = df_struct["df"]

                        pos = df.index.searchsorted(time_actual)
                        if pos == 0:
                            continue

                        if self.cumple_condiciones_fast(df_struct, pos - 1, nodo["conditions"]):

                            cumple_alguno = True

                            if symbol == self.list_symbols[-1]:
                                open_price_open = open_price
                                is_open = True
                                nodo_open = self.maping_open[nodo["key"]]
                                entry_red_open = nodo_open

                            break

                    if not cumple_alguno:
                        break

            if perdidas_seguidas > mas_perdidas_seguidas:
                mas_perdidas_seguidas = perdidas_seguidas

        # =========================
        # MÉTRICAS
        # =========================

        winrate = (operaciones_acertadas / cantidad_operaciones) if cantidad_operaciones > 0 else 0.0

        profit_factor = (
            ganancia_bruta / perdida_bruta
            if perdida_bruta > 0 else 0
        )

        avg_win = ganancia_bruta / operaciones_acertadas if operaciones_acertadas > 0 else 0
        avg_loss = perdida_bruta / operaciones_perdedoras if operaciones_perdedoras > 0 else 0

        expectancy = (winrate * avg_win) - ((1 - winrate) * avg_loss)

        sharpe = 0
        if len(lista_pips) > 1 and np.std(lista_pips) != 0:
            sharpe = np.mean(lista_pips) / np.std(lista_pips)

        print(f"""
        Operaciones: {cantidad_operaciones}
        Winrate: {winrate*100:.2f}%
        Profit Factor: {profit_factor:.2f}
        Expectancy: {expectancy:.4f}
        Pips Totales: {sum_pips:.2f}
        """)

        return {
            "winrate": winrate,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "sharpe": sharpe,
            "cantidad_operaciones": cantidad_operaciones,
            "sum_pips": sum_pips,
            "lista_pips": lista_pips,
            "mas_perdidas_seguidas": mas_perdidas_seguidas
        }
    
    
    # ============================================================
    # SCORE
    # ============================================================

    def calculate_score(self, metrics, model_data):

        profit_factor_adj = min(metrics["profit_factor"], 5)
        sharpe_adj = min(metrics["sharpe"], 3)

        score_base = (
            (metrics["expectancy"] * 2.0) +
            (profit_factor_adj * 1.5) +
            (sharpe_adj * 2.0) +
            (metrics["winrate"] * 1.0)
        )

        penalizacion = np.log(metrics["cantidad_operaciones"] + 1)

        score = score_base * penalizacion

        print(f"Score calculado: {score:.4f}")

        if score > self.best_score:

            self.best_score = score
            self.best_model_data = model_data
            self.peor_trade = min(metrics["lista_pips"]) if metrics["lista_pips"] else 0

            with open('src/neuronal/data/best_score.json', 'w') as f:
                json.dump({
                    "score": self.best_score,
                    "operaciones": metrics["cantidad_operaciones"],
                    "winrate": metrics["winrate"],
                    "profit_factor": metrics["profit_factor"],
                    "expectancy": metrics["expectancy"],
                    "pips_totales": metrics["sum_pips"],
                    "peor_trade": self.peor_trade,
                    "mas_perdidas_seguidas": metrics["mas_perdidas_seguidas"],
                    "dict_pips_best": self.dict_pips_best
                }, f, indent=4)

        return score
    
    # ============================================================
    # RUN LOOP (PARCIAL)
    # ============================================================

    def run(self):

        for i in range(40):
            print(f"\n--- Iteración {i+1} ---")
            self.index = i + 1
            model_data = self.train_iteration()
            metrics = self.validate_iteration()
            self.calculate_score(metrics, model_data)

        with open('src/neuronal/data/model_trained.json', 'w') as f:
            json.dump(self.best_model_data, f, indent=4)

        print(f'tiempo total: {time.time() - self.comienzo:.2f} segundos')
    
       
      
if __name__ == "__main__":
    inn = time.time()
    print(pd.read_parquet('output/extrac_os/Ext-011023_EURUSD_20210101_20230101_timeframeH1.parquet'))
    backtester = Backtester() 
    backtester.run()
    print(f'segundos {time.time()-inn}')
    