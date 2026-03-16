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
from src.utils.indicadores_for_crossing import extract_indicadores
from src.utils.common_functions import get_previous_4_6, hora_en_mercado
from src.neuronal.entrenar import (
    load_trained_model,
    predict_from_inputs,
    BinaryNN,
    load_data
)


def _max_decimals(series, sample_size=1000):
    s = series.dropna()
    if s.empty:
        return 0
    if len(s) > sample_size:
        s = s.sample(sample_size, random_state=0)
    max_dec = 0
    for value in s:
        text = f"{value:.10f}".rstrip('0').rstrip('.')
        if '.' in text:
            dec = len(text.split('.')[1])
            if dec > max_dec:
                max_dec = dec
    return max_dec


def get_pip_and_point_size(symbol, price_series):
    digits = _max_decimals(price_series)
    pip_size = 0.01 if 'JPY' in symbol.upper() else 0.0001
    point_size = (10 ** (-digits)) if digits > 0 else (pip_size / 10)
    return pip_size, point_size

class Backtester:

    def __init__(self, principal_symbol, mercado, algorithm, date_end=None):

        print("Starting backtest...")
        self.comienzo = time.time()

        # Estado global del sistema
        self.index = 0
        self.best_score = 0.0
        self.best_model_data = None
        self.peor_trade = 0.0
        self.info_score = {}
        self.dict_pips_best = {}
        self.min_trades_score = 40
        self.trade_reliability_k = 120
        self.low_trade_penalty_power = 1.75
        self.data_end = date_end
        self.principal_symbol = principal_symbol
        self.mercado = mercado
        self.algorithm = algorithm
        
        if self.algorithm == "UP":
            self.other_algorithm = "DOWN"
        else:
            self.other_algorithm = "UP"
        
        self.DATA_CACHE = {}
        self.COMBINED = {}

        # Cargar sistema
        self.load_config()
        self.prepare_base_data()
        self.pip_size, self.point_size = get_pip_and_point_size(
            self.principal_symbol,
            self.df_train['open'] if 'open' in self.df_train.columns else self.df_valid['open']
        )

        self.setup_operators_and_mappings()
        self.load_nodes()
        self.extract_required_columns()
        self.preload_data()

        self.horas_mercado = [hora_en_mercado(h, self.mercado) for h in range(24)]
    
    
    def load_config(self):
        with open(f'config/divisas/{self.principal_symbol}/config_{self.principal_symbol}.json', 'r') as file:
            config_symbol = json.load(file)
        with open(f'config/general_config.json', 'r', encoding='utf-8') as f:
            general_config = json.load(f)
        
        self.list_symbols = config_symbol['list_symbol']
        self.list_symbols.insert(0, self.principal_symbol)
        self.general_config = general_config
        self.config_symbol = config_symbol
        data_start_is, data_end_is = get_previous_4_6(
            self.general_config['dateStart'], 
            self.general_config['dateEnd']
        )
        self.data_start_is = data_start_is
        self.data_end_is = data_end_is
        
    def prepare_base_data(self):

        if self.data_end is None:
            df_is = pd.read_csv(f'output/{self.principal_symbol}/is_os/is.csv')
            df_os = pd.read_csv(f'output/{self.principal_symbol}/is_os/os.csv')

            df_base1 = (
                pd.concat([df_is, df_os], ignore_index=True)
                .drop_duplicates(subset='time')
            )

            df_base1['time'] = pd.to_datetime(df_base1['time'])

            df_base = df_base1[
                df_base1['time'] >= datetime.strptime(
                    self.data_start_is, '%Y-%m-%d'
                )
            ]
        else:
            timeframes = get_timeframes()
            fecha_str = self.data_end
            fecha_dt = datetime.strptime(fecha_str, "%Y-%m-%d")

            # Restar 6 años
            fecha_6_atras = fecha_dt - relativedelta(years=3)

            # Volver a formato string
            fecha_6_atras_str = fecha_6_atras.strftime("%Y-%m-%d")
            timeframe = timeframes.get(self.general_config['timeframe']) 
            rates = get_historical_data(self.principal_symbol, timeframe, fecha_6_atras_str, self.data_end)
            df_base = pd.DataFrame(rates)
            df_base['time'] = pd.to_datetime(df_base['time'], unit='s')
            df_base_for_indicators = df_base.copy()
            list_files_ = os.listdir(f'output/{self.principal_symbol}/extrac_os')
            indicadores_m =pd.read_parquet(f'output/{self.principal_symbol}/extrac_os/{list_files_[0]}', columns=['time'])
            time_ultimo_indicador = indicadores_m['time'].iloc[-1]
            index_coincidencia = df_base_for_indicators[df_base_for_indicators['time'] == time_ultimo_indicador].index[0]
            if index_coincidencia >= 830:
                df_base_for_indicators = df_base_for_indicators.iloc[index_coincidencia-830:]
            print(len(df_base_for_indicators))
            if len(df_base_for_indicators) - 830 > 20:
                extract_indicadores(self.principal_symbol, df_base_for_indicators)
            
            
        df_base = df_base.sort_values('time').set_index('time')
        corte = int(0.8 * len(df_base))
        self.df_train = df_base.iloc[:corte]
        self.df_valid = df_base.iloc[corte:]   
    
  
    def actualizar_dict(self, principal, nuevo_dict):
        """Actualiza dict_pips_best con nuevos datos."""
        for k, v in nuevo_dict.items():
            if k not in principal:
                principal[k] = v  # Guardar valor real
            else:
                # Promedio ponderado (como en TradingEngine.record_trade)
                principal[k] = (
                    principal[k] * 0.8 + v * 0.2
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
       
       
    def setup_operators_and_mappings(self):

        self.operadores = {
            "<": operator.lt,
            "<=": operator.le,
            ">": operator.gt,
            ">=": operator.ge,
            "==": operator.eq,
            "!=": operator.ne
        }

        with open(f'output/{self.principal_symbol}/data_for_neuronal/maping/maping_open_{self.mercado}_{self.algorithm}.json', 'r') as file:
            self.maping_open = json.load(file)

        with open(f'output/{self.principal_symbol}/data_for_neuronal/maping/maping_close_{self.mercado}_{self.algorithm}.json', 'r') as file:
            self.maping_close = json.load(file)  
    
  
    def load_nodes(self):

        self.dict_nodos = {}

        for i, symbol in enumerate(self.list_symbols):
            self.dict_nodos[symbol] = self.parsear_nodos(
                get_nodes_by_label(self.principal_symbol, symbol, self.mercado, self.algorithm)
            )

        self.nodos_close = self.parsear_nodos(
            get_nodes_by_label(self.principal_symbol, self.principal_symbol, self.mercado, self.other_algorithm)
        )  
       

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
            "col_map": {col: i for i, col in enumerate(df.columns)},
            "index_values": df.index.values
        }


    def calculate_trade_pips(self, open_price_open, open_price_close, spread_open):
        if self.algorithm == 'UP':
            movement = open_price_close - open_price_open
        else:
            movement = open_price_open - open_price_close

        movement_pips = movement / self.pip_size
        spread_pips = spread_open * self.point_size / self.pip_size
        return movement_pips - spread_pips
        

    def preload_data(self):

        ini = time.time()

        # CLOSE
        FILES_OS_CLOSE = {
            f.split('_')[0]: f
            for f in os.listdir(f'output/{self.principal_symbol}/extrac_os')
        }

        for nodo in self.nodos_close:

            path_is = f'output/{self.principal_symbol}/extrac/{nodo["file"]}'
            file_base = nodo["file"].split('_')[0]
            file_os = FILES_OS_CLOSE[file_base]
            path_os = f'output/{self.principal_symbol}/extrac_os/{file_os}'

            self.COMBINED[("close", nodo["file"])] = self.build_combined(path_is, path_os)

        # OPEN
        for symbol in self.list_symbols:

            if symbol == self.principal_symbol:
                path = f'output/{self.principal_symbol}'
                path_os_root = f'{path}/extrac_os'
            else:
                path = f'output/{self.principal_symbol}/crossing/{symbol}'
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
     

    def train_iteration(self):

        path_data_red = f'output/{self.principal_symbol}/data_for_neuronal/data/data_{self.mercado}_{self.algorithm}.csv'
        X, Y = load_data(path_data_red)

        nn = load_trained_model(
            f"output/{self.principal_symbol}/data_for_neuronal/model_trainer/model_{self.mercado}_{self.algorithm}.json",
            input_dim=X.shape[1]
        )

        is_open = False
        open_price_open = 0.0
        spread_open = 0.0
        sum_pips = 0.0
        entry_red_open = ''
        cierre = 0

        dict_pips = {}

        for row in self.df_train.itertuples():
            
            time_actual = row.Index
            time_actual_np = np.datetime64(time_actual)
            if not is_open:
                if not self.horas_mercado[time_actual.hour]:
                    continue
            
            open_price = row.open 

            # =========================
            # CIERRE
            # =========================
            if is_open:
                cierre += 1
                valid_closes = []

                for nodo in self.nodos_close:

                    df_struct = self.COMBINED[("close", nodo["file"])]
                    pos = df_struct["index_values"].searchsorted(time_actual_np)
                    if pos == 0:
                        continue

                    if self.cumple_condiciones_fast(df_struct, pos - 1, nodo["conditions"]):
                        nodo_close = self.maping_close[nodo["key"]]
                        clase, prob = predict_from_inputs(nn, entry_red_open, nodo_close)

                        if self.index == 1:
                            clase = random.randint(0, 1)

                        valid_closes.append((nodo_close, int(clase)))

                if valid_closes:
                    trade_pips = self.calculate_trade_pips(
                        open_price_open,
                        open_price,
                        spread_open,
                    )

                    # Guardar TODOS los nodos de cierre válidos para enriquecer dict_pips
                    for nodo_close, _ in valid_closes:
                        key = f'{entry_red_open}_{nodo_close}'
                        if key not in dict_pips:
                            dict_pips[key] = trade_pips
                        else:
                            dict_pips[key] = dict_pips[key] * 0.8 + trade_pips * 0.2

                    # Cerrar si al menos un nodo válido da clase de cierre
                    cerrar = any(clase == 1 for _, clase in valid_closes)
                    if cerrar:
                        is_open = False
                        sum_pips += trade_pips
                        print(f"SUM PIPS: {time_actual}: {sum_pips}", cierre)

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
                        pos = df_struct["index_values"].searchsorted(time_actual_np)
                        if pos == 0:
                            continue

                        if self.cumple_condiciones_fast(df_struct, pos - 1, nodo["conditions"]):

                            cumple_alguno = True

                            if symbol == self.list_symbols[-1]:
                                open_price_open = open_price
                                spread_open = getattr(row, 'spread', 0.0)
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
            self.dict_pips_best = self.actualizar_dict(self.dict_pips_best, dict_pips)
        print(len(self.dict_pips_best), "pips para actualizar en la próxima iteración")

        # =========================
        # REENTRENAMIENTO
        # =========================

        config = {
            "general": self.general_config,
            "symbol": self.config_symbol,
            "principal_symbol": self.principal_symbol
        }
        data_for_neuronal(config, self.mercado, self.algorithm, self.dict_pips_best)

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

        with open(f'output/{self.principal_symbol}/data_for_neuronal/model_trainer/model_{self.mercado}_{self.algorithm}.json', 'w') as f:
            json.dump(model_data, f, indent=4)

        print(f"Modelo entrenado guardado en 'output/{self.principal_symbol}/data_for_neuronal/model_trainer/model_{self.mercado}_{self.algorithm}.json'")

        return model_data
    
  
    def validate_iteration(self):

        path_data_red = f'output/{self.principal_symbol}/data_for_neuronal/data/data_{self.mercado}_{self.algorithm}.csv'
        X, Y = load_data(path_data_red)

        nn = load_trained_model(
            f'output/{self.principal_symbol}/data_for_neuronal/model_trainer/model_{self.mercado}_{self.algorithm}.json',
            input_dim=X.shape[1]
        )

        is_open = False
        open_price_open = 0.0
        spread_open = 0.0
        entry_red_open = ''
        cierre = 0
        time_comienzo = None

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
            time_actual_np = np.datetime64(time_actual)
            if not is_open:
                if not self.horas_mercado[time_actual.hour]:
                    continue
            open_price = row.open

            # =========================
            # CIERRE
            # =========================
            if is_open:
                cierre += 1
                for nodo in self.nodos_close:
                    df_struct = self.COMBINED[("close", nodo["file"])]
                    pos = df_struct["index_values"].searchsorted(time_actual_np)
                    if pos == 0:
                        continue

                    cerrar = False

                    if self.cumple_condiciones_fast(df_struct, pos - 1, nodo["conditions"]):

                        nodo_close = self.maping_close[nodo["key"]]
                        clase, prob = predict_from_inputs(nn, entry_red_open, nodo_close)
                        if clase == 1:
                            cerrar = True

                    if cerrar:
                        trade_pips = self.calculate_trade_pips(
                            open_price_open,
                            open_price,
                            spread_open,
                        )

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

                        print(f"SUM PIPS:{time_comienzo} ---- {time_actual}: {sum_pips}", cierre)
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
                        pos = df_struct["index_values"].searchsorted(time_actual_np)
                        
                        if pos == 0:
                            continue

                        if self.cumple_condiciones_fast(df_struct, pos - 1, nodo["conditions"]):

                            cumple_alguno = True

                            if symbol == self.list_symbols[-1]:
                                open_price_open = open_price
                                spread_open = getattr(row, 'spread', 0.0)
                                is_open = True
                                time_comienzo = time_actual
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
    

    def calculate_score(self, metrics, model_data):

        profit_factor_adj = min(metrics["profit_factor"], 5)
        sharpe_adj = min(metrics["sharpe"], 3)
        n_ops = metrics["cantidad_operaciones"]

        score_base = (
            (metrics["expectancy"] * 2.0) +
            (profit_factor_adj * 1.5) +
            (sharpe_adj * 2.0) +
            (metrics["winrate"] * 1.0)
        )

        volumen_factor = np.log(n_ops + 1)
        reliability_factor = n_ops / (n_ops + self.trade_reliability_k) if n_ops > 0 else 0.0
        min_trades_factor = min(1.0, n_ops / self.min_trades_score) if self.min_trades_score > 0 else 1.0
        low_trade_penalty = min_trades_factor ** self.low_trade_penalty_power

        score = score_base * volumen_factor * reliability_factor * low_trade_penalty

        print(
            f"Score calculado: {score:.4f} | base={score_base:.4f} | ops={n_ops} | "
            f"vol={volumen_factor:.4f} | rel={reliability_factor:.4f} | "
            f"low_ops={low_trade_penalty:.4f}"
        )

        if score > self.best_score:

            self.best_score = score
            self.best_model_data = model_data
            self.peor_trade = min(metrics["lista_pips"]) if metrics["lista_pips"] else 0
            metrics["peor_trade"] = self.peor_trade
            metrics["iteracion"] = self.index
            metrics["score"] = score
            metrics["score_base"] = score_base
            metrics["volumen_factor"] = volumen_factor
            metrics["reliability_factor"] = reliability_factor
            metrics["low_trade_penalty"] = low_trade_penalty
            self.info_score = metrics
            print(f"Nuevo mejor modelo encontrado con score: {score:.4f}")
        return score


    def run(self):
        last_model_data = None

        for i in range(40):
            random.shuffle(self.nodos_close)
            print(f"\n--- Iteración {i+1} ---")
            self.index = i + 1
            model_data = self.train_iteration()#
            last_model_data = model_data
            metrics = self.validate_iteration()
            self.calculate_score(metrics, model_data)

        if self.best_model_data is None:
            self.best_model_data = last_model_data

        with open(f'output/{self.principal_symbol}/data_for_neuronal/model_trainer/model_{self.mercado}_{self.algorithm}.json', 'w') as f:
            json.dump(self.best_model_data, f, indent=4)
        with open(f'output/{self.principal_symbol}/data_for_neuronal/best_score/score_{self.mercado}_{self.algorithm}.json', 'w') as f:
            json.dump({
                "metrics": self.info_score
            }, f, indent=4)
        print(f'tiempo total: {time.time() - self.comienzo:.2f} segundos')
    
       
      
if __name__ == "__main__":
    inn = time.time()
    backtester = Backtester('AUDCAD', 'Asia', 'DOWN', date_end=None) 
    backtester.run()
    print(f'segundos {time.time()-inn}')
    