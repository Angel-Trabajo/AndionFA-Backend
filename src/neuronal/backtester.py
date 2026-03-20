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
    load_data,
    _serialize_xgb_to_base64
)


def _normalize_time_column(df, column_name='time'):
    if column_name not in df.columns:
        return df
    normalized_df = df.copy()
    normalized_df[column_name] = pd.to_datetime(normalized_df[column_name], errors='coerce')
    return normalized_df.dropna(subset=[column_name])


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
        self.best_score = float('-inf')
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
        self.n_iterations = 25 if not self.data_end else 10   
        self.use_wfo = True
        self.wfo_train_years = 4
        self.wfo_test_years = 1
        self.wfo_step_years = 1
        # Umbral legado (compatibilidad para métricas/JSON); ya no gobierna cierres en regresión
        self.min_close_prob = 0.05
        # Cierre inteligente por edge (regresión)
        self.edge_threshold = 0.01
        self.edge_decay_factor = 0.6
        self.edge_delta_threshold = -0.05
        self.min_open_edge = -0.05
        self.max_loss_close_pips = -15
        self.hard_stop_pips = -25
        self.future_horizon_bars = 5
        self.stop_loss_pips = -50
        self.take_profit_pips = 120
        self.max_trade_duration = 150
        
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

        self.n_iterations = int(general_config.get('backtester_iterations', self.n_iterations))
        self.use_wfo = bool(general_config.get('backtester_use_wfo', self.use_wfo))
        self.wfo_train_years = int(general_config.get('backtester_wfo_train_years', self.wfo_train_years))
        self.wfo_test_years = int(general_config.get('backtester_wfo_test_years', self.wfo_test_years))
        self.wfo_step_years = int(general_config.get('backtester_wfo_step_years', self.wfo_step_years))

        # Parámetros de edge/riesgo configurables por JSON para ajustar frecuencia sin tocar código
        self.min_open_edge = float(general_config.get('min_open_edge', self.min_open_edge))
        self.edge_threshold = float(general_config.get('edge_threshold', self.edge_threshold))
        self.edge_decay_factor = float(general_config.get('edge_decay_factor', self.edge_decay_factor))
        self.edge_delta_threshold = float(general_config.get('edge_delta_threshold', self.edge_delta_threshold))
        self.max_loss_close_pips = float(general_config.get('max_loss_close_pips', self.max_loss_close_pips))
        self.hard_stop_pips = float(general_config.get('hard_stop_pips', self.hard_stop_pips))
        
        
    def prepare_base_data(self):
        
        df_is = pd.read_csv(f'output/{self.principal_symbol}/is_os/is.csv')
        df_os = pd.read_csv(f'output/{self.principal_symbol}/is_os/os.csv')
        df_is = _normalize_time_column(df_is)
        df_os = _normalize_time_column(df_os)
        
        if self.data_end :
            timeframes = get_timeframes()
            date_start = pd.to_datetime(df_os['time'].iloc[-1]).strftime('%Y-%m-%d')
            timeframe = timeframes.get(self.general_config['timeframe'])      
            rates = get_historical_data(self.principal_symbol, timeframe, date_start, self.data_end)
            new_df = pd.DataFrame(rates)
            new_df['time'] = pd.to_datetime(new_df['time'], unit='s', errors='coerce')
            new_df = new_df.dropna(subset=['time'])
            extract_indicadores(self.principal_symbol, new_df)
            df_os = pd.concat([df_os, new_df], ignore_index=True).drop_duplicates(subset='time').sort_values('time').reset_index(drop=True)

        df_base1 = (
            pd.concat([df_is, df_os], ignore_index=True)
            .drop_duplicates(subset='time')
        )

        df_base = df_base1[
            df_base1['time'] >= datetime.strptime(
                self.data_start_is, '%Y-%m-%d'
            )
        ]  
        self.df_base = df_base.sort_values('time').set_index('time')
        corte = int(0.8 * len(self.df_base))
        self.df_train = self.df_base.iloc[:corte]
        self.df_valid = self.df_base.iloc[corte:]   


    def _reset_iteration_state(self):
        self.index = 0
        self.best_score = float('-inf')
        self.best_model_data = None
        self.peor_trade = 0.0
        self.info_score = {}
        self.dict_pips_best = {}


    def _run_training_loop(self):
        last_model_data = None

        for i in range(self.n_iterations):
            random.shuffle(self.nodos_close)
            print(f"\n--- Iteración {i+1}/{self.n_iterations} ---")
            self.index = i + 1
            model_data = self.train_iteration()
            last_model_data = model_data
            metrics = self.validate_iteration()
            self.calculate_score(metrics, model_data)

        if self.best_model_data is None:
            self.best_model_data = last_model_data

        return {
            "best_model_data": self.best_model_data,
            "best_score": self.best_score,
            "info_score": self.info_score,
            "last_model_data": last_model_data
        }


    def generate_wfo_windows(self):
        windows = []
        if self.df_base.empty:
            return windows

        idx_min = self.df_base.index.min()
        idx_max = self.df_base.index.max()
        current_test_end = idx_max

        while True:
            test_start = current_test_end - relativedelta(years=self.wfo_test_years)
            train_end = test_start
            train_start = train_end - relativedelta(years=self.wfo_train_years)

            if train_start < idx_min:
                break

            df_train = self.df_base[(self.df_base.index >= train_start) & (self.df_base.index < train_end)]
            df_valid = self.df_base[(self.df_base.index >= test_start) & (self.df_base.index <= current_test_end)]

            if len(df_train) > 0 and len(df_valid) > 0:
                windows.append({
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": current_test_end,
                    "df_train": df_train,
                    "df_valid": df_valid,
                })

            current_test_end = current_test_end - relativedelta(years=self.wfo_step_years)
            if current_test_end <= idx_min:
                break

        return windows


    def _max_drawdown_from_pips(self, lista_pips):
        if not lista_pips:
            return 0.0

        equity = np.cumsum(lista_pips)
        peak = np.maximum.accumulate(equity)
        drawdowns = peak - equity
        return float(np.max(drawdowns)) if len(drawdowns) else 0.0


    def _compact_metrics(self, metrics):
        if not metrics:
            return {}

        lista_pips = metrics.get("lista_pips", [])
        return {
            "iteracion": metrics.get("iteracion", 0),
            "score": float(metrics.get("score", 0.0)),
            "score_base": float(metrics.get("score_base", 0.0)),
            "winrate": float(metrics.get("winrate", 0.0)),
            "profit_factor": float(metrics.get("profit_factor", 0.0)),
            "expectancy": float(metrics.get("expectancy", 0.0)),
            "sharpe": float(metrics.get("sharpe", 0.0)),
            "min_close_prob": float(self.min_close_prob),
            "cantidad_operaciones": int(metrics.get("cantidad_operaciones", 0)),
            "sum_pips": float(metrics.get("sum_pips", 0.0)),
            "mas_perdidas_seguidas": int(metrics.get("mas_perdidas_seguidas", 0)),
            "peor_trade": float(metrics.get("peor_trade", 0.0)),
            "max_drawdown": self._max_drawdown_from_pips(lista_pips),
            "volumen_factor": float(metrics.get("volumen_factor", 0.0)),
            "reliability_factor": float(metrics.get("reliability_factor", 0.0)),
            "low_trade_penalty": float(metrics.get("low_trade_penalty", 0.0)),
        }
    
  
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
            get_nodes_by_label(self.principal_symbol, self.principal_symbol, mercado=None, label=self.other_algorithm)
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


    def calculate_trade_pips_with_sl_tp(self, open_price_open, bar_high, bar_low, bar_close, spread_open):
        """
        Ejecuta SL/TP de forma realista usando HIGH/LOW intrabar.
        Si ambos se tocan en la misma vela, usa resolución conservadora (SL primero).
        """
        sl_abs = abs(float(self.stop_loss_pips))
        tp_abs = abs(float(self.take_profit_pips))
        spread_pips = spread_open * self.point_size / self.pip_size

        if self.algorithm == 'UP':
            sl_price = open_price_open - (sl_abs * self.pip_size)
            tp_price = open_price_open + (tp_abs * self.pip_size)

            sl_hit = bar_low <= sl_price
            tp_hit = bar_high >= tp_price

            if sl_hit and tp_hit:
                return float(-sl_abs - spread_pips), "SL"
            if sl_hit:
                return float(-sl_abs - spread_pips), "SL"
            if tp_hit:
                return float(tp_abs - spread_pips), "TP"

            movement_pips = (bar_close - open_price_open) / self.pip_size
            return float(movement_pips - spread_pips), None

        # DOWN
        sl_price = open_price_open + (sl_abs * self.pip_size)
        tp_price = open_price_open - (tp_abs * self.pip_size)

        sl_hit = bar_high >= sl_price
        tp_hit = bar_low <= tp_price

        if sl_hit and tp_hit:
            return float(-sl_abs - spread_pips), "SL"
        if sl_hit:
            return float(-sl_abs - spread_pips), "SL"
        if tp_hit:
            return float(tp_abs - spread_pips), "TP"

        movement_pips = (open_price_open - bar_close) / self.pip_size
        return float(movement_pips - spread_pips), None


    def compute_future_target(self, df_prices, time_actual, current_price, horizon=5):
        """
        Target predictivo profesional: MFE - |MAE| sobre N velas futuras.
        Retorna (target, mfe, mae) en pips.
        """
        if df_prices is None or df_prices.empty:
            return 0.0, 0.0, 0.0

        if time_actual not in df_prices.index:
            return 0.0, 0.0, 0.0

        pos = int(df_prices.index.get_loc(time_actual))
        future = df_prices.iloc[pos + 1: pos + 1 + int(horizon)]
        if future.empty:
            return 0.0, 0.0, 0.0

        if 'high' not in future.columns or 'low' not in future.columns:
            return 0.0, 0.0, 0.0

        highs = future['high'].astype(float).values
        lows = future['low'].astype(float).values
        if len(highs) == 0 or len(lows) == 0:
            return 0.0, 0.0, 0.0

        cp = float(current_price)
        if self.algorithm == 'UP':
            mfe = float((np.max(highs) - cp) / self.pip_size)
            mae = float((np.min(lows) - cp) / self.pip_size)
        else:
            mfe = float((cp - np.min(lows)) / self.pip_size)
            mae = float((cp - np.max(highs)) / self.pip_size)

        target = float(mfe - abs(mae))
        return target, mfe, mae


    def compute_best_open_edge(self, nn, entry_open_signal, time_actual_np, time_actual, spread_open, market_ctx_open):
        """
        Estima el edge de apertura evaluando los nodos de cierre disponibles en el timestamp actual
        y devolviendo la mejor predicción cruda.
        """
        open_candidates = []

        for nodo in self.nodos_close:
            df_struct = self.COMBINED[("close", nodo["file"])]
            pos = df_struct["index_values"].searchsorted(time_actual_np)
            if pos == 0:
                continue

            if not self.cumple_condiciones_fast(df_struct, pos - 1, nodo["conditions"]):
                continue

            nodo_close = self.maping_close[nodo["key"]]
            market_ctx = self._get_market_context(df_struct, pos - 1, time_actual, spread_open)

            _, prob, pred = predict_from_inputs(
                nn,
                entry_open_signal,
                nodo_close,
                atr=market_ctx["atr"],
                adx=market_ctx["adx"],
                rsi=market_ctx["rsi"],
                hour=market_ctx["hour"],
                spread=market_ctx["spread"],
                stoch=market_ctx["stoch"],
                atr_open=market_ctx_open.get("atr", 0.0),
                adx_open=market_ctx_open.get("adx", 0.0),
                rsi_open=market_ctx_open.get("rsi", 0.0),
                stoch_open=market_ctx_open.get("stoch", 50.0),
                returns_1=market_ctx.get("returns_1", 0.0),
                volatility=market_ctx.get("volatility", 0.0),
                trend=market_ctx.get("trend", 0.0),
                return_raw=True,
            )

            open_candidates.append((nodo_close, float(prob), float(pred)))

        if not open_candidates:
            return None, None, None

        best_nodo, best_prob, best_pred = max(open_candidates, key=lambda x: x[2])
        return best_nodo, best_prob, best_pred


    def _get_value_from_struct(self, df_struct, row_idx, aliases, default=0.0):
        col_map = df_struct["col_map"]
        row = df_struct["values"][row_idx]

        for alias in aliases:
            if alias in col_map:
                v = row[col_map[alias]]
                if v is None or pd.isna(v):
                    continue
                try:
                    return float(v)
                except Exception:
                    continue
        return float(default)


    def _get_market_context(self, df_struct, row_idx, time_actual, spread_open):
        atr = self._get_value_from_struct(
            df_struct,
            row_idx,
            ["ATR_23", "ATR23", "ATR", "atr_23", "atr"],
            default=0.0
        )
        adx = self._get_value_from_struct(
            df_struct,
            row_idx,
            ["ADX_20", "ADX20", "ADX", "adx_20", "adx"],
            default=0.0
        )
        rsi = self._get_value_from_struct(
            df_struct,
            row_idx,
            ["RSI_21", "RSI21", "RSI", "rsi_21", "rsi"],
            default=0.0
        )
        stoch = self._get_value_from_struct(
            df_struct,
            row_idx,
            ["STOCH_14_3_SMA_3_SMA_pos0", "STOCH_14_3_3", "STOCH", "stoch"],
            default=50.0
        )

        # Cambio 3: Añadir features crudas de mercado
        # Calcular returns (corto plazo)
        close_curr = self._get_value_from_struct(df_struct, row_idx, ["close", "Close"], default=0.0)
        open_curr = self._get_value_from_struct(df_struct, row_idx, ["open", "Open"], default=0.0)
        high_curr = self._get_value_from_struct(df_struct, row_idx, ["high", "High"], default=0.0)
        low_curr = self._get_value_from_struct(df_struct, row_idx, ["low", "Low"], default=0.0)
        
        # Returns intraday (normalizado)
        returns_1 = (close_curr - open_curr) / max(abs(open_curr), 0.0001) if open_curr != 0 else 0.0
        
        # Volatility intraday (high-low range normalizado)
        volatility = (high_curr - low_curr) / max(abs(close_curr), 0.0001) if close_curr != 0 else 0.0
        
        # Trend simple: si close > open, uptrend (+1), else downtrend (-1)
        trend = 1.0 if close_curr > open_curr else -1.0

        return {
            "atr": atr,
            "adx": adx,
            "rsi": rsi,
            "stoch": stoch,
            "hour": float(time_actual.hour),
            "spread": float(spread_open),
            # Nuevas features de mercado
            "returns_1": float(returns_1),
            "volatility": float(volatility),
            "trend": float(trend),
        }
        

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

        mode = "train"
        path_data_red = f'output/{self.principal_symbol}/data_for_neuronal/data/data_{self.mercado}_{self.algorithm}.csv'
        model_path = f"output/{self.principal_symbol}/data_for_neuronal/model_trainer/model_{self.mercado}_{self.algorithm}.json"
        fallback_input_dim = 26  # 8+8 bits + 6 ctx cierre + 4 ctx apertura
        MIN_TRADES_TO_LEARN = 30
        epsilon = max(0.01, 0.10 * (0.97 ** max(0, self.index - 1)))

        if not os.path.exists(path_data_red):
            print("⚠️ No hay dataset aún, modo bootstrap")
            X, Y = None, None
        else:
            try:
                result = load_data(path_data_red)
                X, Y = result[0], result[1]  # Compatible con (X, Y, sample_weight)
                if len(X) < 20:
                    print("⚠️ Dataset muy pequeño, modo bootstrap")
                    X, Y = None, None
            except Exception as e:
                print(f"⚠️ Error cargando dataset: {e}")
                X, Y = None, None

        if os.path.exists(model_path):
            try:
                nn = load_trained_model(
                    model_path,
                    input_dim=(X.shape[1] if X is not None else fallback_input_dim)
                )
            except Exception as e:
                print(f"⚠️ Error cargando modelo, reiniciando: {e}")
                nn = BinaryNN(input_dim=fallback_input_dim, lr=0.01, target_loss=0.10)
        else:
            print("⚠️ No hay modelo aún, creando modelo inicial")
            nn = BinaryNN(input_dim=fallback_input_dim, lr=0.01, target_loss=0.10)

        bootstrap_mode = (X is None or len(X) < MIN_TRADES_TO_LEARN)
        if bootstrap_mode:
            print("⚠️ Dataset insuficiente, usando modo exploración")

        is_open = False
        open_price_open = 0.0
        spread_open = 0.0
        entry_red_open = ''
        cierre = 0
        market_ctx_open = {}
        time_open_trade = None
        pred_open_edge = None

        trade_samples = []

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
                        market_ctx = self._get_market_context(df_struct, pos - 1, time_actual, spread_open)
                        clase, prob, pred = predict_from_inputs(
                            nn,
                            entry_red_open,
                            nodo_close,
                            atr=market_ctx["atr"],
                            adx=market_ctx["adx"],
                            rsi=market_ctx["rsi"],
                            hour=market_ctx["hour"],
                            spread=market_ctx["spread"],
                            stoch=market_ctx["stoch"],
                            atr_open=market_ctx_open.get("atr", 0.0),
                            adx_open=market_ctx_open.get("adx", 0.0),
                            rsi_open=market_ctx_open.get("rsi", 0.0),
                            stoch_open=market_ctx_open.get("stoch", 50.0),
                            # Cambio 3: Pasar market features
                            returns_1=market_ctx.get("returns_1", 0.0),
                            volatility=market_ctx.get("volatility", 0.0),
                            trend=market_ctx.get("trend", 0.0),
                            return_raw=True,
                        )

                        if bootstrap_mode:
                            clase = 1
                            prob = 1.0
                            pred = 1.0
                        elif nn.model is None:
                            clase = random.randint(0, 1)
                            pred = 0.1 if clase == 1 else -0.1
                            prob = abs(pred)
                        elif random.random() < epsilon:
                            clase = random.randint(0, 1)
                            pred = 0.1 if clase == 1 else -0.1
                            prob = abs(pred)

                        valid_closes.append((nodo_close, int(clase), float(prob), float(pred), market_ctx))

                bar_high = float(getattr(row, 'high', open_price))
                bar_low = float(getattr(row, 'low', open_price))
                bar_close = float(getattr(row, 'close', open_price))
                trade_pips, forced_intrabar_reason = self.calculate_trade_pips_with_sl_tp(
                    open_price_open,
                    bar_high,
                    bar_low,
                    bar_close,
                    spread_open,
                )

                cerrar = False
                reason = None

                if forced_intrabar_reason == "SL":
                    cerrar = True
                    reason = "SL"
                elif forced_intrabar_reason == "TP":
                    cerrar = True
                    reason = "TP"
                elif trade_pips < self.hard_stop_pips:
                    cerrar = True
                    reason = "HARD_STOP"
                elif trade_pips < self.max_loss_close_pips:
                    cerrar = True
                    reason = "RISK_LOSS"
                elif cierre > self.max_trade_duration:
                    cerrar = True
                    reason = "TIMEOUT"
                elif bootstrap_mode:
                    cerrar = (cierre > random.randint(5, 50))
                    if cerrar:
                        reason = "BOOTSTRAP"
                elif valid_closes:
                    # Selección por ranking de edge real (pred), no por probabilidad/threshold legado
                    best_close = max(valid_closes, key=lambda x: x[3])  # (nodo, clase, prob, pred, ctx)
                    best_pred = best_close[3]

                    if pred_open_edge is None:
                        pred_open_edge = best_pred

                    delta_pred = best_pred - pred_open_edge
                    close_by_low_edge = best_pred < self.edge_threshold
                    close_by_decay = best_pred < (pred_open_edge * self.edge_decay_factor)
                    close_by_delta = delta_pred < self.edge_delta_threshold
                    cerrar = bool(close_by_low_edge or close_by_decay or close_by_delta)
                    if cerrar:
                        reason = "MODEL"

                if cierre > 200:
                    print(f"⚠️ trade largo detectado: {cierre}")

                if cerrar:
                    if mode == "train":
                        # Cambio 2 + 5: Guardar TODAS las decisiones de cierre con executed flag,
                        # pero priorizar la mejor decisión cuando hay multi-closes
                        if valid_closes:
                            # Cambio 5: Identificar el mejor nodo por predicción cruda
                            best_close = max(valid_closes, key=lambda x: x[3])
                            best_nodo, _, _, best_pred, _ = best_close
                            target_future, mfe_future, mae_future = self.compute_future_target(
                                self.df_train,
                                time_actual,
                                open_price,
                                horizon=self.future_horizon_bars,
                            )
                            if trade_pips < -20:
                                target_future -= 10.0
                            
                            # Guardar TODAS las decisiones, pero marcar cual fue ejecutada (la mejor)
                            for nodo_close_selected, clase_selected, prob_selected, pred_selected, market_ctx_selected in valid_closes:
                                is_best = (nodo_close_selected == best_nodo and pred_selected == best_pred)
                                trade_samples.append({
                                    "open_signal": entry_red_open,
                                    "close_signal": nodo_close_selected,
                                    "time_open": str(time_open_trade) if time_open_trade is not None else "",
                                    "time_close": str(time_actual),
                                    "atr": market_ctx_selected["atr"],
                                    "adx": market_ctx_selected["adx"],
                                    "rsi": market_ctx_selected["rsi"],
                                    "stoch": market_ctx_selected["stoch"],
                                    "hour": market_ctx_selected["hour"],
                                    "spread": market_ctx_selected["spread"],
                                    "atr_open": market_ctx_open.get("atr", 0.0),
                                    "adx_open": market_ctx_open.get("adx", 0.0),
                                    "rsi_open": market_ctx_open.get("rsi", 0.0),
                                    "stoch_open": market_ctx_open.get("stoch", 50.0),
                                    "profit": target_future,
                                    "profit_real": trade_pips,
                                    "mfe_future": mfe_future,
                                    "mae_future": mae_future,
                                    # Nueva información: flags de decisión
                                    # Con Cambio 5: solo la mejor es realmente ejecutada
                                    "executed": (reason in ["MODEL", "BOOTSTRAP"] and is_best),
                                    "reason": reason,  # Razón del cierre (SL/TP/MODEL/etc)
                                })
                        
                        # Si no hay valid_closes (e.g., timeout, SL, TP), crear una entrada de cierre forzado
                        if not valid_closes:
                            target_future, mfe_future, mae_future = self.compute_future_target(
                                self.df_train,
                                time_actual,
                                open_price,
                                horizon=self.future_horizon_bars,
                            )
                            if trade_pips < -20:
                                target_future -= 10.0
                            trade_samples.append({
                                "open_signal": entry_red_open,
                                "close_signal": "FORCED",
                                "time_open": str(time_open_trade) if time_open_trade is not None else "",
                                "time_close": str(time_actual),
                                "atr": market_ctx_open.get("atr", 0.0),
                                "adx": market_ctx_open.get("adx", 0.0),
                                "rsi": market_ctx_open.get("rsi", 0.0),
                                "stoch": market_ctx_open.get("stoch", 50.0),
                                "hour": float(time_actual.hour),
                                "spread": float(spread_open),
                                "atr_open": market_ctx_open.get("atr", 0.0),
                                "adx_open": market_ctx_open.get("adx", 0.0),
                                "rsi_open": market_ctx_open.get("rsi", 0.0),
                                "stoch_open": market_ctx_open.get("stoch", 50.0),
                                "profit": target_future,
                                "profit_real": trade_pips,
                                "mfe_future": mfe_future,
                                "mae_future": mae_future,
                                "executed": reason in ["SL", "TP", "TIMEOUT", "RISK_LOSS", "HARD_STOP"],  # Forzado pero ejecutado
                                "reason": reason,
                            })

                    is_open = False
                    time_open_trade = None
                    pred_open_edge = None
                    print(f"CLOSE [{reason}] {time_actual} | pips={trade_pips:.2f} | dur={cierre}")
                    cierre = 0

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

                        if bootstrap_mode and random.random() < 0.02:
                            cumple_alguno = True

                            if symbol == self.list_symbols[-1]:
                                open_price_open = open_price
                                spread_open = getattr(row, 'spread', 0.0)
                                is_open = True
                                time_open_trade = time_actual
                                nodo_open = self.maping_open[nodo["key"]]
                                entry_red_open = nodo_open
                                market_ctx_open = self._get_market_context(df_struct, pos - 1, time_actual, spread_open)
                                pred_open_edge = None

                                # Filtro de entrada por edge (evita aperturas con señal débil)
                                if not bootstrap_mode and nn.model is not None:
                                    _, _, open_pred = self.compute_best_open_edge(
                                        nn,
                                        entry_red_open,
                                        time_actual_np,
                                        time_actual,
                                        spread_open,
                                        market_ctx_open,
                                    )
                                    if open_pred is not None and open_pred < self.min_open_edge:
                                        is_open = False
                                        pred_open_edge = None
                                        cumple_alguno = False
                                    else:
                                        pred_open_edge = open_pred if open_pred is not None else 0.0

                            break

                        if self.cumple_condiciones_fast(df_struct, pos - 1, nodo["conditions"]):

                            cumple_alguno = True

                            if symbol == self.list_symbols[-1]:
                                open_price_open = open_price
                                spread_open = getattr(row, 'spread', 0.0)
                                is_open = True
                                time_open_trade = time_actual
                                nodo_open = self.maping_open[nodo["key"]]
                                entry_red_open = nodo_open
                                market_ctx_open = self._get_market_context(df_struct, pos - 1, time_actual, spread_open)
                                pred_open_edge = None

                                # Filtro de entrada por edge (evita aperturas con señal débil)
                                if not bootstrap_mode and nn.model is not None:
                                    _, _, open_pred = self.compute_best_open_edge(
                                        nn,
                                        entry_red_open,
                                        time_actual_np,
                                        time_actual,
                                        spread_open,
                                        market_ctx_open,
                                    )
                                    if open_pred is not None and open_pred < self.min_open_edge:
                                        is_open = False
                                        pred_open_edge = None
                                        cumple_alguno = False
                                    else:
                                        pred_open_edge = open_pred if open_pred is not None else 0.0

                            break

                    if not cumple_alguno:
                        break

        trade_dataset_dir = f'output/{self.principal_symbol}/data_for_neuronal/trade_dataset'
        os.makedirs(trade_dataset_dir, exist_ok=True)
        trade_dataset_path = f'{trade_dataset_dir}/trade_dataset_{self.mercado}_{self.algorithm}.csv'
        trade_df = pd.DataFrame(trade_samples)
        if not trade_df.empty:
            exists = os.path.exists(trade_dataset_path)
            trade_df.to_csv(
                trade_dataset_path,
                mode='a' if exists else 'w',
                header=not exists,
                index=False
            )
        print(f"Trade dataset actualizado en: {trade_dataset_path} (+{len(trade_samples)} samples)")

        # =========================
        # REENTRENAMIENTO
        # =========================

        config = {
            "general": self.general_config,
            "symbol": self.config_symbol,
            "principal_symbol": self.principal_symbol
        }
        data_for_neuronal(
            config,
            self.mercado,
            self.algorithm,
            dict_pips_best=None,
            trade_samples=trade_samples
        )

        if os.path.exists(path_data_red):
            try:
                result = load_data(path_data_red)
                X, Y = result[0], result[1]  # Compatible con (X, Y, sample_weight)
                sample_weight = result[2] if len(result) > 2 else None
            except Exception as e:
                print(f"⚠️ Dataset inválido → se omite entrenamiento: {e}")
                X, Y = None, None
                sample_weight = None

            if X is None or Y is None:
                print("⚠️ No se pudo cargar dataset válido en esta iteración")
            else:
                print("Datos cargados:")
                print("X shape:", X.shape)
                print("Y shape:", Y.shape)

                # Continuar entrenamiento desde el modelo existente (no resetear)
                if len(X) >= MIN_TRADES_TO_LEARN:
                    if nn.model is None:
                        nn.fit(X, Y, epochs=20000, batch_size=32, sample_weight=sample_weight)
                    else:
                        y_reg = Y.ravel().astype(np.float32)
                        if sample_weight is not None:
                            nn.model.fit(X, y_reg, sample_weight=sample_weight)
                        else:
                            nn.model.fit(X, y_reg)
                        payload = _serialize_xgb_to_base64(nn.model)
                        nn.W1 = np.array([f"__XGB__{payload}"], dtype=object)
                    nn.input_dim = X.shape[1]
                else:
                    print(f"⚠️ Dataset insuficiente para entrenar (>={MIN_TRADES_TO_LEARN} requerido)")
        else:
            print("⚠️ No hay dataset, no se entrena en esta iteración")

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

        tmp_path = model_path + ".tmp"
        with open(tmp_path, 'w') as f:
            json.dump(model_data, f, indent=4)
        os.replace(tmp_path, model_path)

        print(f"Modelo guardado en '{model_path}'")

        return model_data
    
  
    def validate_iteration(self):

        path_data_red = f'output/{self.principal_symbol}/data_for_neuronal/data/data_{self.mercado}_{self.algorithm}.csv'
        model_path = f'output/{self.principal_symbol}/data_for_neuronal/model_trainer/model_{self.mercado}_{self.algorithm}.json'
        fallback_input_dim = 26

        input_dim = fallback_input_dim
        if os.path.exists(path_data_red):
            try:
                result = load_data(path_data_red)
                X, Y = result[0], result[1]  # Compatible con (X, Y, sample_weight)
                if len(X) > 0:
                    input_dim = X.shape[1]
            except Exception:
                pass

        if os.path.exists(model_path):
            try:
                nn = load_trained_model(
                    model_path,
                    input_dim=input_dim
                )
            except Exception as e:
                print(f"⚠️ Modelo corrupto detectado, reset: {e}")
                nn = BinaryNN(input_dim=input_dim, lr=0.01, target_loss=0.10)
                nn.model = None
        else:
            nn = BinaryNN(input_dim=input_dim, lr=0.01, target_loss=0.10)
            nn.model = None

        is_open = False
        open_price_open = 0.0
        spread_open = 0.0
        entry_red_open = ''
        cierre = 0
        time_comienzo = None
        market_ctx_open = {}
        pred_open_edge = None

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
                valid_closes = []
                for nodo in self.nodos_close:
                    df_struct = self.COMBINED[("close", nodo["file"])]
                    pos = df_struct["index_values"].searchsorted(time_actual_np)
                    if pos == 0:
                        continue

                    if self.cumple_condiciones_fast(df_struct, pos - 1, nodo["conditions"]):

                        nodo_close = self.maping_close[nodo["key"]]
                        market_ctx = self._get_market_context(df_struct, pos - 1, time_actual, spread_open)
                        clase, prob, pred = predict_from_inputs(
                            nn,
                            entry_red_open,
                            nodo_close,
                            atr=market_ctx["atr"],
                            adx=market_ctx["adx"],
                            rsi=market_ctx["rsi"],
                            hour=market_ctx["hour"],
                            spread=market_ctx["spread"],
                            stoch=market_ctx["stoch"],
                            atr_open=market_ctx_open.get("atr", 0.0),
                            adx_open=market_ctx_open.get("adx", 0.0),
                            rsi_open=market_ctx_open.get("rsi", 0.0),
                            stoch_open=market_ctx_open.get("stoch", 50.0),
                            # Cambio 3: Pasar market features
                            returns_1=market_ctx.get("returns_1", 0.0),
                            volatility=market_ctx.get("volatility", 0.0),
                            trend=market_ctx.get("trend", 0.0),
                            return_raw=True,
                        )
                        if nn.model is None:
                            clase = random.randint(0, 1)
                            pred = 0.1 if clase == 1 else -0.1
                            prob = abs(pred)
                        valid_closes.append((int(clase), float(prob), float(pred)))

                bar_high = float(getattr(row, 'high', open_price))
                bar_low = float(getattr(row, 'low', open_price))
                bar_close = float(getattr(row, 'close', open_price))
                trade_pips, forced_intrabar_reason = self.calculate_trade_pips_with_sl_tp(
                    open_price_open,
                    bar_high,
                    bar_low,
                    bar_close,
                    spread_open,
                )

                cerrar = False
                reason = None
                if forced_intrabar_reason == "SL":
                    cerrar = True
                    reason = "SL"
                elif forced_intrabar_reason == "TP":
                    cerrar = True
                    reason = "TP"
                elif trade_pips < self.hard_stop_pips:
                    cerrar = True
                    reason = "HARD_STOP"
                elif trade_pips < self.max_loss_close_pips:
                    cerrar = True
                    reason = "RISK_LOSS"
                elif cierre > self.max_trade_duration:
                    cerrar = True
                    reason = "TIMEOUT"
                elif valid_closes:
                    best_close = max(valid_closes, key=lambda x: x[2])  # (clase, prob, pred)
                    best_pred = best_close[2]

                    if pred_open_edge is None:
                        pred_open_edge = best_pred

                    delta_pred = best_pred - pred_open_edge
                    close_by_low_edge = best_pred < self.edge_threshold
                    close_by_decay = best_pred < (pred_open_edge * self.edge_decay_factor)
                    close_by_delta = delta_pred < self.edge_delta_threshold
                    cerrar = bool(close_by_low_edge or close_by_decay or close_by_delta)
                    if cerrar:
                        reason = "MODEL"
                if cerrar:
                    if trade_pips < 0:
                        perdidas_seguidas += 1
                    else:
                        perdidas_seguidas = 0

                    is_open = False
                    pred_open_edge = None
                    cantidad_operaciones += 1
                    sum_pips += trade_pips
                    lista_pips.append(trade_pips)

                    if trade_pips > 0:
                        operaciones_acertadas += 1
                        ganancia_bruta += trade_pips
                    else:
                        operaciones_perdedoras += 1
                        perdida_bruta += abs(trade_pips)

                    print(f"CLOSE [{reason}] {time_comienzo} -> {time_actual}: {sum_pips:.2f}")
                    cierre = 0

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
                                market_ctx_open = self._get_market_context(df_struct, pos - 1, time_actual, spread_open)
                                pred_open_edge = None

                                # Filtro de entrada por edge en validación
                                if nn.model is not None:
                                    _, _, open_pred = self.compute_best_open_edge(
                                        nn,
                                        entry_red_open,
                                        time_actual_np,
                                        time_actual,
                                        spread_open,
                                        market_ctx_open,
                                    )
                                    if open_pred is not None and open_pred < self.min_open_edge:
                                        is_open = False
                                        pred_open_edge = None
                                        cumple_alguno = False
                                    else:
                                        pred_open_edge = open_pred if open_pred is not None else 0.0

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

        max_drawdown = self._max_drawdown_from_pips(lista_pips)

        return {
            "winrate": winrate,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "sharpe": sharpe,
            "cantidad_operaciones": cantidad_operaciones,
            "sum_pips": sum_pips,
            "lista_pips": lista_pips,
            "mas_perdidas_seguidas": mas_perdidas_seguidas,
            "max_drawdown": max_drawdown,
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
        max_drawdown = float(metrics.get("max_drawdown", 0.0))
        dd_penalty = max_drawdown * 0.1
        std_pips = float(np.std(metrics["lista_pips"])) if metrics["lista_pips"] else 0.0
        # Penalización de consistencia suavizada para no castigar en exceso sistemas con winners grandes
        consistency = 1.0 / (1.0 + (std_pips * 0.01))

        score = score_base * volumen_factor * reliability_factor * low_trade_penalty
        score = (score - dd_penalty) * consistency

        print(
            f"Score calculado: {score:.4f} | base={score_base:.4f} | ops={n_ops} | "
            f"vol={volumen_factor:.4f} | rel={reliability_factor:.4f} | "
            f"low_ops={low_trade_penalty:.4f} | dd_penalty={dd_penalty:.4f} | "
            f"consistency={consistency:.4f}"
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
            metrics["dd_penalty"] = dd_penalty
            metrics["consistency"] = consistency
            self.info_score = self._compact_metrics(metrics)
            print(f"Nuevo mejor modelo encontrado con score: {score:.4f}")
        return score


    def run(self):
        model_path = f'output/{self.principal_symbol}/data_for_neuronal/model_trainer/model_{self.mercado}_{self.algorithm}.json'
        score_path = f'output/{self.principal_symbol}/data_for_neuronal/best_score/score_{self.mercado}_{self.algorithm}.json'

        if not self.use_wfo:
            self._reset_iteration_state()
            result = self._run_training_loop()
            with open(model_path, 'w') as f:
                json.dump(result["best_model_data"], f, indent=4)
            with open(score_path, 'w') as f:
                json.dump({
                    "mode": "single_split",
                    "winner": result["info_score"]
                }, f, indent=4)
            print(f'tiempo total: {time.time() - self.comienzo:.2f} segundos')
            return

        windows = self.generate_wfo_windows()
        if not windows:
            print("No se pudieron generar ventanas WFO. Se usa split único 80/20.")
            self._reset_iteration_state()
            result = self._run_training_loop()
            with open(model_path, 'w') as f:
                json.dump(result["best_model_data"], f, indent=4)
            with open(score_path, 'w') as f:
                json.dump({
                    "mode": "single_split_fallback",
                    "winner": result["info_score"]
                }, f, indent=4)
            print(f'tiempo total: {time.time() - self.comienzo:.2f} segundos')
            return

        wfo_scores = []
        wfo_drawdowns = []
        latest_window_model_data = None
        latest_window_metrics = {}
        latest_test_end = None
        wfo_windows_summary = []

        for i, window in enumerate(windows, start=1):
            self.df_train = window["df_train"]
            self.df_valid = window["df_valid"]
            self.pip_size, self.point_size = get_pip_and_point_size(
                self.principal_symbol,
                self.df_train['open'] if 'open' in self.df_train.columns else self.df_valid['open']
            )

            print(
                f"\n=== WFO Ventana {i}/{len(windows)} | "
                f"Train: {window['train_start']} -> {window['train_end']} | "
                f"Test: {window['test_start']} -> {window['test_end']} ==="
            )

            self._reset_iteration_state()
            result = self._run_training_loop()

            info = dict(result["info_score"]) if result["info_score"] else {}
            info.update({
                "window": i,
                "train_start": str(window["train_start"]),
                "train_end": str(window["train_end"]),
                "test_start": str(window["test_start"]),
                "test_end": str(window["test_end"]),
                "best_score_window": result["best_score"],
            })
            wfo_scores.append(float(result["best_score"]))
            wfo_drawdowns.append(float(info.get("max_drawdown", 0.0)))
            wfo_windows_summary.append(info)

            current_test_end = window["test_end"]
            if latest_test_end is None or current_test_end > latest_test_end:
                latest_test_end = current_test_end
                latest_window_model_data = result["best_model_data"]
                latest_window_metrics = info

        if latest_window_model_data is None:
            print("WFO sin mejor modelo válido. Se mantiene el modelo actual.")
            print(f'tiempo total: {time.time() - self.comienzo:.2f} segundos')
            return

        mean_score = float(np.mean(wfo_scores)) if wfo_scores else 0.0
        std_score = float(np.std(wfo_scores)) if wfo_scores else 0.0
        min_score = float(np.min(wfo_scores)) if wfo_scores else 0.0
        max_score = float(np.max(wfo_scores)) if wfo_scores else 0.0
        robust_score = mean_score - std_score
        mean_drawdown = float(np.mean(wfo_drawdowns)) if wfo_drawdowns else 0.0
        max_drawdown = float(np.max(wfo_drawdowns)) if wfo_drawdowns else 0.0

        with open(model_path, 'w') as f:
            json.dump(latest_window_model_data, f, indent=4)
        with open(score_path, 'w') as f:
            json.dump({
                "mode": "wfo",
                "selection_rule": "most_recent_test_end",
                "wfo_config": {
                    "train_years": self.wfo_train_years,
                    "test_years": self.wfo_test_years,
                    "step_years": self.wfo_step_years,
                    "iterations_per_window": self.n_iterations,
                    "windows": len(windows)
                },
                "robustness": {
                    "mean_score": mean_score,
                    "std_score": std_score,
                    "robust_score": robust_score,
                    "min_score": min_score,
                    "max_score": max_score,
                    "mean_max_drawdown": mean_drawdown,
                    "max_drawdown": max_drawdown,
                    "window_scores": wfo_scores
                },
                "latest_window": latest_window_metrics,
                "windows_summary": wfo_windows_summary,
                "winner": latest_window_metrics
            }, f, indent=4)
        print(f'tiempo total: {time.time() - self.comienzo:.2f} segundos')
    
       
      
if __name__ == "__main__":
    inn = time.time()
    backtester = Backtester('AUDCAD', 'Asia', 'UP', date_end=None) 
    backtester.run()
    print(f'segundos {time.time()-inn}')
    