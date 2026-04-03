# ============================================================
# IMPORTS
# ============================================================

import os
import sys
import json
import time
import ast
import operator

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import torch

from datetime import datetime
from dateutil.relativedelta import relativedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.routes.peticiones import get_historical_data, get_timeframes
from src.db.query import get_nodes_by_label
from src.utils.indicadores_for_crossing import extract_indicadores
from src.utils.common_functions import get_previous_4_6, hora_en_mercado
from src.neuronal.entrenar import (
    EXTRA_FEATURE_COLUMNS,
    MIN_TRAINING_SAMPLES,
    load_trained_model,
    predict_from_inputs,
    BinaryNN,
    has_minimum_training_data,
    load_data,
    save_trained_model,
    save_ensemble_model,
    get_embedding_vocab_sizes,
    validate_embedding_vocab,
)
from src.signals.event_generator import EVENT_FEATURE_COLUMNS, add_event_features, has_entry_event


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


def _sanitize_for_json(value):
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        numeric_value = float(value)
        return numeric_value if np.isfinite(numeric_value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.generic):
        return _sanitize_for_json(value.item())
    return value

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
        self.iteration_candidates = []
        self.ensemble_model_entries = []
        self.dict_pips_best = {}
        self.stop_loss = 20
        self.take_profit = 150
        self.max_holding = 120
        self.min_model_holding = 15
        self.close_confirmation_bars = 2
        self.close_threshold_floor = 0.60
        self.close_threshold_ceiling = 0.70
        self.close_threshold_history = []
        self.close_threshold_history_window = 5
        self.threshold_validation_split_ratio = 0.50
        self.validation_window_months = 6
        self.min_open_symbol_confirmations = 4
        self.train_b_fraction = 0.30
        self.threshold_min_calibration_trades = 15
        self.threshold_min_selected_trades = 10
        self.robust_score_weights = {
            "expectancy": 2.0,
            "profit_factor": 1.5,
            "sharpe": 2.0,
            "winrate": 1.0,
            "calmar_ratio": 1.8,
            "sortino_ratio": 1.5,
        }
        self.robust_trade_penalty_center = 25
        self.robust_trade_penalty_scale = 12
        self.robust_min_avg_bars = 12
        self.ensemble_top_k = 3
        self.ensemble_decay_factor = 0.9
        self.label_percentile_threshold = 65
        self.label_min_context_samples = 25
        self.label_volatility_floor_pips = 5.0
        self.min_winning_iteration = 8
        self.max_iterations = 15
        self.early_stopping_patience = 8
        self.no_improve_iterations = 0
        self.min_trades_score = 30
        self.trade_reliability_k = 40
        self.low_trade_penalty_power = 1.75
        self.selection_min_trades = 30
        self.selection_min_profit_factor = 1.15
        self.selection_min_expectancy = 0.25
        self.selection_min_winrate = 0.45
        self.selection_max_drawdown_ratio = 1.50
        self.selection_max_losing_streak = 7
        self.selection_min_months = 4
        self.selection_min_positive_month_ratio = 0.50
        self.selection_min_positive_quarter_ratio = 0.50
        self.selection_strong_candidate_min_trades = 20
        self.selection_strong_candidate_min_profit_factor = 1.35
        self.selection_strong_candidate_min_expectancy = 4.0
        self.selection_strong_candidate_min_winrate = 0.40
        self.selection_strong_candidate_max_drawdown_ratio = 1.75
        self.selection_strong_candidate_max_losing_streak = 8
        self.selection_strong_candidate_min_positive_month_ratio = 0.50
        self.selection_strong_candidate_min_positive_quarter_ratio = 0.50
        self.selection_strong_candidate_min_deployment_score = 12.0
        self.selection_strong_candidate_allowed_reasons = {
            "low_trades",
            "low_winrate",
            "low_positive_month_ratio",
        }
        self.train_b_label_blocks = 3
        self.min_training_samples = MIN_TRAINING_SAMPLES
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
            self.df_train_A['open'] if 'open' in self.df_train_A.columns else self.df_valid['open']
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
        self.min_open_symbol_confirmations = int(
            self.general_config.get('MinOpenSymbolConfirmations', self.min_open_symbol_confirmations)
        )
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
        df_base = self.add_market_features(df_base)
        self.split_temporal_windows(df_base)


    def split_temporal_windows(self, df_base):
        if df_base.empty:
            self.df_train_A = df_base
            self.df_train_B = df_base
            self.df_valid = df_base
            return

        validation_end = df_base.index.max()
        validation_start = validation_end - relativedelta(months=self.validation_window_months)
        df_valid = df_base[df_base.index >= validation_start]
        df_pre_validation = df_base[df_base.index < validation_start]

        if df_valid.empty:
            fallback_cut = max(1, int(len(df_base) * 0.15))
            df_valid = df_base.iloc[-fallback_cut:]
            df_pre_validation = df_base.iloc[:-fallback_cut]

        if df_pre_validation.empty:
            split_idx = max(1, len(df_base) - len(df_valid))
            self.df_train_A = df_base.iloc[:split_idx]
            self.df_train_B = df_base.iloc[:0].copy()
            self.df_valid = df_valid
            return

        train_b_size = int(len(df_pre_validation) * self.train_b_fraction)
        train_b_size = max(1, train_b_size)
        train_b_size = min(train_b_size, len(df_pre_validation))
        split_idx = len(df_pre_validation) - train_b_size

        if split_idx <= 0:
            split_idx = max(1, len(df_pre_validation) - 1)

        self.df_train_A = df_pre_validation.iloc[:split_idx]
        self.df_train_B = df_pre_validation.iloc[split_idx:]
        self.df_valid = df_valid

        print(
            "Ventanas temporales preparadas | "
            f"train_A: {self.df_train_A.index.min()} -> {self.df_train_A.index.max()} ({len(self.df_train_A)}) | "
            f"train_B: {self.df_train_B.index.min() if not self.df_train_B.empty else 'empty'} -> "
            f"{self.df_train_B.index.max() if not self.df_train_B.empty else 'empty'} ({len(self.df_train_B)}) | "
            f"valid: {self.df_valid.index.min()} -> {self.df_valid.index.max()} ({len(self.df_valid)})"
        )


    def add_market_features(self, df):
        df = df.copy()
        df['ret_1'] = df['close'] - df['open']
        df['ret_3'] = df['close'] - df['close'].shift(3)
        df['ret_10'] = df['close'] - df['close'].shift(10)
        df['range_1'] = df['high'] - df['low']
        df['ma_5'] = df['close'].rolling(5).mean()
        df['ma_10'] = df['close'].rolling(10).mean()
        df['ma_20'] = df['close'].rolling(20).mean()
        df['trend'] = df['close'] - df['ma_5']
        df['trend_10'] = df['close'] - df['ma_10']
        df['trend_20'] = df['close'] - df['ma_20']
        df['vol'] = df['close'].rolling(5).std()
        df['vol_10'] = df['close'].rolling(10).std()
        df['vol_20'] = df['close'].rolling(20).std()
        df['zscore_20'] = (df['close'] - df['ma_20']) / (df['vol_20'] + 1e-8)
        df['momentum_ratio'] = df['ret_3'] / (df['vol_10'] + 1e-8)
        df = add_event_features(df)
        return df.dropna()


    def get_market_features(self, row):
        if isinstance(row, pd.Series):
            return np.array([
                float(row.get(column, 0.0))
                for column in EXTRA_FEATURE_COLUMNS
            ], dtype=np.float32)

        return np.array([
            float(getattr(row, column, 0.0))
            for column in EXTRA_FEATURE_COLUMNS
        ], dtype=np.float32)


    def is_entry_event_active(self, row):
        return has_entry_event(row, self.algorithm)


    def get_trade_risk_limits(self, row):
        adaptive_stop = max(self.stop_loss, 1.5 * float(getattr(row, 'vol_10', 0.0)) / self.pip_size)
        adaptive_take = max(self.take_profit, 2.0 * float(getattr(row, 'vol_10', 0.0)) / self.pip_size)
        return adaptive_stop, adaptive_take
    
  
    def actualizar_dict(self, principal, nuevo_dict):
        """Actualiza dict_pips_best con nuevos datos."""
        for k, v in nuevo_dict.items():
            if k not in principal:
                principal[k] = v  # Guardar valor real
            else:
                # Promedio ponderado (como en TradingEngine.record_trade)
                principal[k] = (
                    principal[k] + v 
                )
        
        return principal 


    def parsear_nodos(self, nodos):
        if not nodos:
            return []
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

        open_mapping_path = f'output/{self.principal_symbol}/data_for_neuronal/maping/maping_open_{self.mercado}_{self.algorithm}.json'
        close_mapping_path = f'output/{self.principal_symbol}/data_for_neuronal/maping/maping_close_{self.mercado}_{self.algorithm}.json'

        self.maping_open = {}
        self.maping_close = {}

        if os.path.exists(open_mapping_path):
            with open(open_mapping_path, 'r') as file:
                self.maping_open = json.load(file)

        if os.path.exists(close_mapping_path):
            with open(close_mapping_path, 'r') as file:
                self.maping_close = json.load(file)
    
  
    def load_nodes(self):

        self.dict_nodos = {}

        for i, symbol in enumerate(self.list_symbols):
            self.dict_nodos[symbol] = self.parsear_nodos(
                get_nodes_by_label(self.principal_symbol, symbol, self.mercado, self.algorithm) or []
            )

        self.nodos_close = self.parsear_nodos(
            get_nodes_by_label(self.principal_symbol, self.principal_symbol, mercado=None, label=self.other_algorithm) or []
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


    def simulate_training_trade(
        self,
        df,
        entry_idx,
        entry_red_opens,
        open_price_open,
        spread_open,
        current_stop_loss,
        current_take_profit,
        dict_pips,
        list_keys,
    ):
        last_idx = entry_idx

        for offset in range(1, self.max_holding + 1):
            current_idx = entry_idx + offset
            if current_idx >= len(df):
                break

            current_row = df.iloc[current_idx]
            current_time = df.index[current_idx]
            current_time_np = np.datetime64(current_time)
            current_hour = format(current_time.hour, "05b")
            current_pips = self.calculate_trade_pips(
                open_price_open,
                current_row.open,
                spread_open,
            )
            last_idx = current_idx

            matching_close_nodes = []
            if offset >= self.min_model_holding:
                for nodo in self.nodos_close:
                    df_struct = self.COMBINED[("close", nodo["file"])]
                    pos = df_struct["index_values"].searchsorted(current_time_np)
                    if pos == 0:
                        continue

                    if self.cumple_condiciones_fast(df_struct, pos - 1, nodo["conditions"]):
                        mapped_close = self.maping_close.get(nodo["key"])
                        if mapped_close is not None:
                            matching_close_nodes.append(mapped_close)

            if matching_close_nodes:
                for entry_red_open in entry_red_opens:
                    for nodo_close_selected in matching_close_nodes:
                        key_pru = f'{entry_red_open}_{nodo_close_selected}_{current_hour}_{current_time}'
                        if key_pru in list_keys:
                            continue
                        list_keys.add(key_pru)

                        key = f'{entry_red_open}_{nodo_close_selected}_{current_hour}'
                        if key not in dict_pips:
                            dict_pips[key] = [current_pips]
                        else:
                            dict_pips[key].append(current_pips)

            if (
                current_pips <= -current_stop_loss or
                current_pips >= current_take_profit or
                offset >= self.max_holding
            ):
                break

        return last_idx


    def collect_training_pips_map(self, df):
        list_keys = set()
        dict_pips = {}
        row_pos = 0
        total_rows = len(df)

        while row_pos < total_rows:
            row = df.iloc[row_pos]
            time_actual = df.index[row_pos]
            time_actual_np = np.datetime64(time_actual)

            if not self.horas_mercado[time_actual.hour]:
                row_pos += 1
                continue

            if not self.is_entry_event_active(row):
                row_pos += 1
                continue

            open_price = row.open

            entry_red_opens = self.resolve_entry_open_nodes(time_actual_np)

            if not entry_red_opens:
                row_pos += 1
                continue

            spread_open = getattr(row, 'spread', 0.0)
            current_stop_loss, current_take_profit = self.get_trade_risk_limits(row)
            exit_idx = self.simulate_training_trade(
                df=df,
                entry_idx=row_pos,
                entry_red_opens=entry_red_opens,
                open_price_open=open_price,
                spread_open=spread_open,
                current_stop_loss=current_stop_loss,
                current_take_profit=current_take_profit,
                dict_pips=dict_pips,
                list_keys=list_keys,
            )
            row_pos = max(exit_idx + 1, row_pos + 1)

        return {
            key: float(np.mean(values))
            for key, values in dict_pips.items()
            if values
        }


    def build_walk_forward_dataset(self, df_target, dict_pips_seed):
        if df_target.empty:
            return self.build_dataset_from_df(df_target, dict_pips_seed)

        block_count = max(1, min(self.train_b_label_blocks, len(df_target)))
        split_points = np.linspace(0, len(df_target), block_count + 1, dtype=int)
        rolling_dict = dict(dict_pips_seed)
        dataset_parts = []

        for block_idx in range(block_count):
            start_idx = split_points[block_idx]
            end_idx = split_points[block_idx + 1]
            if end_idx <= start_idx:
                continue

            df_block = df_target.iloc[start_idx:end_idx]
            block_dataset = self.build_dataset_from_df(df_block, rolling_dict, allow_empty=True)
            if not block_dataset.empty:
                dataset_parts.append(block_dataset)

            block_pips_map = self.collect_training_pips_map(df_block)
            if block_pips_map:
                rolling_dict = self.actualizar_dict(rolling_dict, block_pips_map)

        if not dataset_parts:
            return self.build_dataset_from_df(df_target, dict_pips_seed)

        return pd.concat(dataset_parts, ignore_index=True)


    def build_selection_reference(self, metrics):
        raw_reasons = []
        if metrics["cantidad_operaciones"] < self.selection_min_trades:
            raw_reasons.append("low_trades")
        if metrics["profit_factor"] < self.selection_min_profit_factor:
            raw_reasons.append("low_profit_factor")
        if metrics["expectancy"] < self.selection_min_expectancy:
            raw_reasons.append("low_expectancy")
        if metrics["winrate"] < self.selection_min_winrate:
            raw_reasons.append("low_winrate")
        if metrics["max_drawdown_ratio"] > self.selection_max_drawdown_ratio:
            raw_reasons.append("high_drawdown_ratio")
        if metrics["mas_perdidas_seguidas"] > self.selection_max_losing_streak:
            raw_reasons.append("high_losing_streak")
        if metrics["temporal_stats"]["n_months"] < self.selection_min_months:
            raw_reasons.append("low_month_coverage")
        if metrics["temporal_stats"]["positive_month_ratio"] < self.selection_min_positive_month_ratio:
            raw_reasons.append("low_positive_month_ratio")
        if metrics["temporal_stats"]["positive_quarter_ratio"] < self.selection_min_positive_quarter_ratio:
            raw_reasons.append("low_positive_quarter_ratio")

        deployment_score = (
            (metrics["profit_factor"] * 2.0) +
            (metrics["expectancy"] * 1.5) +
            (metrics["winrate"] * 100.0 * 0.03) +
            (metrics["sharpe"] * 1.5) -
            (metrics["max_drawdown_ratio"] * 2.0) +
            (metrics["temporal_stats"]["positive_month_ratio"] * 3.0) +
            (metrics["temporal_stats"]["positive_quarter_ratio"] * 2.0)
        )
        if not np.isfinite(deployment_score):
            deployment_score = -9999.0

        return {
            "raw_reasons": raw_reasons,
            "deployment_score": float(deployment_score),
            "thresholds": {
                "min_trades": self.selection_min_trades,
                "min_profit_factor": self.selection_min_profit_factor,
                "min_expectancy": self.selection_min_expectancy,
                "min_winrate": self.selection_min_winrate,
                "max_drawdown_ratio": self.selection_max_drawdown_ratio,
                "max_losing_streak": self.selection_max_losing_streak,
                "min_months": self.selection_min_months,
                "min_positive_month_ratio": self.selection_min_positive_month_ratio,
                "min_positive_quarter_ratio": self.selection_min_positive_quarter_ratio,
                "strong_candidate": {
                    "min_trades": self.selection_strong_candidate_min_trades,
                    "min_profit_factor": self.selection_strong_candidate_min_profit_factor,
                    "min_expectancy": self.selection_strong_candidate_min_expectancy,
                    "min_winrate": self.selection_strong_candidate_min_winrate,
                    "max_drawdown_ratio": self.selection_strong_candidate_max_drawdown_ratio,
                    "max_losing_streak": self.selection_strong_candidate_max_losing_streak,
                    "min_positive_month_ratio": self.selection_strong_candidate_min_positive_month_ratio,
                    "min_positive_quarter_ratio": self.selection_strong_candidate_min_positive_quarter_ratio,
                    "min_deployment_score": self.selection_strong_candidate_min_deployment_score,
                    "allowed_reasons": sorted(self.selection_strong_candidate_allowed_reasons),
                },
            },
        }


    def summarize_temporal_performance(self, lista_pips):
        monthly_pips = {}
        quarterly_pips = {}

        for item in lista_pips:
            time_close = item.get("time_close")
            if not time_close:
                continue
            close_ts = pd.to_datetime(time_close)
            month_key = close_ts.strftime("%Y-%m")
            quarter_key = f"{close_ts.year}-Q{((close_ts.month - 1) // 3) + 1}"
            monthly_pips[month_key] = monthly_pips.get(month_key, 0.0) + float(item["pips"])
            quarterly_pips[quarter_key] = quarterly_pips.get(quarter_key, 0.0) + float(item["pips"])

        month_values = list(monthly_pips.values())
        quarter_values = list(quarterly_pips.values())

        positive_month_ratio = (
            float(sum(1 for value in month_values if value > 0) / len(month_values))
            if month_values else 0.0
        )
        positive_quarter_ratio = (
            float(sum(1 for value in quarter_values if value > 0) / len(quarter_values))
            if quarter_values else 0.0
        )

        return {
            "n_months": len(month_values),
            "n_quarters": len(quarter_values),
            "positive_month_ratio": positive_month_ratio,
            "positive_quarter_ratio": positive_quarter_ratio,
            "best_month_pips": float(max(month_values)) if month_values else 0.0,
            "worst_month_pips": float(min(month_values)) if month_values else 0.0,
            "best_quarter_pips": float(max(quarter_values)) if quarter_values else 0.0,
            "worst_quarter_pips": float(min(quarter_values)) if quarter_values else 0.0,
            "monthly_pips": monthly_pips,
            "quarterly_pips": quarterly_pips,
        }


    def get_validation_window_df(self):
        return self.df_valid


    def resolve_entry_open_nodes(self, time_actual_np):
        matched_symbols = 0
        open_nodes = []

        for symbol in self.list_symbols:
            nodo_open_list = self.dict_nodos[symbol]

            for nodo in nodo_open_list:
                df_struct = self.COMBINED[("open", symbol, nodo["file"])]
                pos = df_struct["index_values"].searchsorted(time_actual_np)

                if pos == 0:
                    continue

                if self.cumple_condiciones_fast(df_struct, pos - 1, nodo["conditions"]):
                    matched_symbols += 1

                    if symbol == self.principal_symbol:
                        nodo_open = self.maping_open.get(nodo["key"])
                        if nodo_open is not None:
                            open_nodes.append(nodo_open)
                    break

        if matched_symbols < self.min_open_symbol_confirmations:
            return []

        if not open_nodes:
            return []

        return list(dict.fromkeys(open_nodes))


    def select_label_reference_values(self, hour_key, context_values, global_values):
        hour_values = context_values.get(hour_key, [])
        hour_unique = len({round(float(value), 8) for value in hour_values})
        global_unique = len({round(float(value), 8) for value in global_values})

        if len(hour_values) >= self.label_min_context_samples and hour_unique >= 2:
            return hour_values
        if len(global_values) >= 2 and global_unique >= 2:
            return global_values
        if len(hour_values) >= 2 and hour_unique >= 2:
            return hour_values
        return []


    def get_effective_label_percentile(self, candidate_count):
        base_percentile = float(self.label_percentile_threshold)
        if candidate_count <= 0:
            return base_percentile

        target_keep_ratio = min(1.0, (self.min_training_samples / max(candidate_count, 1)) * 1.05)
        adaptive_percentile = 100.0 - (50.0 * target_keep_ratio)
        return float(min(base_percentile, max(50.0, adaptive_percentile)))


    def build_dataset_from_df(self, df, dict_pips_best, allow_empty=False):

        candidate_samples = []

        for row in df.itertuples():
            time_actual = row.Index
            time_actual_np = np.datetime64(time_actual)
            hour = format(time_actual.hour, "05b")

            if not self.is_entry_event_active(row):
                continue

            open_nodes = self.resolve_entry_open_nodes(time_actual_np)

            if not open_nodes:
                continue

            valid_closes = []
            for nodo_c in self.nodos_close:
                df_struct_c = self.COMBINED[("close", nodo_c["file"])]
                pos_c = df_struct_c["index_values"].searchsorted(time_actual_np)

                if pos_c == 0:
                    continue

                if self.cumple_condiciones_fast(df_struct_c, pos_c - 1, nodo_c["conditions"]):
                    nodo_close = self.maping_close.get(nodo_c["key"])
                    if nodo_close is not None:
                        valid_closes.append(nodo_close)

            if not valid_closes:
                continue

            raw_volatility_pips = max(
                abs(float(getattr(row, 'vol_10', 0.0))) / self.pip_size,
                abs(float(getattr(row, 'vol_20', 0.0))) / self.pip_size,
            )
            volatility_pips = max(raw_volatility_pips, self.label_volatility_floor_pips)

            for nodo_open in open_nodes:
                for nodo_close in valid_closes:
                    key = f"{nodo_open}_{nodo_close}_{hour}"

                    if key not in dict_pips_best:
                        continue

                    expected_pips = float(dict_pips_best[key])
                    normalized_pips = float(expected_pips / volatility_pips)
                    candidate_samples.append({
                        "input1": nodo_open,
                        "input2": nodo_close,
                        "hour": hour,
                        "ret_1": float(row.ret_1),
                        "range_1": float(row.range_1),
                        "trend": float(row.trend),
                        "vol": float(row.vol),
                        "ret_3": float(row.ret_3),
                        "ret_10": float(row.ret_10),
                        "trend_10": float(row.trend_10),
                        "trend_20": float(row.trend_20),
                        "vol_10": float(row.vol_10),
                        "vol_20": float(row.vol_20),
                        "zscore_20": float(row.zscore_20),
                        "momentum_ratio": float(row.momentum_ratio),
                        "event_breakout_up_20": float(getattr(row, 'event_breakout_up_20', 0.0)),
                        "event_breakout_down_20": float(getattr(row, 'event_breakout_down_20', 0.0)),
                        "event_volatility_expansion_10": float(getattr(row, 'event_volatility_expansion_10', 0.0)),
                        "event_momentum_shift_up": float(getattr(row, 'event_momentum_shift_up', 0.0)),
                        "event_momentum_shift_down": float(getattr(row, 'event_momentum_shift_down', 0.0)),
                        "event_trend_alignment_up": float(getattr(row, 'event_trend_alignment_up', 0.0)),
                        "event_trend_alignment_down": float(getattr(row, 'event_trend_alignment_down', 0.0)),
                        "event_range_impulse": float(getattr(row, 'event_range_impulse', 0.0)),
                        "expected_pips": expected_pips,
                        "normalized_pips": normalized_pips,
                    })

        context_values = {}
        global_values = []
        for sample in candidate_samples:
            context_values.setdefault(sample["hour"], []).append(sample["normalized_pips"])
            global_values.append(sample["normalized_pips"])
        effective_percentile = self.get_effective_label_percentile(len(candidate_samples))
        lower_percentile = 100.0 - effective_percentile

        data = {
            'input1': [],
            'input2': [],
            'hour': [],
            'ret_1': [],
            'range_1': [],
            'trend': [],
            'vol': [],
            'ret_3': [],
            'ret_10': [],
            'trend_10': [],
            'trend_20': [],
            'vol_10': [],
            'vol_20': [],
            'zscore_20': [],
            'momentum_ratio': [],
            'output': []
        }
        for event_column in EVENT_FEATURE_COLUMNS:
            data[event_column] = []

        for sample in candidate_samples:
            reference_values = self.select_label_reference_values(
                sample["hour"],
                context_values,
                global_values,
            )
            if len(reference_values) < 2:
                continue

            upper_threshold = float(np.percentile(reference_values, effective_percentile))
            lower_threshold = float(np.percentile(reference_values, lower_percentile))
            normalized_pips = sample["normalized_pips"]

            if np.isclose(upper_threshold, lower_threshold):
                median_threshold = float(np.median(reference_values))
                if normalized_pips > median_threshold:
                    label = 1.0
                elif normalized_pips < median_threshold:
                    label = 0.0
                else:
                    continue
            elif normalized_pips >= upper_threshold:
                label = 1.0
            elif normalized_pips <= lower_threshold:
                label = 0.0
            else:
                continue

            data['input1'].append(sample['input1'])
            data['input2'].append(sample['input2'])
            data['hour'].append(sample['hour'])
            data['ret_1'].append(sample['ret_1'])
            data['range_1'].append(sample['range_1'])
            data['trend'].append(sample['trend'])
            data['vol'].append(sample['vol'])
            data['ret_3'].append(sample['ret_3'])
            data['ret_10'].append(sample['ret_10'])
            data['trend_10'].append(sample['trend_10'])
            data['trend_20'].append(sample['trend_20'])
            data['vol_10'].append(sample['vol_10'])
            data['vol_20'].append(sample['vol_20'])
            data['zscore_20'].append(sample['zscore_20'])
            data['momentum_ratio'].append(sample['momentum_ratio'])
            for event_column in EVENT_FEATURE_COLUMNS:
                data[event_column].append(sample[event_column])
            data['output'].append(label)

        df_dataset = pd.DataFrame(data)

        return df_dataset


    def get_operational_close_threshold(self):
        if self.close_threshold_history:
            recent = self.close_threshold_history[-self.close_threshold_history_window:]
            threshold = float(np.median(recent))
        else:
            threshold = self.info_score.get("best_threshold", self.close_threshold_floor)

        threshold = max(threshold, self.close_threshold_floor)
        threshold = min(threshold, self.close_threshold_ceiling)
        return float(threshold)


    def split_threshold_validation_trades(self, lista_pips):
        trades_with_prob = [item for item in lista_pips if item.get("prob") is not None]
        if len(trades_with_prob) < (self.threshold_min_calibration_trades + self.threshold_min_selected_trades):
            return trades_with_prob, trades_with_prob

        split_idx = int(len(trades_with_prob) * self.threshold_validation_split_ratio)
        split_idx = max(self.threshold_min_calibration_trades, split_idx)
        split_idx = min(split_idx, len(trades_with_prob) - self.threshold_min_selected_trades)

        calibration = trades_with_prob[:split_idx]
        scoring = trades_with_prob[split_idx:]
        if not scoring:
            scoring = calibration
        return calibration, scoring


    def score_threshold_subset(self, trade_subset, threshold):
        selected = [
            item["pips"]
            for item in trade_subset
            if item.get("prob") is not None and item["prob"] >= threshold
        ]

        if len(selected) < self.threshold_min_selected_trades:
            return None

        expectancy = float(np.mean(selected))
        gross_profit = float(sum(value for value in selected if value > 0))
        gross_loss = float(abs(sum(value for value in selected if value < 0)))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (2.0 if gross_profit > 0 else 0.0)
        equity_curve = np.cumsum(selected)
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = equity_curve - running_max
        max_drawdown = float(abs(drawdowns.min())) if len(drawdowns) else 0.0
        drawdown_ratio = max_drawdown / max(abs(float(sum(selected))), 1.0)
        score = (
            (expectancy * np.sqrt(len(selected))) +
            (min(profit_factor, 3.0) * 2.0) -
            (drawdown_ratio * 2.0)
        )
        return {
            "score": float(score),
            "selected_trades": int(len(selected)),
            "expectancy": expectancy,
            "profit_factor": float(profit_factor),
            "drawdown_ratio": float(drawdown_ratio),
        }


    def find_best_threshold(self, lista_pips):
        calibration_trades, scoring_trades = self.split_threshold_validation_trades(lista_pips)
        thresholds = np.linspace(self.close_threshold_floor, self.close_threshold_ceiling, 15)
        best_thr = self.get_operational_close_threshold()
        best_score = -np.inf
        best_calibration_stats = None

        for thr in thresholds:
            calibration_stats = self.score_threshold_subset(calibration_trades, float(thr))
            if calibration_stats is None:
                continue

            if calibration_stats["score"] > best_score:
                best_score = calibration_stats["score"]
                best_thr = float(thr)
                best_calibration_stats = calibration_stats

        if best_calibration_stats is None:
            fallback_thr = self.get_operational_close_threshold()
            fallback_score_stats = self.score_threshold_subset(scoring_trades, fallback_thr)
            fallback_score = fallback_score_stats["score"] if fallback_score_stats else 0.0
            return {
                "threshold": float(fallback_thr),
                "raw_threshold": float(fallback_thr),
                "score": float(fallback_score),
                "calibration_trades": int(len(calibration_trades)),
                "scoring_trades": int(len(scoring_trades)),
                "smoothed": bool(self.close_threshold_history),
            }

        self.close_threshold_history.append(best_thr)
        smoothed_thr = self.get_operational_close_threshold()
        scoring_stats = self.score_threshold_subset(scoring_trades, smoothed_thr)
        scoring_score = scoring_stats["score"] if scoring_stats else 0.0

        return {
            "threshold": float(smoothed_thr),
            "raw_threshold": float(best_thr),
            "score": float(scoring_score),
            "calibration_trades": int(len(calibration_trades)),
            "scoring_trades": int(len(scoring_trades)),
            "smoothed": True,
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

        path_data_red = f'output/{self.principal_symbol}/data_for_neuronal/data/data_{self.mercado}_{self.algorithm}.csv'
        dict_pips_mean = self.collect_training_pips_map(self.df_train_A)

                            
        # =========================
        # SELECCIÓN TOP/BOTTOM
        # =========================
        if self.index == 1:
            self.dict_pips_best = dict_pips_mean
        else:        
            self.dict_pips_best = self.actualizar_dict(self.dict_pips_best, dict_pips_mean)
        print(len(self.dict_pips_best), "pips para actualizar en la próxima iteración")

        # =========================
        # REENTRENAMIENTO
        # =========================

        df_dataset = self.build_walk_forward_dataset(self.df_train_B, self.dict_pips_best)
        if len(df_dataset) < self.min_training_samples or df_dataset['output'].nunique() < 2:
            df_pre_validation = pd.concat([self.df_train_A, self.df_train_B])
            df_dataset_fallback = self.build_walk_forward_dataset(df_pre_validation, self.dict_pips_best)
            if len(df_dataset_fallback) > len(df_dataset):
                print(
                    "Fallback de entrenamiento activado | "
                    f"dataset original={len(df_dataset)} -> expandido={len(df_dataset_fallback)}"
                )
                df_dataset = df_dataset_fallback

        if len(df_dataset) < self.min_training_samples or df_dataset['output'].nunique() < 2:
            raise ValueError(
                f"Dataset insuficiente para entrenar {self.principal_symbol}-{self.mercado}-{self.algorithm}: "
                f"samples={len(df_dataset)} classes={df_dataset['output'].nunique()}"
            )
        df_dataset.to_csv(path_data_red, index=False)

        is_valid, dataset_info = has_minimum_training_data(path_data_red, min_samples=self.min_training_samples)
        if not is_valid:
            raise ValueError(
                f"Dataset inválido para entrenar {self.principal_symbol}-{self.mercado}-{self.algorithm}: "
                f"{dataset_info['reason']} samples={dataset_info['samples']} classes={dataset_info['class_count']}"
            )

        input1_ids, input2_ids, hour_ids, X_extra, Y, norm_stats = load_data(path_data_red, return_stats=True)

        print("Datos cargados:")
        print("X_extra shape:", X_extra.shape)
        print("Y shape:", Y.shape)

        num_open, num_close, num_hours = get_embedding_vocab_sizes(input1_ids, input2_ids, hour_ids)
        nn = BinaryNN(
            input_dim_extra=X_extra.shape[1],
            num_open=num_open,
            num_close=num_close,
            num_hours=num_hours,
            lr=0.001,
            target_loss=0.10,
        )
        nn.feature_mean = norm_stats["mean"]
        nn.feature_std = norm_stats["std"]
        nn.fit(input1_ids, input2_ids, hour_ids, X_extra, Y, epochs=200, batch_size=32)

        model_path = f'output/{self.principal_symbol}/data_for_neuronal/model_trainer/model_{self.mercado}_{self.algorithm}.pt'
        save_trained_model(nn, model_path)

        print(f"Modelo entrenado guardado en '{model_path}'")

        return {
            "model_type": "embedding",
            "input_dim_extra": nn.input_dim_extra,
            "num_open": nn.num_open,
            "num_close": nn.num_close,
            "num_hours": nn.num_hours,
            "emb_open_dim": nn.emb_open_dim,
            "emb_close_dim": nn.emb_close_dim,
            "emb_hour_dim": nn.emb_hour_dim,
            "dropout_rate": nn.dropout_rate,
            "weight_decay": nn.weight_decay,
            "false_positive_cost_ratio": nn.false_positive_cost_ratio,
            "state_dict": {k: v.detach().cpu() for k, v in nn.state_dict().items()},
            "feature_mean": None if nn.feature_mean is None else np.asarray(nn.feature_mean, dtype=np.float32).tolist(),
            "feature_std": None if nn.feature_std is None else np.asarray(nn.feature_std, dtype=np.float32).tolist(),
        }
    
  
    def validate_iteration(self):

        path_data_red = f'output/{self.principal_symbol}/data_for_neuronal/data/data_{self.mercado}_{self.algorithm}.csv'
        input1_ids, input2_ids, hour_ids, X_extra, Y = load_data(path_data_red)

        nn = load_trained_model(
            f'output/{self.principal_symbol}/data_for_neuronal/model_trainer/model_{self.mercado}_{self.algorithm}.pt',
            input_dim_extra=X_extra.shape[1]
        )
        validate_embedding_vocab(nn, input1_ids, input2_ids, hour_ids)

        df_validation = self.get_validation_window_df()
        validation_start = df_validation.index.min()
        validation_end = df_validation.index.max()
        print(
            f"Validando sobre los últimos {self.validation_window_months} meses: "
            f"{validation_start} -> {validation_end} | barras={len(df_validation)}"
        )

        is_open = False
        open_price_open = 0.0
        spread_open = 0.0
        entry_red_open = ''
        cierre = 0
        model_close_streak = 0
        time_comienzo = None
        current_stop_loss = self.stop_loss
        current_take_profit = self.take_profit

        cantidad_operaciones = 0
        operaciones_acertadas = 0
        operaciones_perdedoras = 0

        ganancia_bruta = 0.0
        perdida_bruta = 0.0
        sum_pips = 0.0

        lista_pips = []
        holding_bars_list = []
        close_reason_counts = {
            "stop_loss": 0,
            "take_profit": 0,
            "max_holding": 0,
            "model": 0,
        }
        perdidas_seguidas = 0
        mas_perdidas_seguidas = 0
        close_threshold = self.get_operational_close_threshold()

        for row in df_validation.itertuples():

            time_actual = row.Index
            time_actual_np = np.datetime64(time_actual)
            if not is_open:
                if not self.horas_mercado[time_actual.hour]:
                    continue
                
            open_price = row.open
            hour = format(time_actual.hour, "05b")
            market_features = self.get_market_features(row)

            # =========================
            # CIERRE
            # =========================
            if is_open:
                cierre += 1
                current_pips = self.calculate_trade_pips(
                    open_price_open,
                    open_price,
                    spread_open,
                )
                prob = None
                close_reason = None
                if current_pips <= -current_stop_loss:
                    structural_close = True
                    close_reason = "stop_loss"
                elif current_pips >= current_take_profit:
                    structural_close = True
                    close_reason = "take_profit"
                elif cierre >= self.max_holding:
                    structural_close = True
                    close_reason = "max_holding"
                else:
                    structural_close = False
                model_close_signal = False
                if (not structural_close) and cierre >= self.min_model_holding:
                    for nodo in self.nodos_close:
                        df_struct = self.COMBINED[("close", nodo["file"])]
                        pos = df_struct["index_values"].searchsorted(time_actual_np)
                        if pos == 0:
                            continue

                        if self.cumple_condiciones_fast(df_struct, pos - 1, nodo["conditions"]):

                            nodo_close = self.maping_close.get(nodo["key"])
                            if nodo_close is None:
                                continue
                            prob = predict_from_inputs(nn, entry_red_open, nodo_close, hour, market_features)
                            if prob > close_threshold:
                                model_close_signal = True
                                break

                if model_close_signal:
                    model_close_streak += 1
                else:
                    model_close_streak = 0

                if (not structural_close) and model_close_streak >= self.close_confirmation_bars:
                    close_reason = "model"

                cerrar = structural_close or (model_close_streak >= self.close_confirmation_bars)

                if cerrar:
                    trade_pips = current_pips

                    if trade_pips < 0:
                        perdidas_seguidas += 1
                    else:
                        perdidas_seguidas = 0

                    is_open = False
                    model_close_streak = 0
                    cantidad_operaciones += 1
                    sum_pips += trade_pips
                    holding_bars_list.append(cierre)
                    if close_reason is not None:
                        close_reason_counts[close_reason] += 1
                    lista_pips.append({
                        "pips": trade_pips,
                        "prob": prob,
                        "bars_held": cierre,
                        "close_reason": close_reason,
                        "time_open": time_comienzo.isoformat() if hasattr(time_comienzo, "isoformat") else str(time_comienzo),
                        "time_close": time_actual.isoformat() if hasattr(time_actual, "isoformat") else str(time_actual),
                    })

                    if trade_pips > 0:
                        operaciones_acertadas += 1
                        ganancia_bruta += trade_pips
                    else:
                        operaciones_perdedoras += 1
                        perdida_bruta += abs(trade_pips)

                    print(f"SUM PIPS:{time_comienzo} ---- {time_actual}: {sum_pips}", cierre)

            # =========================
            # APERTURA
            # =========================
            else:

                cierre = 0

                if not self.is_entry_event_active(row):
                    continue

                open_nodes = self.resolve_entry_open_nodes(time_actual_np)
                if open_nodes:
                    open_price_open = open_price
                    spread_open = getattr(row, 'spread', 0.0)
                    current_stop_loss, current_take_profit = self.get_trade_risk_limits(row)
                    is_open = True
                    time_comienzo = time_actual
                    model_close_streak = 0
                    entry_red_open = open_nodes[0]

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

        pips_values = [item["pips"] for item in lista_pips]
        sharpe = 0
        if len(pips_values) > 1 and np.std(pips_values) != 0:
            sharpe = np.mean(pips_values) / np.std(pips_values)

        max_drawdown = 0.0
        if pips_values:
            equity_curve = np.cumsum(pips_values)
            running_max = np.maximum.accumulate(equity_curve)
            drawdowns = equity_curve - running_max
            max_drawdown = float(drawdowns.min())

        avg_bars_held = float(np.mean(holding_bars_list)) if holding_bars_list else 0.0
        median_bars_held = float(np.median(holding_bars_list)) if holding_bars_list else 0.0
        max_drawdown_abs = abs(max_drawdown)
        drawdown_denominator = max(abs(sum_pips), 1.0)
        max_drawdown_ratio = float(max_drawdown_abs / drawdown_denominator)
        calmar_ratio = float(sum_pips / max_drawdown_abs) if max_drawdown_abs > 0 else (3.0 if sum_pips > 0 else 0.0)
        negative_pips = [value for value in pips_values if value < 0]
        if len(negative_pips) > 1 and np.std(negative_pips) != 0:
            sortino_ratio = float(np.mean(pips_values) / np.std(negative_pips))
        elif pips_values and np.mean(pips_values) > 0:
            sortino_ratio = 3.0
        else:
            sortino_ratio = 0.0
        temporal_stats = self.summarize_temporal_performance(lista_pips)

        threshold_info = self.find_best_threshold(lista_pips)
        best_thr = threshold_info["threshold"]
        best_threshold_score = threshold_info["score"]
        print(
            f"Best threshold: {best_thr:.3f} | Score: {best_threshold_score:.4f} | "
            f"raw={threshold_info['raw_threshold']:.3f}"
        )

        print(f"""
        Operaciones: {cantidad_operaciones}
        Winrate: {winrate*100:.2f}%
        Profit Factor: {profit_factor:.2f}
        Expectancy: {expectancy:.4f}
        Pips Totales: {sum_pips:.2f}
        """)

        metrics = {
            "winrate": winrate,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "lista_pips": lista_pips,
            "best_threshold": best_thr,
            "best_threshold_raw": threshold_info["raw_threshold"],
            "best_threshold_score": best_threshold_score,
            "best_threshold_meta": threshold_info,
            "operational_close_threshold": close_threshold,
            "sharpe": sharpe,
            "calmar_ratio": calmar_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "max_drawdown_ratio": max_drawdown_ratio,
            "avg_bars_held": avg_bars_held,
            "median_bars_held": median_bars_held,
            "close_reason_counts": close_reason_counts,
            "temporal_stats": temporal_stats,
            "validation_window": {
                "months": self.validation_window_months,
                "start": validation_start.isoformat() if hasattr(validation_start, "isoformat") else str(validation_start),
                "end": validation_end.isoformat() if hasattr(validation_end, "isoformat") else str(validation_end),
                "bars": int(len(df_validation)),
            },
            "cantidad_operaciones": cantidad_operaciones,
            "sum_pips": sum_pips,
            "mas_perdidas_seguidas": mas_perdidas_seguidas
        }
        metrics["selection_reference"] = self.build_selection_reference(metrics)
        return metrics
    

    def calculate_preliminary_score(self, metrics):

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

        return {
            "score": float(score),
            "score_base": float(score_base),
            "volumen_factor": float(volumen_factor),
            "reliability_factor": float(reliability_factor),
            "low_trade_penalty": float(low_trade_penalty),
        }


    def percentile_rank(self, values, value):
        if not values:
            return 0.5

        less_count = sum(1 for item in values if item < value)
        equal_count = sum(1 for item in values if item == value)
        return float((less_count + 0.5 * equal_count) / len(values))


    def finalize_best_candidate(self, last_model_data):
        eligible_candidates = [
            candidate
            for candidate in self.iteration_candidates
            if candidate["metrics"]["iteracion"] >= self.min_winning_iteration
        ]

        if not eligible_candidates:
            if self.iteration_candidates:
                fallback_candidate = self.iteration_candidates[-1]
                self.best_model_data = fallback_candidate["model_data"]
                fallback_metrics = dict(fallback_candidate["metrics"])
                fallback_metrics["score"] = fallback_metrics.get("preliminary_score", 0.0)
                fallback_metrics["score_model"] = "preliminary_fallback"
                fallback_metrics["peor_trade"] = min((item["pips"] for item in fallback_metrics["lista_pips"]), default=0)
                self.info_score = fallback_metrics
                del self.info_score["lista_pips"]
                self.best_score = fallback_metrics["score"]
            elif self.best_model_data is None:
                self.best_model_data = last_model_data
            return

        metric_values = {
            metric_name: [candidate["metrics"].get(metric_name, 0.0) for candidate in eligible_candidates]
            for metric_name in self.robust_score_weights
        }

        best_candidate = None
        best_robust_score = float("-inf")
        ranked_candidates = []

        for candidate in eligible_candidates:
            metrics = candidate["metrics"]
            score_components = {}
            for metric_name, weight in self.robust_score_weights.items():
                percentile = self.percentile_rank(metric_values[metric_name], metrics.get(metric_name, 0.0))
                score_components[metric_name] = float(percentile * weight)

            robust_score_base = float(sum(score_components.values()))
            n_ops = metrics.get("cantidad_operaciones", 0)
            trade_penalty = float(
                1.0 / (1.0 + np.exp(-(n_ops - self.robust_trade_penalty_center) / self.robust_trade_penalty_scale))
            )
            avg_bars = metrics.get("avg_bars_held", 0.0)
            duration_penalty = 1.0 if avg_bars >= self.robust_min_avg_bars else float(max(0.75, avg_bars / max(self.robust_min_avg_bars, 1.0)))
            robust_score = robust_score_base * trade_penalty * duration_penalty

            metrics["score_components"] = score_components
            metrics["score_base"] = robust_score_base
            metrics["trade_penalty"] = trade_penalty
            metrics["duration_penalty"] = duration_penalty
            metrics["score_model"] = "percentile_batch_v1"
            metrics["score"] = float(robust_score)
            ranked_candidates.append(candidate)

            if robust_score > best_robust_score:
                best_robust_score = robust_score
                best_candidate = candidate

        if best_candidate is None:
            if self.best_model_data is None:
                self.best_model_data = last_model_data
            return

        self.best_score = float(best_robust_score)
        self.best_model_data = best_candidate["model_data"]
        best_metrics = dict(best_candidate["metrics"])
        self.peor_trade = min((item["pips"] for item in best_metrics["lista_pips"]), default=0)
        best_metrics["peor_trade"] = self.peor_trade

        ranked_candidates.sort(key=lambda item: item["metrics"].get("score", float("-inf")), reverse=True)
        top_candidates = ranked_candidates[:self.ensemble_top_k]
        ensemble_weight_sum = 0.0
        self.ensemble_model_entries = []
        for rank, candidate in enumerate(top_candidates):
            candidate_score = max(float(candidate["metrics"].get("score", 0.0)), 0.0) + 1e-6
            weight = candidate_score * (self.ensemble_decay_factor ** rank)
            ensemble_weight_sum += weight
            self.ensemble_model_entries.append({
                "model_data": candidate["model_data"],
                "weight": float(weight),
                "iteration": candidate["metrics"].get("iteracion"),
                "score": candidate["metrics"].get("score"),
                "threshold": candidate["metrics"].get("best_threshold", self.close_threshold_floor),
            })

        if ensemble_weight_sum > 0:
            for entry in self.ensemble_model_entries:
                entry["weight"] = float(entry["weight"] / ensemble_weight_sum)

        ensemble_threshold = best_metrics.get("best_threshold", self.close_threshold_floor)
        if self.ensemble_model_entries:
            ensemble_threshold = float(sum(entry["weight"] * entry["threshold"] for entry in self.ensemble_model_entries))

        best_metrics["ensemble"] = {
            "enabled": len(self.ensemble_model_entries) > 1,
            "top_k": len(self.ensemble_model_entries),
            "decay_factor": self.ensemble_decay_factor,
            "threshold": ensemble_threshold,
            "members": [
                {
                    "iteration": entry["iteration"],
                    "score": entry["score"],
                    "weight": entry["weight"],
                }
                for entry in self.ensemble_model_entries
            ],
        }
        self.info_score = best_metrics
        self.info_score["best_threshold"] = best_metrics["best_threshold"]
        del self.info_score["lista_pips"]
        print(
            f"Mejor modelo final por percentil batch: iteración {best_metrics['iteracion']} | "
            f"score={best_metrics['score']:.4f}"
        )


    def calculate_score(self, metrics, model_data):
        preliminary = self.calculate_preliminary_score(metrics)
        n_ops = metrics["cantidad_operaciones"]

        print(
            f"Score preliminar: {preliminary['score']:.4f} | base={preliminary['score_base']:.4f} | ops={n_ops} | "
            f"vol={preliminary['volumen_factor']:.4f} | rel={preliminary['reliability_factor']:.4f} | "
            f"low_ops={preliminary['low_trade_penalty']:.4f}"
        )

        metrics["iteracion"] = self.index
        metrics["preliminary_score"] = preliminary["score"]
        metrics["preliminary_score_base"] = preliminary["score_base"]
        metrics["volumen_factor"] = preliminary["volumen_factor"]
        metrics["reliability_factor"] = preliminary["reliability_factor"]
        metrics["low_trade_penalty"] = preliminary["low_trade_penalty"]
        self.iteration_candidates.append({
            "metrics": metrics,
            "model_data": model_data,
        })

        if self.index < self.min_winning_iteration:
            print(
                f"Iteración {self.index}: score preliminar ignorado para ganador; "
                f"el mejor modelo solo se actualiza desde la iteración {self.min_winning_iteration}."
            )
            return preliminary["score"]

        if preliminary["score"] > self.best_score:
            self.best_score = preliminary["score"]
            self.no_improve_iterations = 0
            print(f"Nueva mejor referencia preliminar: {preliminary['score']:.4f}")
        else:
            self.no_improve_iterations += 1
            print(
                f"Sin mejora en score preliminar durante {self.no_improve_iterations} "
                f"iteraciones elegibles consecutivas."
            )
        return preliminary["score"]


    def run(self):
        last_model_data = None

        for i in range(self.max_iterations):
            print(f"\n--- Iteración {i+1} ---")
            self.index = i + 1
            try:
                model_data = self.train_iteration()
            except ValueError as exc:
                print(f"Backtester cancelado por dataset insuficiente: {exc}")
                self.best_model_data = None
                self.info_score = {
                    "skipped": True,
                    "skip_reason": str(exc),
                    "selection_reference": {
                        "raw_reasons": ["insufficient_training_data"],
                        "deployment_score": -9999.0,
                        "thresholds": {},
                    },
                }
                break
            last_model_data = model_data
            metrics = self.validate_iteration()
            self.calculate_score(metrics, model_data)

            if (
                self.early_stopping_patience > 0
                and self.index >= self.min_winning_iteration
                and self.no_improve_iterations >= self.early_stopping_patience
            ):
                print(
                    "Early stopping activado: "
                    f"{self.no_improve_iterations} iteraciones elegibles sin mejora."
                )
                break

        self.finalize_best_candidate(last_model_data)

        if self.best_model_data is None:
            with open(f'output/{self.principal_symbol}/data_for_neuronal/best_score/score_{self.mercado}_{self.algorithm}.json', 'w') as f:
                json.dump({
                    "metrics": _sanitize_for_json(self.info_score)
                }, f, indent=4, allow_nan=False)
            print(f'tiempo total: {time.time() - self.comienzo:.2f} segundos')
            return

        torch_model_path = f'output/{self.principal_symbol}/data_for_neuronal/model_trainer/model_{self.mercado}_{self.algorithm}.pt'
        torch.save(self.best_model_data, torch_model_path)
        if self.ensemble_model_entries:
            ensemble_model_path = f'output/{self.principal_symbol}/data_for_neuronal/model_trainer/ensemble_{self.mercado}_{self.algorithm}.pt'
            save_ensemble_model(self.ensemble_model_entries, ensemble_model_path)
            self.info_score.setdefault("ensemble", {})["model_path"] = ensemble_model_path
        with open(f'output/{self.principal_symbol}/data_for_neuronal/best_score/score_{self.mercado}_{self.algorithm}.json', 'w') as f:
            json.dump({
                "metrics": _sanitize_for_json(self.info_score)
            }, f, indent=4, allow_nan=False)
        print(f'tiempo total: {time.time() - self.comienzo:.2f} segundos')
    
       
      
if __name__ == "__main__":
    inn = time.time()
    backtester = Backtester('AUDCHF', 'Asia', 'DOWN', date_end=None) 
    backtester.run()
    print(f'segundos {time.time()-inn}')
    