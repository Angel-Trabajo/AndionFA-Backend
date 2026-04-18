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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.db.query import get_nodes_by_label

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
    get_embedding_vocab_sizes,
    validate_embedding_vocab,
)
from src.signals.event_generator import add_event_features, has_entry_event


VERBOSE_TRADE_LOGS = os.getenv("VERBOSE_TRADE_LOGS", "0") == "1"
TRADE_LOG_EVERY = max(1, int(os.getenv("TRADE_LOG_EVERY", "25")))


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

    def __init__(self, principal_symbol, mercado, algorithm):

        print("Starting backtest...")
        self.comienzo = time.time()

        # Estado global del sistema
        self.index = 0
        self.best_score = float('-inf')
        self.best_model_data = None
        self.peor_trade = 0.0
        self.info_score = {}
        self.iteration_candidates = []
        self.dict_pips_best = {}
        self.stop_loss = 20
        self.take_profit = 150
        self.max_holding = 120
        self.min_model_holding = 15
        self.close_confirmation_bars = 2
        self.close_threshold_floor = 0.60
        self.validation_window_fraction = 0.15
        self.min_open_symbol_confirmations = 4
        self.train_b_fraction = 0.30
        self.walk_forward_enabled = True
        self.walk_forward_min_train_fraction = 0.50
        self.walk_forward_step_fraction = None
        self.walk_forward_folds = []
        self.current_fold_index = 0
        self.label_percentile_threshold = 65
        self.label_min_context_samples = 25
        self.label_volatility_floor_pips = 5.0
        self.max_iterations = 5
        self.early_stopping_patience = 2
        self.no_improve_iterations = 0
        self.train_b_label_blocks = 3
        self.min_training_samples = MIN_TRAINING_SAMPLES
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
        self.min_open_symbol_confirmations = int(
            self.general_config.get('MinOpenSymbolConfirmations', self.min_open_symbol_confirmations)
        )
        self.walk_forward_enabled = bool(
            self.general_config.get('walk_forward_enabled', self.walk_forward_enabled)
        )
        self.validation_window_fraction = float(
            self.general_config.get('validation_window_fraction', self.validation_window_fraction)
        )
        self.validation_window_fraction = min(max(self.validation_window_fraction, 0.05), 0.95)
        self.walk_forward_min_train_fraction = float(
            self.general_config.get('walk_forward_min_train_fraction', self.walk_forward_min_train_fraction)
        )
        self.walk_forward_min_train_fraction = min(
            max(self.walk_forward_min_train_fraction, self.validation_window_fraction),
            0.95
        )
        step_fraction_cfg = self.general_config.get(
            'walk_forward_step_fraction',
            self.validation_window_fraction
        )
        self.walk_forward_step_fraction = min(max(float(step_fraction_cfg), 0.01), 0.95)
        self.max_iterations = max(1, int(self.general_config.get('max_iterations', self.max_iterations)))
        self.early_stopping_patience = max(
            0, int(self.general_config.get('early_stopping_patience', self.early_stopping_patience))
        )
        self.stop_loss = int(self.general_config.get('stop_loss', self.stop_loss))
        self.take_profit = int(self.general_config.get('take_profit', self.take_profit))
        data_start_is, _ = get_previous_4_6(
            self.general_config['dateStart'], 
            self.general_config['dateEnd']
        )
        self.data_start_is = data_start_is
        
    def prepare_base_data(self):

        df_is = pd.read_csv(f'output/symbol_data/{self.principal_symbol}/is_os/is.csv')
        df_os = pd.read_csv(f'output/symbol_data/{self.principal_symbol}/is_os/os.csv')

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

        df_base = df_base.sort_values('time').set_index('time')
        df_base = self.add_market_features(df_base)
        self.split_temporal_windows(df_base)


    def split_temporal_windows(self, df_base):
        if self.walk_forward_enabled:
            self.walk_forward_folds = self.build_walk_forward_folds(df_base)
            if self.walk_forward_folds:
                self.apply_walk_forward_fold(0)
                return

        if df_base.empty:
            self.df_train_A = df_base
            self.df_train_B = df_base
            self.df_valid = df_base
            return

        total_rows = len(df_base)
        valid_size = int(round(total_rows * self.validation_window_fraction))
        valid_size = max(1, valid_size)
        if total_rows > 1:
            valid_size = min(valid_size, total_rows - 1)
        df_valid = df_base.iloc[-valid_size:]
        df_pre_validation = df_base.iloc[:-valid_size]

        if df_pre_validation.empty:
            self.df_train_A = df_base.iloc[:0].copy()
            self.df_train_B = df_base.iloc[:0].copy()
            self.df_valid = df_valid
            return

        if len(df_pre_validation) == 1:
            split_idx = 1
        else:
            train_b_size = int(round(len(df_pre_validation) * self.train_b_fraction))
            train_b_size = max(1, min(train_b_size, len(df_pre_validation) - 1))
            split_idx = len(df_pre_validation) - train_b_size

        self.df_train_A = df_pre_validation.iloc[:split_idx]
        self.df_train_B = df_pre_validation.iloc[split_idx:]
        self.df_valid = df_valid

        print(
            "Ventanas temporales por porcentaje | "
            f"train_A: {self.df_train_A.index.min()} -> {self.df_train_A.index.max()} ({len(self.df_train_A)}) | "
            f"train_B: {self.df_train_B.index.min() if not self.df_train_B.empty else 'empty'} -> "
            f"{self.df_train_B.index.max() if not self.df_train_B.empty else 'empty'} ({len(self.df_train_B)}) | "
            f"valid: {self.df_valid.index.min()} -> {self.df_valid.index.max()} ({len(self.df_valid)}) | "
            f"valid_fraction: {self.validation_window_fraction:.2f}"
        )


    def build_walk_forward_folds(self, df_base):
        if df_base.empty:
            return []

        total_rows = len(df_base)
        valid_size = int(round(total_rows * self.validation_window_fraction))
        valid_size = max(1, valid_size)
        if total_rows > 1:
            valid_size = min(valid_size, total_rows - 1)

        min_train_size = int(round(total_rows * self.walk_forward_min_train_fraction))
        min_train_size = max(valid_size, min_train_size)
        if total_rows > 1:
            min_train_size = min(min_train_size, total_rows - 1)

        step_size = int(round(total_rows * self.walk_forward_step_fraction))
        step_size = max(1, step_size)

        folds = []
        valid_start_idx = min_train_size

        while valid_start_idx < total_rows:
            valid_end_idx = min(total_rows, valid_start_idx + valid_size)
            df_train = df_base.iloc[:valid_start_idx]
            df_valid = df_base.iloc[valid_start_idx:valid_end_idx]

            if not df_train.empty and not df_valid.empty:
                if len(df_train) == 1:
                    split_idx = 1
                else:
                    train_b_size = int(round(len(df_train) * self.train_b_fraction))
                    train_b_size = max(1, min(train_b_size, len(df_train) - 1))
                    split_idx = len(df_train) - train_b_size

                fold = {
                    "train_A": df_train.iloc[:split_idx],
                    "train_B": df_train.iloc[split_idx:],
                    "valid": df_valid,
                    "meta": {
                        "train_start": df_train.index.min(),
                        "train_end": df_train.index.max(),
                        "valid_start": df_valid.index.min(),
                        "valid_end": df_valid.index.max(),
                    },
                }
                folds.append(fold)

            if valid_end_idx >= total_rows:
                break
            valid_start_idx = valid_start_idx + step_size

        if folds:
            print(
                "Walk-forward por porcentaje | "
                f"folds: {len(folds)} | "
                f"valid_fraction: {self.validation_window_fraction:.2f} | "
                f"min_train_fraction: {self.walk_forward_min_train_fraction:.2f} | "
                f"step_fraction: {self.walk_forward_step_fraction:.2f}"
            )
        return folds


    def apply_walk_forward_fold(self, fold_index):
        if not self.walk_forward_folds:
            return

        bounded_index = max(0, min(fold_index, len(self.walk_forward_folds) - 1))
        fold = self.walk_forward_folds[bounded_index]
        self.current_fold_index = bounded_index
        self.df_train_A = fold["train_A"]
        self.df_train_B = fold["train_B"]
        self.df_valid = fold["valid"]

        print(
            "Walk-forward fold activo | "
            f"fold={bounded_index + 1}/{len(self.walk_forward_folds)} | "
            f"train: {fold['meta']['train_start']} -> {fold['meta']['train_end']} ({len(self.df_train_A) + len(self.df_train_B)}) | "
            f"valid: {fold['meta']['valid_start']} -> {fold['meta']['valid_end']} ({len(self.df_valid)})"
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

        for symbol in self.list_symbols:
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
        
        std_monthly = float(np.std(month_values)) if len(month_values) > 1 else 0.0
        std_quarterly = float(np.std(quarter_values)) if len(quarter_values) > 1 else 0.0

        return {
            "n_months": len(month_values),
            "n_quarters": len(quarter_values),
            "positive_month_ratio": positive_month_ratio,
            "positive_quarter_ratio": positive_quarter_ratio,
            "best_month_pips": float(max(month_values)) if month_values else 0.0,
            "worst_month_pips": float(min(month_values)) if month_values else 0.0,
            "best_quarter_pips": float(max(quarter_values)) if quarter_values else 0.0,
            "worst_quarter_pips": float(min(quarter_values)) if quarter_values else 0.0,
            "std_monthly": std_monthly,
            "std_quarterly": std_quarterly,
            "monthly_pips": monthly_pips,
            "quarterly_pips": quarterly_pips,
        }

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
                        "vol_10": float(row.vol_10),
                        "zscore_20": float(row.zscore_20),
                        "momentum_ratio": float(row.momentum_ratio),
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
            'vol_10': [],
            'zscore_20': [],
            'momentum_ratio': [],
            'output': []
        }

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
            data['vol_10'].append(sample['vol_10'])
            data['zscore_20'].append(sample['zscore_20'])
            data['momentum_ratio'].append(sample['momentum_ratio'])
            data['output'].append(label)

        df_dataset = pd.DataFrame(data)

        return df_dataset


    def preload_data(self):

        ini = time.time()

        # CLOSE
        FILES_OS_CLOSE = {
            f.split('_')[0]: f
            for f in os.listdir(f'output/symbol_data/{self.principal_symbol}/extrac_os')
        }

        for nodo in self.nodos_close:

            path_is = f'output/symbol_data/{self.principal_symbol}/extrac/{nodo["file"]}'
            file_base = nodo["file"].split('_')[0]
            file_os = FILES_OS_CLOSE[file_base]
            path_os = f'output/symbol_data/{self.principal_symbol}/extrac_os/{file_os}'

            self.COMBINED[("close", nodo["file"])] = self.build_combined(path_is, path_os)

        # OPEN
        for symbol in self.list_symbols:

            if symbol == self.principal_symbol:
                path = f'output/symbol_data/{self.principal_symbol}'
                path_os_root = f'{path}/extrac_os'
            else:
                path = f'output/symbol_data/{symbol}'
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
        nn.fit(
            input1_ids,
            input2_ids,
            hour_ids,
            X_extra,
            Y,
            epochs=80,
            batch_size=64,
        )

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
        input1_ids, input2_ids, hour_ids, X_extra, _ = load_data(path_data_red)

        nn = load_trained_model(
            f'output/{self.principal_symbol}/data_for_neuronal/model_trainer/model_{self.mercado}_{self.algorithm}.pt',
            input_dim_extra=X_extra.shape[1]
        )
        validate_embedding_vocab(nn, input1_ids, input2_ids, hour_ids)

        df_validation = self.df_valid
        validation_start = df_validation.index.min()
        validation_end = df_validation.index.max()
        print(
            f"Validando sobre el {self.validation_window_fraction:.2f} del dataset: "
            f"{validation_start} -> {validation_end} | barras={len(df_validation)}"
        )

        is_open = False
        open_price_open = 0.0
        spread_open = 0.0
        entry_red_open = []
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
        close_threshold = self.close_threshold_floor

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
                            for entry in entry_red_open:
                                prob = predict_from_inputs(nn, entry, nodo_close, hour, market_features)
                                if prob > close_threshold:
                                    model_close_signal = True
                                    break
                            if model_close_signal:
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

                    if VERBOSE_TRADE_LOGS:
                        print(f"SUM PIPS:{time_comienzo} ---- {time_actual}: {sum_pips}", cierre)
                    elif cantidad_operaciones % TRADE_LOG_EVERY == 0:
                        print(
                            f"RESUMEN {self.principal_symbol} {self.mercado}_{self.algorithm} | "
                            f"ops={cantidad_operaciones} | pips={sum_pips:.2f}"
                        )

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
                    entry_red_open = open_nodes

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

        best_thr = float(self.close_threshold_floor)
        threshold_info = {
            "threshold": best_thr,
            "raw_threshold": best_thr,
            "score": 0.0,
            "calibration_trades": int(len(lista_pips)),
            "scoring_trades": int(len(lista_pips)),
            "smoothed": False,
        }
        best_threshold_score = 0.0
        mc_score = 1.0
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
        Monte Carlo Stability: {mc_score:.4f}
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
            "monte_carlo_score": mc_score,
            "validation_window": {
                "fraction": float(self.validation_window_fraction),
                "start": validation_start.isoformat() if hasattr(validation_start, "isoformat") else str(validation_start),
                "end": validation_end.isoformat() if hasattr(validation_end, "isoformat") else str(validation_end),
                "bars": int(len(df_validation)),
            },
            "walk_forward": {
                "enabled": bool(self.walk_forward_folds),
                "fold_index": int(self.current_fold_index + 1) if self.walk_forward_folds else 1,
                "total_folds": int(len(self.walk_forward_folds)) if self.walk_forward_folds else 1,
            },
            "walk_forward_penalty": 1.0,
            "cantidad_operaciones": cantidad_operaciones,
            "sum_pips": sum_pips,
            "mas_perdidas_seguidas": mas_perdidas_seguidas
        }
        return metrics
    

    def calculate_preliminary_score(self, metrics):
        profit_factor_adj = min(metrics["profit_factor"], 5)
        sharpe_adj = min(metrics["sharpe"], 3)
        n_ops = metrics["cantidad_operaciones"]
        score_base = (
            (metrics["expectancy"] * 2.0) +
            (profit_factor_adj * 1.5) +
            (sharpe_adj * 1.5) +
            (metrics["winrate"] * 1.0)
        )

        volumen_factor = np.log(n_ops + 1)
        score = score_base * volumen_factor

        return {
            "score": float(score),
            "score_base": float(score_base),
            "volumen_factor": float(volumen_factor),
            "reliability_factor": 1.0,
            "low_trade_penalty": 1.0,
            "temporal_penalty": 1.0,
            "monte_carlo_penalty": 1.0,
            "walk_forward_penalty": 1.0,
        }


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

        if preliminary["score"] > self.best_score:
            self.best_score = preliminary["score"]
            self.no_improve_iterations = 0
            self.best_model_data = model_data
            best_metrics = dict(metrics)
            self.peor_trade = min((item["pips"] for item in best_metrics["lista_pips"]), default=0)
            best_metrics["peor_trade"] = self.peor_trade
            self.info_score = best_metrics
            if "lista_pips" in self.info_score:
                del self.info_score["lista_pips"]
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

        total_iterations = self.max_iterations
        if self.walk_forward_folds:
            total_iterations = min(self.max_iterations, len(self.walk_forward_folds))

        for i in range(total_iterations):
            print(f"\n--- Iteración {i+1} ---")
            self.index = i + 1

            if self.walk_forward_folds:
                self.apply_walk_forward_fold(i)

            try:
                model_data = self.train_iteration()
            except ValueError as exc:
                print(f"Backtester cancelado por dataset insuficiente: {exc}")
                self.best_model_data = None
                self.info_score = {
                    "skipped": True,
                    "skip_reason": str(exc),
                }
                break
            last_model_data = model_data
            metrics = self.validate_iteration()
            self.calculate_score(metrics, model_data)

            if (
                self.early_stopping_patience > 0
                and self.no_improve_iterations >= self.early_stopping_patience
            ):
                print(
                    "Early stopping activado: "
                    f"{self.no_improve_iterations} iteraciones elegibles sin mejora."
                )
                break

        if self.best_model_data is None and last_model_data is not None:
            self.best_model_data = last_model_data
        if not self.info_score and self.iteration_candidates:
            fallback_metrics = dict(self.iteration_candidates[-1]["metrics"])
            fallback_metrics["score"] = fallback_metrics.get("preliminary_score", 0.0)
            fallback_metrics["score_model"] = "simple_preliminary_fallback"
            fallback_metrics["peor_trade"] = min((item["pips"] for item in fallback_metrics["lista_pips"]), default=0)
            self.info_score = fallback_metrics
            if "lista_pips" in self.info_score:
                del self.info_score["lista_pips"]

        if self.best_model_data is None:
            with open(f'output/{self.principal_symbol}/data_for_neuronal/best_score/score_{self.mercado}_{self.algorithm}.json', 'w') as f:
                json.dump({
                    "metrics": _sanitize_for_json(self.info_score)
                }, f, indent=4, allow_nan=False)
            print(f'tiempo total: {time.time() - self.comienzo:.2f} segundos')
            return

        torch_model_path = f'output/{self.principal_symbol}/data_for_neuronal/model_trainer/model_{self.mercado}_{self.algorithm}.pt'
        torch.save(self.best_model_data, torch_model_path)
        with open(f'output/{self.principal_symbol}/data_for_neuronal/best_score/score_{self.mercado}_{self.algorithm}.json', 'w') as f:
            json.dump({
                "metrics": _sanitize_for_json(self.info_score)
            }, f, indent=4, allow_nan=False)
        print(f'tiempo total: {time.time() - self.comienzo:.2f} segundos')
    
       
      
if __name__ == "__main__":
    inn = time.time()
    backtester = Backtester('AUDCHF', 'Asia', 'DOWN')
    backtester.run()
    print(f'segundos {time.time()-inn}')
    