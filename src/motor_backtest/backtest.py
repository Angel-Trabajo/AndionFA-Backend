import os
import sys
import json
import operator
import ast
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.routes.peticiones import get_historical_data, get_timeframes
from src.utils.indicadores_for_principal_script import generate_files
from src.db.query import get_nodes_by_label
from src.neuronal.entrenar import load_trained_model, predict_from_inputs, load_data, BinaryNN
from src.utils.common_functions import hora_en_mercado, crear_carpeta_si_no_existe
from src.neuronal.backtester import Backtester


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


class Backtest:
    # Cache compartido: {(principal_symbol, date_start, date_end): {base_data, nodes, indicators}}
    _shared_cache = {}

    @staticmethod
    def _get_cache_key(principal_symbol, date_start, date_end):
        return (principal_symbol, date_start, date_end)

    @staticmethod
    def clear_cache(principal_symbol=None):
        """Libera memoria del caché. Si se indica symbol, solo borra ese symbol.
        Si no, limpia todo el caché."""
        if principal_symbol is None:
            Backtest._shared_cache.clear()
            print("🗑️ Caché completo liberado")
        else:
            keys_to_delete = [
                k for k in Backtest._shared_cache
                if k[0] == principal_symbol
            ]
            for k in keys_to_delete:
                del Backtest._shared_cache[k]
            print(f"🗑️ Caché liberado para {principal_symbol} ({len(keys_to_delete)} entradas)")

    @staticmethod
    def generate_global_results(root_dir='output/x_backtest_results'):
        all_dfs = []

        if not os.path.exists(root_dir):
            print(f"No existe la carpeta de resultados: {root_dir}")
            return None, None

        for symbol in os.listdir(root_dir):
            symbol_dir = os.path.join(root_dir, symbol)
            if not os.path.isdir(symbol_dir) or symbol == 'general':
                continue

            for mercado_alg in os.listdir(symbol_dir):
                bt_dir = os.path.join(symbol_dir, mercado_alg)
                if not os.path.isdir(bt_dir):
                    continue

                results_path = os.path.join(bt_dir, 'results.csv')
                if not os.path.exists(results_path):
                    continue

                try:
                    df = pd.read_csv(results_path)
                except Exception as e:
                    print(f"⚠️ No se pudo leer {results_path}: {e}")
                    continue

                if 'pips' not in df.columns:
                    continue

                if '_' in mercado_alg:
                    mercado, algorithm = mercado_alg.rsplit('_', 1)
                else:
                    mercado, algorithm = mercado_alg, 'N/A'

                df['principal_symbol'] = symbol
                df['mercado'] = mercado
                df['algorithm'] = algorithm
                df['backtest_name'] = mercado_alg
                df['pips'] = pd.to_numeric(df['pips'], errors='coerce').fillna(0.0)
                if 'time_open' in df.columns:
                    df['time_open'] = pd.to_datetime(df['time_open'], errors='coerce')
                else:
                    df['time_open'] = pd.NaT
                if 'time_close' in df.columns:
                    df['time_close'] = pd.to_datetime(df['time_close'], errors='coerce')
                else:
                    df['time_close'] = pd.NaT

                all_dfs.append(df[[
                    'principal_symbol', 'mercado', 'algorithm', 'backtest_name',
                    'time_open', 'time_close', 'pips'
                ]])

        if not all_dfs:
            print("No se encontraron resultados para consolidar.")
            return None, None

        df_general = pd.concat(all_dfs, ignore_index=True)
        df_general = df_general.reset_index(drop=True)
        df_general['operation_id'] = np.arange(1, len(df_general) + 1)

        use_dates = df_general['time_close'].notna().all()
        if use_dates:
            df_general = df_general.sort_values('time_close').reset_index(drop=True)
            x = df_general['time_close']
            x_label = 'Fecha'
        else:
            x = np.arange(1, len(df_general) + 1)
            x_label = 'Número de operación global'

        df_general['pips_acumulados'] = df_general['pips'].cumsum()

        general_dir = os.path.join(root_dir, 'general')
        crear_carpeta_si_no_existe(general_dir)

        csv_path = os.path.join(general_dir, 'results_general.csv')
        df_general.to_csv(csv_path, index=False)

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(x, df_general['pips_acumulados'], color='#1f77b4', linewidth=1.8)
        ax.scatter(x, df_general['pips_acumulados'], color='#d62728', s=12, zorder=3, label='Operaciones')
        ax.axhline(0, color='#000000', linewidth=2.0, linestyle='--', alpha=0.75, label='Cero pips')

        ax.set_title(f"Resultado global de backtests | operaciones: {len(df_general)}")
        ax.set_xlabel(x_label)
        ax.set_ylabel('Pips acumulados')
        ax.grid(True, alpha=0.3)

        if use_dates:
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)

        ax.legend()
        fig.tight_layout()

        plot_path = os.path.join(general_dir, 'results_general_plot.png')
        fig.savefig(plot_path, dpi=140)
        plt.close(fig)

        print(f"Archivo general guardado en {csv_path}")
        print(f"Gráfico general guardado en {plot_path}")

        return csv_path, plot_path

    def _get_score_path(self):
        return f'output/{self.principal_symbol}/data_for_neuronal/best_score/score_{self.mercado}_{self.algorithm}.json'

    def _load_best_score(self):
        score_path = self._get_score_path()
        self.best_score = {}
        if os.path.exists(score_path) and os.path.getsize(score_path) > 0:
            try:
                with open(score_path, 'r', encoding='utf-8') as f:
                    self.best_score = json.load(f)
            except Exception as e:
                print(f"⚠️ No se pudo cargar best_score, se usan defaults: {e}")

    def _apply_conditioning_from_best_score(self):
        winner = self.best_score.get('winner', {}) if isinstance(self.best_score, dict) else {}
        peor_trade = winner.get('peor_trade', -80)
        maxima_perdidas = winner.get('mas_perdidas_seguidas', 6)
        try:
            peor_trade = float(peor_trade)
        except Exception:
            peor_trade = -80.0
        try:
            maxima_perdidas = int(maxima_perdidas)
        except Exception:
            maxima_perdidas = 6

        self.stop_loss_pips = peor_trade if peor_trade < 0 else -80.0
        self.maxima_perdidas = max(5, maxima_perdidas)

    def _refresh_conditioning_from_score(self):
        self._load_best_score()
        self._apply_conditioning_from_best_score()
    
    def __init__(self, principal_symbol, mercado, algorithm, date_start, date_end):
        self.principal_symbol = principal_symbol
        self.mercado = mercado
        self.algorithm = algorithm
        self.date_start = date_start
        self.date_end = date_end
        
        with open('config/general_config.json', 'r', encoding='utf-8') as f:
            self.general_config = json.load(f)
        with open(f'config/divisas/{self.principal_symbol}/config_{self.principal_symbol}.json', 'r', encoding='utf-8') as f:
            self.config_symbol = json.load(f)
        self.best_score = {}
        self._load_best_score()
            
        if self.algorithm == "UP":
            self.other_algorithm = "DOWN"
        else:
            self.other_algorithm = "UP" 
        self.horas_mercado = [hora_en_mercado(h, self.mercado) for h in range(24)]    
        self.list_symbols = self.config_symbol['list_symbol']
        self.list_symbols.insert(0, self.principal_symbol)
        self.timeframe = get_timeframes().get(self.general_config['timeframe'])
        
        # ====== CACHÉ COMPARTIDO ======
        cache_key = self._get_cache_key(principal_symbol, date_start, date_end)
        if cache_key not in Backtest._shared_cache:
            print(f"📥 Cargando datos base para {principal_symbol} (NEW)")
            Backtest._shared_cache[cache_key] = {
                "base_data": self.prepare_base_data(),
                "indicators": {}
            }
        
        shared = Backtest._shared_cache[cache_key]
        self.base_data = shared["base_data"]
        self.indicators = shared["indicators"]
        
        self.pip_size, self.point_size = get_pip_and_point_size(
            self.principal_symbol,
            self.base_data['open'] if 'open' in self.base_data.columns else self.base_data['close']
        )
        self.results = {
            "time_open": [],
            "time_close": [],
            "pips": []
        }
        # Parámetros de cierre por edge (alineados con entrenamiento)
        self.edge_threshold = float(self.general_config.get('edge_threshold', 0.01))
        self.edge_decay_factor = float(self.general_config.get('edge_decay_factor', 0.6))
        self.edge_delta_threshold = float(self.general_config.get('edge_delta_threshold', -0.05))
        self.min_open_edge = float(self.general_config.get('min_open_edge', -0.05))
        self.max_loss_close_pips = float(self.general_config.get('max_loss_close_pips', -15.0))
        self.hard_stop_pips = float(self.general_config.get('hard_stop_pips', -25.0))
        self.take_profit_pips = float(self.general_config.get('take_profit_pips', 120.0))
        self.max_trade_duration = int(self.general_config.get('max_trade_duration', 150))
        self.stop_loss_pips = float(self.general_config.get('stop_loss_pips', -50.0))
        self.maxima_perdidas = int(self.general_config.get('maxima_perdidas', 6))
        self._apply_conditioning_from_best_score()
        
        self.setup_operators_and_mappings()
        
        # ====== NODES (NO CACHÉ: dependen del algoritmo) ======
        print(f"📥 Cargando nodes para {principal_symbol} {algorithm}")
        self.load_nodes()
        
        # ====== INDICATORS COMPARTIDOS (algoritmo-agnóstico) ======
        if not self.indicators:
            print(f"📥 Calculando indicadores para {principal_symbol} (NEW)")
            self.calculate_indicators()
        else:
            print(f"♻️ Reutilizando indicadores para {principal_symbol}")
        
        
    def prepare_base_data(self):    
        data = get_historical_data(self.principal_symbol, self.timeframe, self.date_start, self.date_end)
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df


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
    
    
    def load_nodes(self):

        self.dict_nodos = {}

        for i, symbol in enumerate(self.list_symbols):
            self.dict_nodos[symbol] = self.parsear_nodos(
                get_nodes_by_label(self.principal_symbol, symbol, self.mercado, self.algorithm)
            )

        self.nodos_close = self.parsear_nodos(
            get_nodes_by_label(self.principal_symbol, self.principal_symbol, mercado=None, label=self.other_algorithm)
        )

    
    def build_combined(self, df):

        return {
            "df": df,
            "values": df.values,
            "col_map": {col: i for i, col in enumerate(df.columns)},
            "index_values": df.time.values
        }

    
    def calculate_indicators(self):
        print(f"Calculando indicadores para {self.principal_symbol}...")
        list_files = self.general_config['indicators_files']
        for symbol in self.list_symbols:
            data = get_historical_data(symbol, self.timeframe, self.date_start, self.date_end)
            df = pd.DataFrame(data)
            df['time'] = pd.to_datetime(df['time'], unit='s')   
            for file in list_files:
                indicator = generate_files(file, df)
                indicator = indicator[indicator['time'].between(self.date_start, self.date_end)]
                self.indicators[f'{self.principal_symbol}_{symbol}_{file.split(".")[0]}'] = self.build_combined(indicator)
     
      
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
        Ejecuta SL/TP usando HIGH/LOW intrabar.
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


    def compute_best_open_edge(self, nn, entry_open_signal, time_actual_np, time_actual, spread_open, market_ctx_open):
        open_candidates = []

        for nodo in self.nodos_close:
            df_struct = self.indicators[f'{self.principal_symbol}_{self.principal_symbol}_{nodo["file"].split("_")[0]}']
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
                value = row[col_map[alias]]
                if value is None or pd.isna(value):
                    continue
                try:
                    return float(value)
                except Exception:
                    continue
        return float(default)


    def _get_market_context(self, df_struct, row_idx, time_actual, spread_open):
        atr = self._get_value_from_struct(
            df_struct,
            row_idx,
            ["ATR_23", "ATR23", "ATR", "atr_23", "atr"],
            default=0.0,
        )
        adx = self._get_value_from_struct(
            df_struct,
            row_idx,
            ["ADX_20", "ADX20", "ADX", "adx_20", "adx"],
            default=0.0,
        )
        rsi = self._get_value_from_struct(
            df_struct,
            row_idx,
            ["RSI_21", "RSI21", "RSI", "rsi_21", "rsi"],
            default=0.0,
        )
        stoch = self._get_value_from_struct(
            df_struct,
            row_idx,
            ["STOCH_14_3_SMA_3_SMA_pos0", "STOCH_14_3_3", "STOCH", "stoch"],
            default=50.0,
        )

        close_curr = self._get_value_from_struct(df_struct, row_idx, ["close", "Close"], 0.0)
        open_curr = self._get_value_from_struct(df_struct, row_idx, ["open", "Open"], 0.0)
        high_curr = self._get_value_from_struct(df_struct, row_idx, ["high", "High"], 0.0)
        low_curr = self._get_value_from_struct(df_struct, row_idx, ["low", "Low"], 0.0)

        returns_1 = (close_curr - open_curr) / max(abs(open_curr), 1e-6)
        volatility = (high_curr - low_curr) / max(abs(close_curr), 1e-6)
        trend = 1.0 if close_curr > open_curr else -1.0

        return {
            "atr": atr,
            "adx": adx,
            "rsi": rsi,
            "stoch": stoch,
            "hour": float(time_actual.hour),
            "spread": float(spread_open),
            "returns_1": float(returns_1),
            "volatility": float(volatility),
            "trend": float(trend),
        }
    
    
    def test_iteration(self):

        path_data_red = f'output/{self.principal_symbol}/data_for_neuronal/data/data_{self.mercado}_{self.algorithm}.csv'
        model_path = f'output/{self.principal_symbol}/data_for_neuronal/model_trainer/model_{self.mercado}_{self.algorithm}.json'
        fallback_input_dim = 26  # 16 binary + 6 context_close + 4 context_open

        input_dim = fallback_input_dim
        if os.path.exists(path_data_red):
            try:
                X, Y = load_data(path_data_red)
                if len(X) > 0:
                    input_dim = X.shape[1]
            except Exception as e:
                print(f"⚠️ Dataset inválido para inferencia, se usa fallback dim={fallback_input_dim}: {e}")

        if os.path.exists(model_path):
            try:
                nn = load_trained_model(model_path, input_dim=input_dim)
            except Exception as e:
                print(f"⚠️ Modelo inválido, usando fallback sin entrenar: {e}")
                nn = BinaryNN(input_dim=input_dim, lr=0.01, target_loss=0.10)
                nn.model = None
        else:
            print(f"⚠️ Modelo no encontrado, usando fallback sin entrenar: {model_path}")
            nn = BinaryNN(input_dim=input_dim, lr=0.01, target_loss=0.10)
            nn.model = None
        base_data = self.base_data[self.base_data['time'].between(self.date_start, self.date_end)].copy()

        is_open = False
        open_price_open = 0.0
        spread_open = 0.0
        entry_red_open = ''
        cierre = 0
        time_comienzo = None
        market_ctx_open = {}
        pred_open_edge = None
        perdidas_seguidas = 0

        for row in base_data.itertuples():

            time_actual = row.time
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
                cerrar = False
                reason = ""

                valid_closes = []

                for nodo in self.nodos_close:
                    df_struct = self.indicators[f'{self.principal_symbol}_{self.principal_symbol}_{nodo["file"].split("_")[0]}']
                    pos = df_struct["index_values"].searchsorted(time_actual_np)
                    if pos == 0:
                        continue

                    if self.cumple_condiciones_fast(df_struct, pos - 1, nodo["conditions"]):
                        nodo_close = self.maping_close[nodo["key"]]
                        market_ctx = self._get_market_context(df_struct, pos - 1, time_actual, spread_open)
                        _, prob, pred = predict_from_inputs(
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
                            returns_1=market_ctx.get("returns_1", 0.0),
                            volatility=market_ctx.get("volatility", 0.0),
                            trend=market_ctx.get("trend", 0.0),
                            return_raw=True,
                        )
                        valid_closes.append((nodo_close, float(prob), float(pred)))

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

                if is_open and cierre % 20 == 0:
                    print(f"Trade abierto desde {time_comienzo} | duración={cierre}")

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
                    best_close = max(valid_closes, key=lambda x: x[2])
                    best_pred = best_close[2]

                    if pred_open_edge is None:
                        pred_open_edge = best_pred

                    delta_pred = best_pred - pred_open_edge
                    close_by_low_edge = best_pred < self.edge_threshold
                    close_by_decay = best_pred < (pred_open_edge * self.edge_decay_factor)
                    close_by_delta = delta_pred < self.edge_delta_threshold
                    cerrar = bool(close_by_low_edge or close_by_decay or close_by_delta)

                    if cerrar:
                        reason = f"MODEL_EDGE ({best_pred:.3f})"

                if cerrar:
                    if trade_pips < 0:
                        perdidas_seguidas += 1
                    else:
                        perdidas_seguidas = 0    
                    self.results["time_open"].append(time_comienzo)
                    self.results["time_close"].append(time_actual)
                    self.results["pips"].append(trade_pips)
                    is_open = False
                    cierre = 0
                    pred_open_edge = None
                    print(f"CLOSE [{reason}] {time_comienzo} -> {time_actual} | pips={trade_pips:.2f}")
                    if perdidas_seguidas >= self.maxima_perdidas:
                        # time_str = time_actual.strftime('%Y-%m-%d')
                        # bachtester = Backtester(self.principal_symbol, self.mercado, self.algorithm, time_str)
                        # bachtester.run()
                        self._refresh_conditioning_from_score()
                        print(
                            f"🔄 Parámetros actualizados desde best_score | "
                            f"stop_loss_pips={self.stop_loss_pips:.2f} | "
                            f"maxima_perdidas={self.maxima_perdidas}"
                        )
                        perdidas_seguidas = 0
                        
            # =========================
            # APERTURA
            # =========================
            else:

                cierre = 0

                for symbol in self.list_symbols:

                    cumple_alguno = False
                    nodo_open_list = self.dict_nodos[symbol]

                    for nodo in nodo_open_list:

                        df_struct = self.indicators[f'{self.principal_symbol}_{symbol}_{nodo["file"].split("_")[0]}']
                        pos = df_struct["index_values"].searchsorted(time_actual_np)
                        
                        if pos == 0:
                            continue

                        if self.cumple_condiciones_fast(df_struct, pos - 1, nodo["conditions"]):

                            cumple_alguno = True

                            if symbol == self.list_symbols[-1]:
                                nodo_open = self.maping_open[nodo["key"]]
                                spread_candidate = float(getattr(row, 'spread', 0.0))
                                market_ctx_candidate = self._get_market_context(df_struct, pos - 1, time_actual, spread_candidate)

                                _, _, open_pred = self.compute_best_open_edge(
                                    nn,
                                    nodo_open,
                                    time_actual_np,
                                    time_actual,
                                    spread_candidate,
                                    market_ctx_candidate,
                                )

                                if open_pred is not None and open_pred < self.min_open_edge:
                                    continue

                                open_price_open = open_price
                                spread_open = spread_candidate
                                is_open = True
                                time_comienzo = time_actual
                                entry_red_open = nodo_open
                                market_ctx_open = market_ctx_candidate
                                pred_open_edge = open_pred if open_pred is not None else 0.0

                            break

                    if not cumple_alguno:
                        break

        return self.results


    def plot_results(self, results, output_dir):
        pips = results.get("pips", [])
        if not pips:
            print("No hay operaciones para graficar.")
            return None

        pips_acumulados = np.cumsum(pips)
        fechas_cierre = pd.to_datetime(results.get("time_close", []), errors="coerce")
        usar_fechas = len(fechas_cierre) == len(pips_acumulados) and not fechas_cierre.isna().any()

        if usar_fechas:
            x = fechas_cierre
            x_label = "Fecha"
        else:
            x = np.arange(1, len(pips_acumulados) + 1)
            x_label = "Número de operación"

        n_operaciones = len(pips)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(x, pips_acumulados, color="#1f77b4", linewidth=1.8)
        ax.scatter(x, pips_acumulados, color="#d62728", s=18, zorder=3, label="Operaciones")
        ax.axhline(0, color="#000000", linewidth=2.0, linestyle="--", alpha=0.75, label="Cero pips")

        ax.set_title(
            f"Backtest {self.principal_symbol} - {self.mercado} - {self.algorithm} | Operaciones: {n_operaciones}"
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel("Pips acumulados")
        ax.grid(True, alpha=0.3)

        if usar_fechas:
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)

        ax.legend()

        image_path = f"{output_dir}/results_plot.png"
        fig.tight_layout()
        fig.savefig(image_path, dpi=140)
        plt.close(fig)

        print(f"Gráfica guardada en {image_path}")
        return image_path
    
    
    def run(self):
        crear_carpeta_si_no_existe(f'output/y_backtest_results')
        crear_carpeta_si_no_existe(f'output/y_backtest_results/{self.principal_symbol}')
        crear_carpeta_si_no_existe(f'output/y_backtest_results/{self.principal_symbol}/{self.mercado}_{self.algorithm}')
        output_dir = f'output/y_backtest_results/{self.principal_symbol}/{self.mercado}_{self.algorithm}'
        results = self.test_iteration()
        df_results = pd.DataFrame(results)
        df_results.to_csv(f'{output_dir}/results.csv', index=False)
        print(f"Resultados guardados en {output_dir}/results.csv")
        self.plot_results(results, output_dir)
        
    
if __name__ == "__main__":
    
    
    # Ejemplo de uso
    with open('config/general_config.json', 'r', encoding='utf-8') as f:
        general_config = json.load(f)
    list_principal_symbols = general_config["list_principal_symbols"]
    mercados = ["Asia", "Europa", "America"]
    algorithms = ["UP", "DOWN"]
    date_start = "2023-01-01"
    date_end = "2025-01-01"
    for principal_symbol in list_principal_symbols:
        for mercado in mercados:
            for algorithm in algorithms:
                backtest = Backtest(principal_symbol, mercado, algorithm, date_start, date_end)
                backtest.run()
        # Liberar caché al terminar todos los algoritmos del symbol
        Backtest.clear_cache(principal_symbol)

    Backtest.generate_global_results('output/y_backtest_results')