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

from src.routes.peticiones import get_historical_data, get_timeframes, initialize_mt5
from src.utils.indicadores_for_principal_script import generate_files
from src.db.query import get_nodes_by_label
from src.neuronal.entrenar import EXTRA_FEATURE_COLUMNS, load_trained_model, predict_from_inputs, load_data, validate_embedding_vocab
from src.signals.event_generator import add_event_features, has_entry_event
from src.utils.common_functions import hora_en_mercado, crear_carpeta_si_no_existe, should_backtest_strategy


VERBOSE_TRADE_LOGS = os.getenv("VERBOSE_TRADE_LOGS", "0") == "1"
TRADE_LOG_EVERY = max(1, int(os.getenv("TRADE_LOG_EVERY", "25")))


def plot_all_backtests_results(base_dir='output/x_backtest_results'):
    if not os.path.exists(base_dir):
        print(f"No existe la carpeta base: {base_dir}")
        return None
    
    result_files = []
    for root, _, files in os.walk(base_dir):
        if 'results.csv' in files:
            result_files.append(os.path.join(root, 'results.csv'))

    if not result_files:
        print("No se encontraron archivos results.csv para graficar.")
        return None
    df_all_results = pd.DataFrame()
    for file in result_files:
        df = pd.read_csv(file)
        df['time_open'] = pd.to_datetime(df['time_open'], errors='coerce')
        df['time_close'] = pd.to_datetime(df['time_close'], errors='coerce')
        df_all_results = pd.concat([df_all_results, df], ignore_index=True)
    df_all_results.sort_values(by='time_close', inplace=True)
    df_all_results['pips_acumulados'] = df_all_results['pips'].cumsum()
    df_all_results.to_csv(f'{base_dir}/all_results.csv', index=False)
    
   
    x = df_all_results['time_close']
    pips_acumulados = df_all_results['pips_acumulados']
    x_label = "Fecha"
  
    
    fig, ax = plt.subplots(figsize=(24, 12))
    ax.plot(x, pips_acumulados, color="#0cf35d", linewidth=1.8)
    ax.scatter(x, pips_acumulados, color="#1935ea", s=18, zorder=3, label="Operaciones")
    ax.axhline(0, color="#000000", linewidth=2.0, linestyle="--", alpha=0.75, label="Cero pips")

    ax.set_title(
        f"Backtest general | Operaciones: {len(df_all_results)}"
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Pips acumulados")
    ax.grid(True, alpha=0.3)

    
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.legend()

    image_path = f"{base_dir}/all_results_plot.png"
    fig.tight_layout()
    fig.savefig(image_path, dpi=140)
    plt.close(fig)

    print(f"Gráfica guardada en {image_path}")
    return image_path

    
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
    
    def __init__(self, principal_symbol, mercado, algorithm, date_start, date_end):
        self.principal_symbol = principal_symbol
        self.mercado = mercado
        self.algorithm = algorithm
        self.closed_trades_count = 0
        self.closed_pips_total = 0.0
        self.date_start = date_start
        self.date_end = date_end
        self.indicators = {} 
       
        
        with open('config/general_config.json', 'r', encoding='utf-8') as f:
            self.general_config = json.load(f)
        with open(f'config/divisas/{self.principal_symbol}/config_{self.principal_symbol}.json', 'r', encoding='utf-8') as f:
            self.config_symbol = json.load(f)
            
        if self.algorithm == "UP": 
            self.other_algorithm = "DOWN"
        else:
            self.other_algorithm = "UP" 
        
        self.horas_mercado = [hora_en_mercado(h, self.mercado) for h in range(24)]    
        self.list_symbols = self.config_symbol['list_symbol']
        self.list_symbols.insert(0, self.principal_symbol)
        self.timeframe = get_timeframes()['timeframes'].get(self.general_config['timeframe'])
        self.base_data = self.prepare_base_data()
        self.pip_size, self.point_size = get_pip_and_point_size(
            self.principal_symbol,
            self.base_data['open'] if 'open' in self.base_data.columns else self.base_data['close']
        )
        self.results = {
            "time_open": [],
            "time_close": [],
            "pips": [],
            "bars_held": [],
            "close_reason": [],
        }
        self.stop_loss = int(self.general_config.get('stop_loss', 20))
        self.take_profit = int(self.general_config.get('take_profit', 150))
        self.max_holding = 120
        self.min_model_holding = 15
        self.close_confirmation_bars = 2
        self.close_threshold_floor = 0.60
        self.min_open_symbol_confirmations = int(
        self.general_config.get('MinOpenSymbolConfirmations', 4)
        )
        
        self.setup_operators_and_mappings()
        self.load_nodes()
        self.calculate_indicators()
        
        
    def prepare_base_data(self):   
        result = get_historical_data(self.principal_symbol, self.timeframe, self.date_start, self.date_end)
        if 'data' not in result:
            raise ValueError(f"No se pudo obtener datos para {self.principal_symbol}: {result.get('error', 'unknown error')}")
        data = result['data']
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'], unit='s')
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
        return df.dropna().reset_index(drop=True)


    def get_market_features(self, row):
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

        score_path = f'output/{self.principal_symbol}/data_for_neuronal/best_score/score_{self.mercado}_{self.algorithm}.json'
        if os.path.exists(score_path):
            with open(score_path, 'r') as file:
                score_data = json.load(file)
            self.strategy_metrics = score_data.get("metrics", {})
            raw_threshold = self.strategy_metrics.get("best_threshold", self.close_threshold_floor)
            try:
                threshold_value = float(raw_threshold)
            except (TypeError, ValueError):
                threshold_value = self.close_threshold_floor
            self.close_threshold = max(
                threshold_value,
                self.close_threshold_floor,
            )
        else:
            self.strategy_metrics = {}
            self.close_threshold = self.close_threshold_floor
      
        
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

        for symbol in self.list_symbols:
            self.dict_nodos[symbol] = self.parsear_nodos(
                get_nodes_by_label(self.principal_symbol, symbol, self.mercado, self.algorithm) or []
            )

        self.nodos_close = self.parsear_nodos(
            get_nodes_by_label(self.principal_symbol, self.principal_symbol, mercado=None, label=self.other_algorithm) or []
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
        count_proces = int(self.general_config.get('use_proces', 40))//2
        if len(list_files) > count_proces:
            list_files = list_files[:count_proces]
        for symbol in self.list_symbols:
            data = get_historical_data(symbol, self.timeframe, self.date_start, self.date_end)['data']
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


    def resolve_entry_open_nodes(self, time_actual_np):
        matched_symbols = 0
        open_nodes = []

        for symbol in self.list_symbols:
            nodo_open_list = self.dict_nodos[symbol]

            for nodo in nodo_open_list:
                df_struct = self.indicators[f'{self.principal_symbol}_{symbol}_{nodo["file"].split("_")[0]}']
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
    
    
    def calculate_trade_pips(self, open_price_open, open_price_close, spread_open):
        if self.algorithm == 'UP':
            movement = open_price_close - open_price_open
        else:
            movement = open_price_open - open_price_close

        movement_pips = movement / self.pip_size
        spread_pips = spread_open * self.point_size / self.pip_size
        return movement_pips - spread_pips
    
    
    def test_iteration(self):

        path_data_red = f'output/{self.principal_symbol}/data_for_neuronal/data/data_{self.mercado}_{self.algorithm}.csv'
        input1_ids, input2_ids, hour_ids, X_extra, _ = load_data(path_data_red)

        model_path = f'output/{self.principal_symbol}/data_for_neuronal/model_trainer/model_{self.mercado}_{self.algorithm}.pt'

        nn = load_trained_model(
            model_path,
            input_dim_extra=X_extra.shape[1]
        )
        validate_embedding_vocab(nn, input1_ids, input2_ids, hour_ids)
        base_data = self.base_data[self.base_data['time'].between(self.date_start, self.date_end)].copy()

        is_open = False
        open_price_open = 0.0
        spread_open = 0.0
        entry_red_open = []
        cierre = 0
        model_close_streak = 0
        time_comienzo = None
        current_stop_loss = self.stop_loss
        current_take_profit = self.take_profit

        for row in base_data.itertuples():

            time_actual = row.time
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
                current_pips = self.calculate_trade_pips(open_price_open, open_price, spread_open)
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
                        df_struct = self.indicators[f'{self.principal_symbol}_{self.principal_symbol}_{nodo["file"].split("_")[0]}']
                        pos = df_struct["index_values"].searchsorted(time_actual_np)
                        if pos == 0:
                            continue

                        if self.cumple_condiciones_fast(df_struct, pos - 1, nodo["conditions"]):

                            nodo_close = self.maping_close.get(nodo["key"])
                            if nodo_close is None:
                                continue
                            for entry in entry_red_open:
                                prob = predict_from_inputs(nn, entry, nodo_close, hour, market_features)
                                if prob > self.close_threshold:
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
                    self.closed_trades_count += 1
                    self.closed_pips_total += trade_pips
                    self.results["time_open"].append(time_comienzo)
                    self.results["time_close"].append(time_actual)
                    self.results["pips"].append(trade_pips)
                    self.results["bars_held"].append(cierre)
                    self.results["close_reason"].append(close_reason)
                    is_open = False
                    model_close_streak = 0
                    if VERBOSE_TRADE_LOGS:
                        print(f"SUM PIPS:{time_comienzo} ---- {time_actual}: {trade_pips}", cierre)
                    elif self.closed_trades_count % TRADE_LOG_EVERY == 0:
                        print(
                            f"RESUMEN {self.principal_symbol} {self.mercado}_{self.algorithm} | "
                            f"ops={self.closed_trades_count} | pips={self.closed_pips_total:.2f}"
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
        crear_carpeta_si_no_existe(f'output/x_backtest_results')
        crear_carpeta_si_no_existe(f'output/x_backtest_results/{self.principal_symbol}')
        crear_carpeta_si_no_existe(f'output/x_backtest_results/{self.principal_symbol}/{self.mercado}_{self.algorithm}')
        output_dir = f'output/x_backtest_results/{self.principal_symbol}/{self.mercado}_{self.algorithm}'
        results = self.test_iteration()
        df_results = pd.DataFrame(results)
        df_results.to_csv(f'{output_dir}/results.csv', index=False)
        print(f"Resultados guardados en {output_dir}/results.csv")
        self.plot_results(results, output_dir)


      
    
if __name__ == "__main__":
    initialize_mt5()
    with open('config/general_config.json', 'r', encoding='utf8') as file:
        config = json.load(file)
    backtest_config_path = 'config/backtest_config.json'
    if not os.path.exists(backtest_config_path):
        with open(backtest_config_path, 'w', encoding='utf8') as file:
            json.dump(
                {
                    "backtest_config": {
                        "date_start": "2025-01-01",
                        "date_end": "2026-01-01"
                    }
                },
                file,
                indent=4,
                ensure_ascii=False,
            )

    with open(backtest_config_path, 'r', encoding='utf8') as file:
        backtest_config = json.load(file)['backtest_config']
    list_mercado = ['Asia', 'Europa', 'America'] 
    list_algorithms = ['UP', 'DOWN']
    list_principal_symbols = config['list_principal_symbols']
    date_start = backtest_config.get("date_start", "2025-01-01")
    date_end = backtest_config.get("date_end", "2026-01-01")
    for principal_symbol in list_principal_symbols:
        for mercado in list_mercado:
            for algorithm in list_algorithms:
                score_path = f'output/{principal_symbol}/data_for_neuronal/best_score/score_{mercado}_{algorithm}.json'
                if not os.path.exists(score_path):
                    print(
                        f"Saltando backtest para {principal_symbol} - {mercado} - {algorithm}: "
                        f"no existe {score_path}"
                    )
                    continue

                try:
                    with open(score_path, 'r', encoding='utf8') as file:
                        best_score_data = json.load(file)
                except FileNotFoundError:
                    print(
                        f"Saltando backtest para {principal_symbol} - {mercado} - {algorithm}: "
                        f"no existe {score_path}"
                    )
                    continue
                metrics = best_score_data.get("metrics", {})
                should_run = should_backtest_strategy(metrics)
                if not should_run:
                    print(
                        f"Saltando backtest para {principal_symbol} - {mercado} - {algorithm} "
                    )
                    continue
              
                try:
                    backtest = Backtest(principal_symbol, mercado, algorithm, date_start, date_end)
                    backtest.run()
                except ValueError as e:
                    print(f"Saltando backtest para {principal_symbol} - {mercado} - {algorithm}: {e}")
                    continue
    plot_all_backtests_results('output/x_backtest_results')
    