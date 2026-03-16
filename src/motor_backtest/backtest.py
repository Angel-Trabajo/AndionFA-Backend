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
from src.neuronal.entrenar import load_trained_model, predict_from_inputs, load_data
from src.utils.common_functions import hora_en_mercado, crear_carpeta_si_no_existe


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
        self.timeframe = get_timeframes().get(self.general_config['timeframe'])
        self.base_data = self.prepare_base_data()
        self.pip_size, self.point_size = get_pip_and_point_size(
            self.principal_symbol,
            self.base_data['open'] if 'open' in self.base_data.columns else self.base_data['close']
        )
        self.results = {
            "time_open": [],
            "time_close": [],
            "pips": []
        }
        
        self.setup_operators_and_mappings()
        self.load_nodes()
        self.calculate_indicators()
        
        
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
            get_nodes_by_label(self.principal_symbol, self.principal_symbol, self.mercado, self.other_algorithm)
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
    
    
    def test_iteration(self):

        path_data_red = f'output/{self.principal_symbol}/data_for_neuronal/data/data_{self.mercado}_{self.algorithm}.csv'
        X, Y = load_data(path_data_red)

        nn = load_trained_model(
            f'output/{self.principal_symbol}/data_for_neuronal/model_trainer/model_{self.mercado}_{self.algorithm}.json',
            input_dim=X.shape[1]
        )
        base_data = self.base_data[self.base_data['time'].between(self.date_start, self.date_end)].copy()

        is_open = False
        open_price_open = 0.0
        spread_open = 0.0
        entry_red_open = ''
        cierre = 0
        time_comienzo = None

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
                for nodo in self.nodos_close:
                    df_struct = self.indicators[f'{self.principal_symbol}_{self.principal_symbol}_{nodo["file"].split("_")[0]}']
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
                        self.results["time_open"].append(time_comienzo)
                        self.results["time_close"].append(time_actual)
                        self.results["pips"].append(trade_pips)
                        is_open = False
                        print(f"SUM PIPS:{time_comienzo} ---- {time_actual}: {trade_pips}", cierre)
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

                        df_struct = self.indicators[f'{self.principal_symbol}_{symbol}_{nodo["file"].split("_")[0]}']
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
    # Ejemplo de uso
    principal_symbols = "AUDCAD"
    mercado = "Asia"
    algorithm = "DOWN"
    date_start = "2025-01-01"
    date_end = "2026-03-01"
    backtest = Backtest(principal_symbols, mercado, algorithm, date_start, date_end)
    backtest.run()