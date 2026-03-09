import os
import sys
import json
import operator
import ast
import pandas as pd 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.routes.peticiones import get_historical_data, get_timeframes
from src.utils.indicadores_for_principal_script import generate_files
from src.db.query import get_nodes_by_label


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
             
        self.list_symbols = self.config_symbol['list_symbol']
        self.list_symbols.insert(0, self.principal_symbol)
        self.timeframe = get_timeframes().get(self.general_config['timeframe'])
        self.base_data = self.prepare_base_data()
        
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
                self.indicators[f'{self.principal_symbol}_{symbol}_{file}'] = indicator
        print(self.indicators['EURUSD_EURUSD_Ext-151614.csv'])      
    
    
    
    
    def run(self):
        pass
    
if __name__ == "__main__":
    # Ejemplo de uso
    principal_symbols = "EURUSD"
    mercado = "Europa"
    algorithm = "UP"
    date_start = "2023-01-01"
    date_end = "2025-01-01"
    backtest = Backtest(principal_symbols, mercado, algorithm, date_start, date_end)
    backtest.run()