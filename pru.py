import os
import json
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime
from src.routes.peticiones import initialize_mt5, get_active_symbols, get_timeframes, get_historical_data

perritos = ["Firulais", "Rex", "Luna", "Max", "Bella", "Charlie", "Molly", "Buddy", "Daisy", "Rocky", "Lucy", "Cooper", "Lola", "Duke", "Sadie"]
rang = min(2, 5)
print(20%15)


# initialize_mt5()
# print(get_active_symbols())
# print(get_timeframes())
# rates = get_historical_data('EURUSD', 16385, '2019-01-01', '2023-12-31')['data']


# positions = mt5.history_deals_get(datetime(2026,1,1),datetime.now())
# print()
# print(positions)

# base_path_results = 'output/y_backtest_results'
# list_symbols = os.listdir(base_path_results)
# if 'paths_win.json' in list_symbols:
#     list_symbols.remove('paths_win.json')
# list_algorithms = os.listdir(f'{base_path_results}/{list_symbols[0]}')
# list_paths_win = []
# for symbol in list_symbols:
#     for algorithm in list_algorithms:
#         path = f'{base_path_results}/{symbol}/{algorithm}/results.csv'
#         df = pd.read_csv(path)
#         sumpis = df['pips'].sum()
#         if sumpis > 0:  
#             list_paths_win.append(path)
# data = {
#     "paths": list_paths_win
# }
# with open(f'{base_path_results}/paths_win.json', 'w') as file:
#     json.dump(data, file, indent=4)


# paths_win = 'output/y_backtest_results/paths_win.json'
# with open(paths_win, 'r') as file:
#     paths_win_data = json.load(file)
# list_paths_win = paths_win_data.get("paths", [])
# for path in list_paths_win:
#     path = path.replace('/y_backtest_results/','/') 
#     list_palabras = path.split('/')
#     list_palabras[2:2] = ['data_for_neuronal', 'best_score']
#     list_palabras.remove('results.csv')
#     name = list_palabras[-1]
#     name = f'score_{name}.json'
#     list_palabras[-1] = name
#     path = '/'.join(list_palabras)
#     with open(path, 'r') as file:
#         data = json.load(file)
#     metrics = data.get("metrics", {})
    
#     list_pips_monthly = list(metrics.get("temporal_stats", {}).get("monthly_pips",{}).values())
#     def score(fila):
#         fila = np.array(fila)
        
#         suma = np.sum(fila)
#         volatilidad = np.std(fila)
#         maximo = np.max(fila)
#         minimo = np.min(fila)
        
#         return (
#             0.4 * suma
#             - 0.3 * volatilidad
#             + 0.2 * maximo
#             + 0.1 * minimo
#         )
        
#     def probabilidad(score):
#         import math
#         return 1 / (1 + math.exp(-score / 100))
#     s = score(list_pips_monthly)
#     prob = probabilidad(s) 
#     print(prob)


