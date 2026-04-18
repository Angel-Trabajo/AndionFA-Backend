import os
import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

list_symbol_bruto = [
    "AUDCHF",
    "AUDCAD",
    "AUDUSD",
    "CADCHF",
    "CADJPY",
    "CHFJPY",
    "EURAUD",
    "EURCAD",
    "EURCHF",
    "EURGBP",
    "EURJPY",
    "EURNZD",
    "EURUSD",
    "GBPAUD",
    "GBPCAD",
    "GBPCHF",
    "GBPJPY",
    "GBPNZD",
    "GBPUSD",
    "NZDCAD",
    "NZDCHF",
    "NZDJPY",
    "NZDUSD",
    "USDCAD",
    "USDCHF",
    "USDJPY",
    "USDSEK",
    "USDTRY"
]



def _create_label(df):
    df["label"] = np.where(df["open"] > df["close"], 1, -1)
    return df   

       
def _pearson_binario_simple(df1, df2):
    merged = df1.merge(df2, on="time", how="inner")
    x = merged["label_x"].to_numpy()
    y = merged["label_y"].to_numpy()
    return round(np.corrcoef(x, y)[0, 1], 2)

      
def select_symbols_correl(principal_symbol):
    symbols = list_symbol_bruto.copy()
    if principal_symbol in symbols:
        symbols.remove(principal_symbol)
    df_os_principal = _create_label(pd.read_csv(f'output/symbol_data/{principal_symbol}/is_os/os.csv'))
    df_is_principal = _create_label(pd.read_csv(f'output/symbol_data/{principal_symbol}/is_os/is.csv'))
    
    list_symbol = []
    list_symbol_inversos = []
    list_symbol_delete = []
    dict_symbol_correl = {}
    
    
    for i ,symbol in enumerate(symbols):
        
        paht_os = Path(f'output/symbol_data/{symbol}/is_os/os.csv')
        paht_is = Path(f'output/symbol_data/{symbol}/is_os/is.csv')
        if not paht_os.exists() or not paht_is.exists():
            list_symbol_delete.append(symbol)
            continue
        df_os = _create_label(pd.read_csv(paht_os))
        correla_os = _pearson_binario_simple(df_os_principal, df_os)
        if abs(correla_os) < 0.2:
            list_symbol_delete.append(symbol)
            continue
         
        df_is = _create_label(pd.read_csv(paht_is))
        correla_is =_pearson_binario_simple(df_is_principal, df_is)
        if abs(correla_is) < 0.2:
            list_symbol_delete.append(symbol)
            continue
        if abs(correla_os - correla_is) > 0.1:
            list_symbol_delete.append(symbol)
            continue
        
        list_symbol.append(symbol)
        dict_symbol_correl[symbol]= round(((abs(correla_is) + abs(correla_os))/2), 3)
        
        if correla_os < 0 :
            list_symbol_inversos.append(symbol) 
            print(f'{symbol}-> agregado a la lista de symbols inversos con correlación: os {correla_os}  is {correla_is}')
        else:
            print(f'{symbol}-> agregado a la lista de symbols con correlación: os {correla_os}  is {correla_is}')
    
    new_list_symbol = []

    for symbol in list_symbol:
        
        if len(new_list_symbol) == 0:
            new_list_symbol.append(symbol)
            continue

        inserted = False

        for i, existing_symbol in enumerate(new_list_symbol):
            
            if dict_symbol_correl[symbol] > dict_symbol_correl[existing_symbol]:
                new_list_symbol.insert(i, symbol)
                inserted = True
                break

        if not inserted:
            new_list_symbol.append(symbol)
        
    max_symbols = 6
    list_symbol = new_list_symbol[:max_symbols]
    list_symbol_inversos = [symbol for symbol in list_symbol_inversos if symbol in list_symbol]
    dict_symbol_correl = {symbol: dict_symbol_correl[symbol] for symbol in list_symbol}

    data = {
        "list_symbol": list_symbol,
        "list_symbol_inversos": list_symbol_inversos,
        "dict_symbol_correl": dict_symbol_correl,
        "list_UP_Asia": [],
        "list_UP_Europa": [],
        "list_UP_America": [],
        "list_DOWN_Asia": [],
        "list_DOWN_Europa": [],
        "list_DOWN_America": []
    }       
    with open(f'config/divisas/{principal_symbol}/config_{principal_symbol}.json', 'w') as file:
        json.dump(data, file, indent=4)           
    
if __name__ == '__main__':
    select_symbols_correl()
    

    