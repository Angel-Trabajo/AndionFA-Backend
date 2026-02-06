import os
import json
import re
import sys
import pandas as pd
import time as tim
import numpy as np
import shutil

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from src.routes import peticiones


_mapping_time = peticiones.get_timeframes()

with open('config/config_crossing/config_crossing.json', 'r') as file:
    config = json.load(file)

with open('config/config_node/config_node.json', encoding='utf-8') as f:
    config_node = json.load(f)

principal_symbol = config['principal_symbol']
timeframe = config['timeframe'] 
peticiones.initialize_mt5()
list_symbol_bruto = peticiones.get_active_symbols()
try:
    list_symbol_bruto.remove(principal_symbol)
except:
    print(f'El symbol {principal_symbol} no está en la lista')



def _buscar_data(folder, sym, str_start, end):  
    # Obtener datos UNA sola vez
    df = None
    
    try:
        timeframe_mapped = _mapping_time.get(timeframe)
        rates = peticiones.get_historical_data(sym, timeframe_mapped, str_start, end)
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        
        # Guardar CSV inmediatamente
        if folder == 'extrac':
            df.to_csv(f'output/crossing_{principal_symbol}/{sym}/is_os/is.csv', index=False)
        else:
            df.to_csv(f'output/crossing_{principal_symbol}/{sym}/is_os/os.csv', index=False)
        
        print(sym, "rows:", folder, len(df))
    except Exception as e:
        print(f"Error MT5 {sym}: {e}")
        tim.sleep(3)


def extract_data_crossing():
    """Extrae indicadores optimizado - SIN cambios estructurales grandes"""
    indicators_files = os.listdir(f'output/extrac')
    texto = indicators_files[0]
    fechas = re.findall(r'\d{8}', texto)
    str_start = f"{fechas[0][:4]}-{fechas[0][4:6]}-{fechas[0][6:]}"
    end = f"{fechas[1][:4]}-{fechas[1][4:6]}-{fechas[1][6:]}"
    
    for sym in list_symbol_bruto:
        # Crear directorios UNA sola vez
        base_path = f'output/crossing_{principal_symbol}/{sym}'
        os.makedirs(f'{base_path}/is_os', exist_ok=True)
        os.makedirs(f'{base_path}/extrac', exist_ok=True)
        os.makedirs(f'{base_path}/extrac_os', exist_ok=True)  
        os.makedirs(f'{base_path}/data_arff', exist_ok=True) 
        os.makedirs(f'output/db/crossing_{principal_symbol}_dbs', exist_ok=True)
        
       
        _buscar_data('extrac', sym, str_start, end)
        _buscar_data('extrac_os', sym, config_node['dateStart'], config_node['dateEnd'])
        print('-----------------------------------')


#----------------------------------------------------------------------------------------------------
def _create_label(df):
    df["label"] = np.where(df["open"] > df["close"], 1, -1)
    return df   

       
def _pearson_binario_simple(df1, df2):
    merged = df1.merge(df2, on="time", how="inner")
    x = merged["label_x"].to_numpy()
    y = merged["label_y"].to_numpy()
    return round(np.corrcoef(x, y)[0, 1], 2)

      
def select_symbols_correl():
    df_os_principal = _create_label(pd.read_csv('output/is_os/os.csv'))
    df_is_principal = _create_label(pd.read_csv('output/is_os/is.csv'))
    
    list_symbol = []
    list_symbol_inversos = []
    list_symbol_delete = []
    dict_symbol_correl = {}
    
    
    for i ,symbol in enumerate(list_symbol_bruto):

        df_os = _create_label(pd.read_csv(f'output/crossing_{principal_symbol}/{symbol}/is_os/os.csv'))
        correla_os = _pearson_binario_simple(df_os_principal, df_os)
        if abs(correla_os) < 0.2:
            list_symbol_delete.append(symbol)
            continue
         
        df_is = _create_label(pd.read_csv(f'output/crossing_{principal_symbol}/{symbol}/is_os/is.csv'))
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
        
           
    config['dict_symbol_correl'] = dict_symbol_correl
    config['list_symbol_inversos'] = list_symbol_inversos
    config['list_symbol'] = new_list_symbol
    with open('config/config_crossing/config_crossing.json', 'w') as file:
        json.dump(config, file, indent=4)           
    
    for symbol in list_symbol_delete:
        ruta = f"output/crossing_{principal_symbol}/{symbol}"
        if os.path.exists(ruta):
            shutil.rmtree(ruta)
            print(f'{symbol} eliminada data')
    
    
if __name__ == '__main__':
    select_symbols_correl()
    

    