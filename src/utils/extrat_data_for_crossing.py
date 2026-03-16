import os
import json
import re
import sys
import pandas as pd
import time as tim
import numpy as np
import shutil
from pathlib import Path

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from src.routes import peticiones
from src.utils.common_functions import get_previous_4_6, crear_carpeta_si_no_existe

list_symbol_bruto = peticiones.get_active_symbols()
_mapping_time = peticiones.get_timeframes()
list_malas = ["EURMXN","EURNOK","EURSEK"]  # Ejemplo de símbolo a eliminar, se puede ajustar según necesidades
for mala in list_malas:
    if mala in list_symbol_bruto:
        list_symbol_bruto.remove(mala)

def _buscar_data(folder, sym, str_start, end, config, principal_symbol):  
    # Obtener datos UNA sola vez
    df = None
    
    try:
        timeframe_mapped = _mapping_time.get(config['timeframe'])
        rates = peticiones.get_historical_data(sym, timeframe_mapped, str_start, end)
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        
        # Guardar CSV inmediatamente
        if folder == 'extrac':
            df.to_csv(f'output/{principal_symbol}/crossing/{sym}/is_os/is.csv', index=False)
        else:
            df.to_csv(f'output/{principal_symbol}/crossing/{sym}/is_os/os.csv', index=False)
        
        print(sym, "rows:", folder, len(df))
    except Exception as e:
        print(f"Error MT5 {sym}: {e}")
        tim.sleep(3)


def extract_data_crossing(principal_symbol):
    """Extrae indicadores optimizado - SIN cambios estructurales grandes"""
    with open('config/general_config.json', encoding='utf-8') as f:
        config= json.load(f)
    if principal_symbol in list_symbol_bruto:
        list_symbol_bruto.remove(principal_symbol)
        
    base_path = f'output/{principal_symbol}/crossing'
    crear_carpeta_si_no_existe(base_path)
    
    for sym in list_symbol_bruto:
        base_path1 = f'{base_path}/{sym}'
        crear_carpeta_si_no_existe(base_path1)
        crear_carpeta_si_no_existe(f'{base_path1}/is_os')
        crear_carpeta_si_no_existe(f'{base_path1}/extrac')
        crear_carpeta_si_no_existe(f'{base_path1}/extrac_os')
        str_start, end = get_previous_4_6(config['dateStart'], config['dateEnd'])
        
        
        _buscar_data('extrac', sym, str_start, end, config, principal_symbol)
        _buscar_data('extrac_os', sym, config['dateStart'], config['dateEnd'], config, principal_symbol)
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

      
def select_symbols_correl(principal_symbol):
    df_os_principal = _create_label(pd.read_csv(f'output/{principal_symbol}/is_os/os.csv'))
    df_is_principal = _create_label(pd.read_csv(f'output/{principal_symbol}/is_os/is.csv'))
    
    list_symbol = []
    list_symbol_inversos = []
    list_symbol_delete = []
    dict_symbol_correl = {}
    
    
    for i ,symbol in enumerate(list_symbol_bruto):
        
        paht_os = Path(f'output/{principal_symbol}/crossing/{symbol}/is_os/os.csv')
        paht_is = Path(f'output/{principal_symbol}/crossing/{symbol}/is_os/is.csv')
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
    
    for symbol in list_symbol_delete:
        ruta = f"output/{principal_symbol}/crossing/{symbol}"
        if os.path.exists(ruta):
            shutil.rmtree(ruta)
            print(f'{symbol} eliminada data')
    
    
if __name__ == '__main__':
    select_symbols_correl()
    

    