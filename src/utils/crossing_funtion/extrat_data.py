import os
import json
import re
import sys
import pandas as pd
import time as tim
import numpy as np

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
#list_symbol = config['list_symbols']
list_symbol_bruto = config['list_symbols_bruto']



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
    peticiones.initialize_mt5()
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

      
def select_symbols():

    df_os_principal = _create_label(pd.read_csv('output/is_os/os.csv'))
    df_is_principal = _create_label(pd.read_csv('output/is_os/is.csv'))
    
    list_symbol = []
    list_symbol_inversos = []
    list_symbol_delete = []
    
    
    for i ,symbol in enumerate(list_symbol_bruto):

        df_os = _create_label(pd.read_csv(f'output/crossing_{principal_symbol}/{symbol}/is_os/os.csv'))
        df_is = _create_label(pd.read_csv(f'output/crossing_{principal_symbol}/{symbol}/is_os/is.csv'))
        
        print(i+1, '---->', _pearson_binario_simple(df_os_principal, df_os),':', _pearson_binario_simple(df_is_principal, df_is))
        
        
            
    
if __name__ == '__main__':
    select_symbols()

    