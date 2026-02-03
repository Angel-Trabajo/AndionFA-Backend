# save as: create_erff_fast.py
import pandas as pd
import os
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor


 
def process_file(symbol, principal_symbol):
    input_dir = Path(f'output/crossing_{principal_symbol}/{symbol}/extrac')
    output_dir = Path(f'output/crossing_{principal_symbol}/{symbol}/data_arff')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for file_name in os.listdir(input_dir):
        if not file_name.endswith('.csv'):
            continue
        
        file_start = time.time()
        input_path = input_dir / file_name
        output_path = output_dir / f"{file_name.replace('.csv', '')}.arff"
        
        print(f"  {file_name[:30]:<30}...", end="", flush=True)
        
        try:
            # 1. Leer CSV rápido
            df = pd.read_csv(input_path, low_memory=False)
            
            # 2. Eliminar 'time' y reordenar 'label'
            if 'time' in df.columns:
                df = df.drop('time', axis=1)
            
            if 'label' in df.columns:
                cols = [c for c in df.columns if c != 'label'] + ['label']
                df = df[cols]
            
            # 3. Preparar atributos (IGUAL que tu original)
            attributes = []
            for col in df.columns:
                if df[col].dtype == object:
                    unique_vals = sorted(df[col].dropna().unique().tolist())
                    attributes.append((col, unique_vals))
                else:
                    attributes.append((col, 'REAL'))
            
            # 4. Escribir ARFF (MÁS RÁPIDO que arff.dump)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("@relation mi_dataset\n\n")
                
                for attr_name, attr_type in attributes:
                    if isinstance(attr_type, list):
                        f.write(f"@attribute {attr_name} {{{','.join(str(v) for v in attr_type)}}}\n")
                    else:
                        f.write(f"@attribute {attr_name} {attr_type}\n")
                
                f.write("\n@data\n")
                df.to_csv(f, header=False, index=False, float_format='%g')
            
            file_time = time.time() - file_start
            print(f"✓ {len(df)} filas en {file_time:.1f}s")
            
        except Exception as e:
            print(f"✗ Error: {e}")


def create_erff(list_symbol, principal_symbol):
    """
    Versión MÁS RÁPIDA que mantiene EXACTAMENTE el mismo formato ARFF.
    """
    start_total = time.time()
    MAX_PROCESOS = len(list_symbol)
    with ProcessPoolExecutor(max_workers=MAX_PROCESOS) as executor:
        for symbol in list_symbol:
            print(f"📂 Procesando {symbol}...")
            executor.submit(process_file, symbol, principal_symbol)
       
        

    total_time = time.time() - start_total
    print(f"\n✅ COMPLETADO en {total_time/60:.1f} minutos")
    return total_time