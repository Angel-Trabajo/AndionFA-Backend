import os, sys
import json
import shutil
import sqlite3
import subprocess
import psutil
import asyncio
from pathlib import Path


from fastapi import  APIRouter, HTTPException, BackgroundTasks, Query, WebSocket, WebSocketDisconnect
from typing import List

from src.routes import peticiones 
from ..models.indicators import ConfigRequest
from src.scripts.create_indicators import create_files
from src.scripts.node_builder import execute_node_builder
from src.scripts.crossing_builder_cpu import execute_crossing_builder
from dotenv import load_dotenv
from src.utils.common_functions import crear_carpeta_si_no_existe, get_previous_4_6
from src.neuronal.data_para_entrenar import execute_data_for_neuronal
from src.db import query

load_dotenv()

PYTHON_PATH = os.getenv("PYTHON_PATH")  
SCRIPT_PATH = os.getenv("SCRIPT_PATH")
SCRIPT_PATH_CROSS = os.getenv("SCRIPT_PATH_CROSS")
SCRIPT_PATH_CROSS_F = os.getenv("SCRIPT_PATH_CROSS_F")

PATH_GENERAL_CONFIG = 'config/general_config.json'
PATH_EXTRACTOR = 'config/extractor'


router = APIRouter()
process = None
process2 = None
connected_clients: List[WebSocket] = []


@router.get('/extractor-files')
async def extractor_file():
    peticiones.initialize_mt5()
    list_extractor_files = os.listdir(PATH_EXTRACTOR)
    return {"list_files": list_extractor_files}


@router.post('/extractor-files')
async def extractor_file(request: list[str]):
    with open(PATH_GENERAL_CONFIG, 'r', encoding='utf8') as file:
        data = json.load(file)
    data['indicators_files'] = request
    with open(PATH_GENERAL_CONFIG, 'w', encoding='utf8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
    return {
        "status": "ok",}


@router.get('/list-symbol')
async def list_simbol():
    peticiones.initialize_mt5()
    active_symbols = peticiones.get_active_symbols()
    return {"symbols": active_symbols}


@router.get("/general-config") 
async def getnode_config():
    peticiones.initialize_mt5()
    with open(PATH_GENERAL_CONFIG, 'r', encoding='utf8') as file:
        data = json.load(file)
    return {"data":data}   


@router.post("/general-config") 
async def postnode_config(request: ConfigRequest):
    with open(PATH_GENERAL_CONFIG, 'r', encoding='utf8') as file:
        config = json.load(file)
    
    config["list_principal_symbols"] = request.symbols
    config["timeframe"] = request.timeFrame
    config["dateStart"] = request.dateStart
    config["dateEnd"] = request.dateEnd
    config["SimilarityMax"] = request.SimilarityMax
    config["NTotal"] = request.NTotal
    config["MinOperationsIS"] = request.MinOperationsIS
    config["MinOperationsOS"] = request.MinOperationsOS
    config["NumMaxOperations"] = request.NumMaxOperations
    config["min_operaciones"] = request.MinOperations
    config["MinSuccessRate"] = request.MinSuccessRate
    config["MaxSuccessRate"] = request.MaxSuccessRate
    config["ProgressiveVariation"] = request.ProgressiveVariation
    
    with open(PATH_GENERAL_CONFIG, 'w', encoding='utf8') as file:
        json.dump(config, file, indent=4, ensure_ascii=False)
    
    
    return {
        "status": "ok",
        "message": "Configuracion completado",
    } 


@router.post("/execute-algorithm")
def execute_algorithm():
    #query.eliminar_nodos_y_registros()
    with open(PATH_GENERAL_CONFIG, 'r', encoding='utf8') as file:
        config = json.load(file)
    list_mercado = ['Asia', 'Europa', 'America'] 
    list_principal_symbols = config['list_principal_symbols']
    timeframe = config['timeframe']
    date_start_os = config['dateStart']
    date_end_os = config['dateEnd']
    indicadors_files = config['indicators_files']
    date_start_is, date_end_is = get_previous_4_6(date_start_os, date_end_os)
    
    for symbol in list_principal_symbols:
        # crear_carpeta_si_no_existe(f'config/divisas/{symbol}')
        # crear_carpeta_si_no_existe(f'output/{symbol}')
        # create_files(symbol, timeframe, date_start_os, date_end_os, indicadors_files, 'extrac_os')
        # create_files(symbol, timeframe, date_start_is, date_end_is, indicadors_files, 'extrac')
        # execute_node_builder(symbol, list_mercado)
        # execute_crossing_builder(symbol, list_mercado)
        execute_data_for_neuronal(symbol, list_mercado, list_algorithms = None, dict_pips_best= {})  
        pass

    return {
        "status": "ok",
        "message": "Procesamiento completado",
    }
    

# def node_builder(symbol, timeframe):
#     create_trees(symbol, timeframe)
#     print("Node Builde completado")
 

# @router.post('/execute-node-builder')
# async def execute_node_builder(request: NodeIndRequest, background_tasks: BackgroundTasks):
#     symbol = request.symbol
#     timeframe = request.timeframe
    
#     eliminar_archivo('output/is_os/os.csv')
#     limpiar_carpeta('output/extrac_os')
#     limpiar_carpeta('output/nodos')
#     limpiar_carpeta('output/data_arff')
    
#     background_tasks.add_task(node_builder, symbol, timeframe)
#     return {"message": "Tarea lanzada en segundo plano"}



# @router.get("/nodos")
# async def get_nodos(symbol: str = Query(...)):
#     db_path = f"output/db/{symbol}.db"

#     if Path(db_path).exists() and tabla_nodes_existe(db_path):
#         result = get_nodes(symbol)
#         return {"data": result}
#     else:
#         return {"data": []}
        
        
        
# @router.get('/symbol-extrac') 
# async def get_symbol_extrac():
#     symbol_extrac = os.listdir('output/extrac')[0].split('_')[1].strip()      
#     if symbol_extrac:
#         return {"data": symbol_extrac}
#     else:
#         return{"data": 'NoSymbol'}
    
    
# @router.websocket("/ws/test-status")
# async def websocket_test_status(websocket: WebSocket):
#     origin = websocket.headers.get("origin")
#     print(f"🛰️ Origen recibido en WebSocket: {origin!r}")
#     print(f"Headers WebSocket: {websocket.headers}")

#     allowed_origins = [
#         o.strip().rstrip('/')
#         for o in os.getenv("FRONTEND_VITE", "").split(",")
#         if o.strip()
#     ]
#     print(f"🎯 Orígenes permitidos: {allowed_origins}")

#     if origin is not None and origin.lower().strip().rstrip('/') not in [o.lower() for o in allowed_origins]:
#         print(f"❌ Origen '{origin}' no permitido")
#         await websocket.close(code=1008)  # Policy violation
#         return

#     await websocket.accept()
#     print(f"✅ WebSocket aceptado desde {origin}")
#     connected_clients.append(websocket)

#     try:
#         while True:
#             if process and process.poll() is not None:
#                 await websocket.send_json({"status": "finished"})
#                 break
#             await asyncio.sleep(1)
#     except WebSocketDisconnect:
#         print("🔌 Cliente WebSocket desconectado")
#     finally:
#         if websocket in connected_clients:
#             connected_clients.remove(websocket)

    
    
# @router.post('/start-test')
# async def make_test(request: TestRequest):
#     global process

#     symbol = request.symbol
#     list_id = request.list_id

#     with open('config/config_test/config_test.json', 'w', encoding='utf8') as file:
#         json.dump({"symbol": symbol, "list_id": list_id}, file, indent=4, ensure_ascii=False)

#     if process and process.poll() is None:
#         return {"message": "Ya hay un proceso en ejecución"}

#     loop = asyncio.get_running_loop()  # ✔️ este sí funciona en FastAPI

#     def run_script():
#         global process
#         process = subprocess.Popen([PYTHON_PATH, SCRIPT_PATH])
#         process.wait()
#         asyncio.run_coroutine_threadsafe(notify_ws_clients(), loop)  # ✔️ correr en loop original

#     asyncio.create_task(asyncio.to_thread(run_script))

#     return {"message": "Proceso iniciado"}



# async def notify_ws_clients():
#     for client in connected_clients:
#         try:
#             await client.send_json({"status": "finished"})
#         except:
#             pass   
    
    
# @router.post("/stop-test")
# def stop_test():
#     global process
#     if process and process.poll() is None:
#         process.terminate()
#         process.wait()  # Espera a que termine
#         return {"message": "Proceso detenido"}
#     else:
#         raise HTTPException(status_code=404, detail="No hay proceso corriendo")    



# @router.get("/get-crossing-config")
# async def get_crossing_config():
#     with open('config/config_crossing/config_crossing.json', 'r', encoding='utf8') as file:
#         crossing_config = json.load(file)

#     if crossing_config:
#         return crossing_config
#     else:
#         return {"data": 'NoCrossingConfig'}



# @router.post("/set-crossing-config")
# async def set_crossing_config(request: ConfigCrossingRequest):
#     print(request.list_symbols)
#     data = {
#         "n_totales": request.n_totales,
#         "min_operaciones": request.min_operaciones,
#         "principal_symbol": request.principal_symbol,
#         "timeframe": request.timeframe,
#         "por_direccion": request.por_direccion,
#     }
#     with open('config/config_crossing/config_crossing.json', 'w', encoding='utf8') as file:
#         json.dump(data, file, indent=4, ensure_ascii=False)

#     return {"message": "Configuración de cruce actualizada"}


# @router.get('/get-principal-symbol')
# async def get_principal_symbol():
#     with open('config/config_crossing/config_crossing.json', 'r', encoding='utf8') as file:
#         config = json.load(file)
#     symbol = config['principal_symbol']
    
#     return {'symbol': symbol}


# @router.get("/crossing-dbs")
# async def crossing_dbs():
#     dirs = os.listdir('output/db')
#     mydir =''
#     for dir in dirs:
#         if dir.split('_')[-1] == 'dbs':
#             mydir=f'output/db/{dir}'
#             break
#     try:
#         mydirs = os.listdir(mydir)
#         return {"dir":mydirs}
#     except:
#         return {"dir":['NOBD.db']}
    
    
# @router.get("/nodos-db-crossing/{symbol}")
# async def get_nodos(symbol: str):
#     symbol = symbol.split('.')[0]
#     dirs = os.listdir('output/db')
#     for dir in dirs:
#         if dir.split('_')[-1] == 'dbs':
#             symbol = f'{dir}/{symbol}'
#             break
    
#     db_path = f"output/db/{symbol}.db"

#     if Path(db_path).exists() and tabla_nodes_existe(db_path):
#         result = get_nodes(symbol)
#         return {"data": result}
#     else:
#         return {"data": []}  
    
    
# @router.post('/execute-crossing')
# async def execute_crossing():
#     global process2
#     if process2 and process2.poll() is None:
#         return {"status": "running", "message": "El proceso ya está en ejecución"}
#     with open('config/list_UP.json', 'w', encoding='utf-8') as f:
#                 json.dump({"list": []}, f, indent=4)
#     with open('config/list_DOWN.json', 'w', encoding='utf-8') as f:
#                 json.dump({"list": []}, f, indent=4)

#     list_dir = os.listdir('output')
#     for dir in list_dir:
#         info = dir.split('_')[0]
#         if info == 'crossing':
#             eliminar_ruta(f'output/{dir}')
#     list_dir = os.listdir('output/db')
#     for dir in list_dir:
#         info = dir.split('_')[-1]
#         if info == 'dbs':
#             eliminar_ruta(f'output/db/{dir}')
#             break 
#     extract_data_crossing()
#     select_symbols_correl()   
     
#     script_path = Path(__file__).resolve().parent.parent / "utils" / "crossing_builder_cpu.py"
#     process2 = subprocess.Popen(
#         [sys.executable, str(script_path)],
#         stdout=None,   # hereda consola
#         stderr=None,
#     )
#     return {"status": "ok", "pid": process2.pid, "message": "Proceso iniciado"}      


# @router.post("/stop-crossing")
# async def stop_crossing():
#     global process2
#     if process2 and process2.poll() is None:
#         parent = psutil.Process(process2.pid)
#         for child in parent.children(recursive=True):
#             child.kill()
#         parent.kill()
#         process2.wait()
#         return {"message": "Proceso detenido"}
#     else:
#         return {"message": "No hay proceso corriendo"}


# @router.post('/start-test-cross')
# async def make_test(request: TestRequestCros):
#     global process
#     prin_symbol = request.prin_symbol
#     symbol = request.symbol
#     list_id = request.list_id
#     future = request.future

    
#     with open('config/config_test/config_test.json', 'w', encoding='utf8') as file:
#         json.dump({"prin_symbol": prin_symbol, "symbol": symbol, "list_id": list_id}, file, indent=4, ensure_ascii=False)

#     if process and process.poll() is None:
#         return {"message": "Ya hay un proceso en ejecución"}

#     loop = asyncio.get_running_loop()  # ✔️ este sí funciona en FastAPI

#     def run_script():
#         global process
#         if future == 'yes':
#             process = subprocess.Popen([PYTHON_PATH, SCRIPT_PATH_CROSS_F])
#         else:
#             process = subprocess.Popen([PYTHON_PATH, SCRIPT_PATH_CROSS])
#         process.wait()
#         asyncio.run_coroutine_threadsafe(notify_ws_clients(), loop)  # ✔️ correr en loop original

#     asyncio.create_task(asyncio.to_thread(run_script))

#     return {"message": "Proceso iniciado"}


# @router.get('/get-state-crossing')
# async def get_state_crossing():
#     with open('config/state.json', 'r', encoding='utf-8') as f:
#         state = json.load(f)
#     return state


# @router.post('/execute-neuronal-red')
# async def execute_neuronal_red():
    base = Path(__file__).resolve().parent.parent / "neuronal"

    p1 = subprocess.Popen([sys.executable, str(base / "data_para_entrenar.py")])
    p1.wait()  # ⬅️ espera a que termine

    p2 = subprocess.Popen([sys.executable, str(base / "entrenar.py")])

    return {
        "status": "ok",
        "pid": p2.pid,
        "message": "Proceso iniciado"
    }