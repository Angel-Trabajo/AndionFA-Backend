import requests


path ="http://192.168.1.4:8000/"

def initialize_mt5():
    url = f"{path}metatrader/inicializar"
    response = requests.post(url)
    if response.status_code == 200:
        datos = response.json()  # Convertir la respuesta en JSON
        print(datos)
    else:
        print("Error:", response.status_code)


def get_active_symbols():
    url = f"{path}metatrader/get_symbols"
    response = requests.get(url)
    if response.status_code == 200:
        datos = response.json()  # Convertir la respuesta en JSON
        return datos.get('symbols')
    else:
        print("Error:", response.status_code)
       
        
def get_timeframes():
    url = f"{path}metatrader/timeframes"
    response = requests.get(url)
    if response.status_code == 200:
        datos = response.json()  # Convertir la respuesta en JSON
        return datos['timeframes']
    else:
        print("Error:", response.status_code)
        


def get_historical_data(symbol, timeframe, start, end):
    url = f"{path}metatrader/data_by_date/{symbol}/{timeframe}/{start}/{end}"
    response = requests.get(url)
    if response.status_code == 200:
        datos = response.json()  # Convertir la respuesta en JSON
        return datos.get('data')
    else:
        print("Error:", response.status_code)


def get_data_by_days(symbol, timeframe, start, count_days):
    url = f"{path}metatrader/data_by_days/{symbol}/{timeframe}/{start}/{count_days}"
    response = requests.get(url)
    if response.status_code == 200:
        datos = response.json()  # Convertir la respuesta en JSON
        return datos.get('data')
    else:
        print("Error:", response.status_code)

