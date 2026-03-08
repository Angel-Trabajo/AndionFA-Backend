from src.db.query import get_nodes

mercados = ["Asia", "Europa", "America"]
algorithms = ["UP", "DOWN"]
symbol = "EURUSD"
task =  [(symbol, mercado, algorithm) for mercado in mercados for algorithm in algorithms]
print(task)