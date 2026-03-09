from src.db.query import get_nodes

nodes = get_nodes(principal_symbol="EURAUD", symbol_cruce="EURAUD", mercado="Asia", label="UP")
for node in nodes:
    print(node)
print(len(nodes))