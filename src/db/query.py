import sqlite3
import os
from collections import Counter

import threading

_thread_local = threading.local()

def get_connection(db_path):

    if not hasattr(_thread_local, "connections"):
        _thread_local.connections = {}

    if db_path not in _thread_local.connections:
        conn = sqlite3.connect(
            db_path,
            timeout=30,
            isolation_level=None,
            check_same_thread=False
        )
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        _thread_local.connections[db_path] = conn

    return _thread_local.connections[db_path]

def insertar_nodo_con_registros(
    name, label, file_in_db, conditions,
    correct_percentage, successful_operations, total_operations,
    correct_percentage_os, successful_operations_os, total_operations_os,
    fechas=None, veneficios=None, fechas_os=None, veneficios_os=None
):

    db_path = f'output/db/{name}.db'
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute(
        'SELECT id FROM nodes WHERE conditions = ?',
        (conditions,)
    )
    nodo_existente = cursor.fetchone()

    if nodo_existente:
        nodo_id = nodo_existente[0]
    else:
        cursor.execute('''
            INSERT INTO nodes (
                label, file_in_db, conditions,
                correct_percentage, successful_operations, total_operations,
                correct_percentage_os, successful_operations_os, total_operations_os
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            label, file_in_db, conditions,
            correct_percentage, successful_operations, total_operations,
            correct_percentage_os, successful_operations_os, total_operations_os
        ))
        nodo_id = cursor.lastrowid

    # 🔥 INSERT MASIVO register
    if fechas and veneficios and len(fechas) == len(veneficios):
        cursor.executemany(
            'INSERT INTO register (node_id, dates, veneficios) VALUES (?, ?, ?)',
            [(nodo_id, f, v) for f, v in zip(fechas, veneficios)]
        )

    # 🔥 INSERT MASIVO register_os
    if fechas_os and veneficios_os and len(fechas_os) == len(veneficios_os):
        cursor.executemany(
            'INSERT INTO register_os (node_id, dates_os, veneficios_os) VALUES (?, ?, ?)',
            [(nodo_id, f, v) for f, v in zip(fechas_os, veneficios_os)]
        )

    conn.commit()


def successful_operations_by_label(name, label: str) -> int:
    path = f'output/db/{name}.db'
    conn = get_connection(path)
    cursor = conn.cursor()

    cursor.execute(
        'SELECT SUM(successful_operations) FROM nodes WHERE label = ?', 
        (label,)
    )
    result = cursor.fetchone()[0]
    return result if result is not None else 0


def nodo_con_mas_fechas_hora_comunes(name, lista_fechas):
    path = f'output/db/{name}.db'
    conn = get_connection(path)
    cursor = conn.cursor()

    # Crear placeholders para la consulta SQL
    placeholders = ','.join('?' for _ in lista_fechas)

    # Buscar coincidencias exactas de fecha+hora en register
    cursor.execute(f'''
        SELECT node_id, dates
        FROM register
        WHERE dates IN ({placeholders})
    ''', lista_fechas)

    resultados = cursor.fetchall()


    if not resultados:
        return None

    # Contar coincidencias por node_id
    conteo_por_nodo = {}
    for node_id, _ in resultados:
        conteo_por_nodo[node_id] = conteo_por_nodo.get(node_id, 0) + 1

    # Encontrar el nodo con más coincidencias
    node_id_mas_comun = max(conteo_por_nodo, key=conteo_por_nodo.get)
    coincidencias = conteo_por_nodo[node_id_mas_comun]

    # Obtener total_operations del nodo
    conn = get_connection(f'output/db/{name}.db')
    cursor = conn.cursor()
    cursor.execute('SELECT total_operations FROM nodes WHERE id = ?', (node_id_mas_comun,))
    resultado = cursor.fetchone()
    

    if resultado:
        total_operations = resultado[0]
        return {
            'node_id': node_id_mas_comun,
            'coincidencias': coincidencias,
            'total_operations': total_operations
        }
    else:
        return None
    
    
def eliminar_nodo_y_registros(name, node_id):
    path = f'output/db/{name}.db'
    conn = get_connection(path)
    cursor = conn.cursor()

    # Eliminar registros relacionados en la tabla register
    cursor.execute('DELETE FROM register_os WHERE node_id = ?', (node_id,))
    cursor.execute('DELETE FROM register WHERE node_id = ?', (node_id,))

    # Eliminar el nodo de la tabla nodes
    cursor.execute('DELETE FROM nodes WHERE id = ?', (node_id,))

    conn.commit()
    

    print(f"Nodo {node_id} y sus registros han sido eliminados.")
    


def get_nodes(name):
    path = f'output/db/{name}.db'
    conn = get_connection(path)
    cursor = conn.cursor()

    # Buscar todos los registros de la tabla 'nodes'
    cursor.execute('SELECT * FROM nodes')
    resultados = cursor.fetchall()


    if not resultados:
        return None
    return resultados


def get_node_by_id(name, id_node):
    path = f'output/db/{name}.db'
    conn = get_connection(path)
    cursor = conn.cursor()

    # Buscar el nodo con el id especificado
    cursor.execute('SELECT * FROM nodes WHERE id = ?', (id_node,))
    resultado = cursor.fetchone()


    if resultado is None:
        return None
    return resultado


def existe_registro(name, fecha_busqueda, label_busqueda, modo):
    db_path = f'output/db/{name}.db'

    if not os.path.exists(db_path):
        print(f"La base de datos {db_path} no existe.")
        return False

    if modo not in ("os", "is"):
        raise ValueError("El modo debe ser 'os' o 'is'")

    conn = get_connection(db_path)
    cursor = conn.cursor()

    if modo == "os":
        query = '''
        SELECT ro.id
        FROM register_os ro
        JOIN nodes n ON ro.node_id = n.id
        WHERE n.label = ? AND ro.dates_os = ?
        LIMIT 1
        '''
    else:
        query = '''
        SELECT r.id
        FROM register r
        JOIN nodes n ON r.node_id = n.id
        WHERE n.label = ? AND r.dates = ?
        LIMIT 1
        '''

    cursor.execute(query, (label_busqueda, fecha_busqueda))
    resultado = cursor.fetchone()

    
    return resultado is not None


def promedio_correct_percentage(name, label_busqueda, modo):
    db_path = f'output/db/{name}.db'

    if not os.path.exists(db_path):
        print(f"La base de datos {db_path} no existe.")
        return None

    if modo not in ("os", "is"):
        raise ValueError("El modo debe ser 'os' o 'is'")

    conn = get_connection(db_path)
    cursor = conn.cursor()

    if modo == "os":
        query = '''
        SELECT AVG(correct_percentage_os)
        FROM nodes
        WHERE label = ?
        '''
    else:
        query = '''
        SELECT AVG(correct_percentage)
        FROM nodes
        WHERE label = ?
        '''

    cursor.execute(query, (label_busqueda,))
    resultado = cursor.fetchone()[0]

    return resultado  #


def sum_successful_operations(db_name, label):
    db_path = f'output/db/{db_name}.db'
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT SUM(successful_operations)
        FROM nodes
        WHERE label = ?
    ''', (label,))
    
    result = cursor.fetchone()[0]
    
    
    # Si no hay registros, devuelve 0 en vez de None
    return result if result is not None else 0

def delete_nodes_by_label(db_name, label):
    db_path = f'output/db/{db_name}.db'
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # Paso 1: obtener todos los IDs de nodos con el label dado
    cursor.execute("SELECT id FROM nodes WHERE label = ?", (label,))
    node_ids = [row[0] for row in cursor.fetchall()]

    if not node_ids:
        print(f"No se encontraron nodos con label '{label}'.")
        return

    # Paso 2: eliminar registros relacionados en register y register_os
    cursor.executemany("DELETE FROM register WHERE node_id = ?", [(nid,) for nid in node_ids])
    cursor.executemany("DELETE FROM register_os WHERE node_id = ?", [(nid,) for nid in node_ids])

    # Paso 3: eliminar nodos
    cursor.execute("DELETE FROM nodes WHERE label = ?", (label,))

    conn.commit()
    print(f"Eliminados {len(node_ids)} nodos y sus registros asociados.")
    
    
def get_dates_by_label(db_name, label, modo="is"):
    db_path = f'output/db/{db_name}.db'
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # Obtener IDs de nodos con ese label
    cursor.execute("SELECT id FROM nodes WHERE label = ?", (label,))
    node_ids = [row[0] for row in cursor.fetchall()]

    if not node_ids:
        return tuple()  # Tupla vacía si no hay nodos

    # Elegir tabla y columna según el modo
    if modo.lower() == "os":
        table = "register_os"
        column = "dates_os"
    else:
        table = "register"
        column = "dates"

    # Construir consulta dinámica
    placeholders = ",".join("?" * len(node_ids))
    query = f"SELECT {column} FROM {table} WHERE node_id IN ({placeholders})"
    cursor.execute(query, tuple(node_ids))

    # Obtener resultados como tupla
    results = tuple(row[0] for row in cursor.fetchall())

    return results


def get_nodes_label(name, label):
    path = f'output/db/{name}.db'
    conn = get_connection(path)
    cursor = conn.cursor()

    # Buscar todos los registros de la tabla 'nodes'
    cursor.execute('SELECT conditions FROM nodes WHERE label = ?', (label,))
    resultados = cursor.fetchall()


    if not resultados:
        return None
    return resultados


def get_nodes_by_label(name, label):
    path = f'output/db/{name}.db'
    conn = get_connection(path)
    cursor = conn.cursor()

    # Buscar todos los registros de la tabla 'nodes'
    cursor.execute('SELECT conditions, file_in_db FROM nodes WHERE label = ?', (label,))
    resultados = cursor.fetchall()


    if not resultados:
        return None
    return resultados