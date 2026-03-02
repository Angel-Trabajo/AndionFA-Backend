import sqlite3
import os

def create_db(name, principal_symbol):
    db_path = f'output/{principal_symbol}/db/{name}.db'

    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tablas = cursor.fetchall()

        for tabla in tablas:
            nombre_tabla = tabla[0]
            if nombre_tabla.startswith("sqlite_"):
                continue  # evita eliminar tablas internas
            cursor.execute(f"DROP TABLE IF EXISTS {nombre_tabla}")
            print(f"Tabla eliminada: {nombre_tabla}")

        conn.commit()
        conn.close()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS nodes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        label TEXT,
        mercado TEXT,
        file_in_db TEXT,
        conditions TEXT,
        correct_percentage REAL,
        successful_operations INTEGER,
        total_operations INTEGER,
        correct_percentage_os REAL,
        successful_operations_os INTEGER,
        total_operations_os INTEGER
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS register (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        node_id INTEGER,
        dates TEXT,
        mercado TEXT,
        veneficios REAL,
        FOREIGN KEY (node_id) REFERENCES nodes(id)
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS register_os (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        node_id INTEGER,
        dates_os TEXT,
        mercado TEXT,
        veneficios_os REAL,
        FOREIGN KEY (node_id) REFERENCES nodes(id)
    )
    ''')

    conn.commit()
    conn.close()
