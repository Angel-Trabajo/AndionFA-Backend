import json
import logging
from pathlib import Path

from psycopg2.extras import execute_values
from src.db.postgres import get_connection, release_connection


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s - %(message)s'
    )


GENERAL_CONFIG_PATH = Path(__file__).resolve().parents[2] / 'config' / 'general_config.json'


def _get_allowed_indicator_files() -> set[str] | None:
    try:
        with GENERAL_CONFIG_PATH.open('r', encoding='utf8') as file:
            data = json.load(file)
    except Exception as exc:
        logger.warning('No se pudo cargar general_config para filtrar extractores: %s', exc)
        return None

    indicators_files = data.get('indicators_files')
    if not indicators_files:
        return None
    return set(indicators_files)


def insertar_nodo_con_registros(
    principal_symbol,
    symbol_cruce,
    label,
    mercado,
    file_in_db,
    conditions,
    correct_percentage,
    successful_operations,
    total_operations,
    correct_percentage_os,
    successful_operations_os,
    total_operations_os,
    fechas=None,
    veneficios=None,
    fechas_os=None,
    veneficios_os=None
):
    if not fechas or not veneficios:
        logger.warning(
            "Insert skip: fechas/veneficios vacíos (principal=%s, cruce=%s, label=%s, mercado=%s)",
            principal_symbol,
            symbol_cruce,
            label,
            mercado
        )
    if not fechas_os or not veneficios_os:
        logger.warning(
            "Insert OS vacío (principal=%s, cruce=%s, label=%s, mercado=%s)",
            principal_symbol,
            symbol_cruce,
            label,
            mercado
        )

    conn = get_connection()
    cursor = conn.cursor()

    # =====================================
    # UPSERT NODE (SIN SELECT)
    # =====================================
    try:
        cursor.execute("""
            INSERT INTO nodes (
                principal_symbol,
                symbol_cruce,
                label,
                mercado,
                file_in_db,
                conditions,
                correct_percentage,
                successful_operations,
                total_operations,
                correct_percentage_os,
                successful_operations_os,
                total_operations_os
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)

            ON CONFLICT (
                principal_symbol,
                symbol_cruce,
                mercado,
                label,
                conditions
            )
            DO UPDATE SET
                correct_percentage = EXCLUDED.correct_percentage,
                successful_operations = EXCLUDED.successful_operations,
                total_operations = EXCLUDED.total_operations,
                correct_percentage_os = EXCLUDED.correct_percentage_os,
                successful_operations_os = EXCLUDED.successful_operations_os,
                total_operations_os = EXCLUDED.total_operations_os

            RETURNING id
        """, (
            principal_symbol,
            symbol_cruce,
            label,
            mercado,
            file_in_db,
            conditions,
            correct_percentage,
            successful_operations,
            total_operations,
            correct_percentage_os,
            successful_operations_os,
            total_operations_os
        ))
    except Exception:
        logger.exception(
            "Error insertando nodo (principal=%s, cruce=%s, label=%s, mercado=%s, file=%s)",
            principal_symbol,
            symbol_cruce,
            label,
            mercado,
            file_in_db
        )
        release_connection(conn)
        raise

    nodo_id = cursor.fetchone()[0]

    # =====================================
    # REGISTER BULK INSERT
    # =====================================
    if fechas and veneficios:

        data = [
            (
                nodo_id,
                f,
                principal_symbol,
                symbol_cruce,
                mercado,
                v
            )
            for f, v in zip(fechas, veneficios)
        ]

        execute_values(
            cursor,
            """
            INSERT INTO register
            (node_id, dates, principal_symbol,
             symbol_cruce, mercado, veneficios)
            VALUES %s
            """,
            data,
            page_size=5000
        )

    # =====================================
    # REGISTER_OS BULK INSERT
    # =====================================
    if fechas_os and veneficios_os:

        data_os = [
            (
                nodo_id,
                f,
                principal_symbol,
                symbol_cruce,
                mercado,
                v
            )
            for f, v in zip(fechas_os, veneficios_os)
        ]

        execute_values(
            cursor,
            """
            INSERT INTO register_os
            (node_id, dates_os, principal_symbol,
             symbol_cruce, mercado, veneficios_os)
            VALUES %s
            """,
            data_os,
            page_size=5000
        )

    conn.commit()
    release_connection(conn)


def successful_operations_by_label(principal_symbol, symbol_cruce: str, label: str, mercado: str) -> int:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        'SELECT SUM(successful_operations) FROM nodes WHERE principal_symbol = %s AND symbol_cruce = %s AND label = %s AND mercado = %s', 
        (principal_symbol, symbol_cruce, label, mercado)
    )
    result = cursor.fetchone()[0]
    release_connection(conn)
    return result if result is not None else 0


def nodo_con_mas_fechas_hora_comunes(principal_symbol, symbol_cruce, mercado, lista_fechas):
    conn = get_connection()
    cursor = conn.cursor()

    query = """
        SELECT 
            r.node_id,
            COUNT(*) AS coincidencias,
            n.total_operations
        FROM register r
        JOIN nodes n ON n.id = r.node_id
        WHERE r.dates = ANY(%s::timestamp[])
        AND r.mercado = %s
        AND r.principal_symbol = %s
        AND r.symbol_cruce = %s
        GROUP BY r.node_id, n.total_operations
        ORDER BY coincidencias DESC
        LIMIT 1
    """

    cursor.execute(query, (lista_fechas, mercado, principal_symbol, symbol_cruce))
    resultado = cursor.fetchone()

    release_connection(conn)

    if resultado:
        return {
            "node_id": resultado[0],
            "coincidencias": resultado[1],
            "total_operations": resultado[2]
        }

    return None
  
    
def eliminar_nodo_y_registros(node_id):
    conn = get_connection()
    cursor = conn.cursor()

    # Eliminar registros relacionados en la tabla register
    cursor.execute('DELETE FROM register_os WHERE node_id = %s', (node_id,))
    cursor.execute('DELETE FROM register WHERE node_id = %s', (node_id,))

    # Eliminar el nodo de la tabla nodes
    cursor.execute('DELETE FROM nodes WHERE id = %s', (node_id,))
    conn.commit()
    release_connection(conn)

    print(f"Nodo {node_id} y sus registros han sido eliminados.")


def eliminar_nodos_y_registros(
    principal_symbol: str | None = None,
    symbol_cruce: str | None = None,
    mercado: str | None = None
    ):
    conn = get_connection()
    cursor = conn.cursor()

    try:
        has_filters = any([principal_symbol, symbol_cruce, mercado])

        if not has_filters:
            cursor.execute("TRUNCATE TABLE register_os, register, nodes")
            conn.commit()
            print(
                "Nodos y registros han sido eliminados. "
                "Filtros aplicados: principal_symbol=None, symbol_cruce=None, mercado=None"
            )
            return

        conditions = []
        params = []

        if principal_symbol:
            conditions.append("n.principal_symbol = %s")
            params.append(principal_symbol)
        if symbol_cruce:
            conditions.append("n.symbol_cruce = %s")
            params.append(symbol_cruce)
        if mercado:
            conditions.append("n.mercado = %s")
            params.append(mercado)

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        cursor.execute(
            f"""
            WITH target_nodes AS (
                SELECT n.id
                FROM nodes n
                WHERE {where_clause}
            ),
            deleted_register_os AS (
                DELETE FROM register_os ro
                USING target_nodes tn
                WHERE ro.node_id = tn.id
                RETURNING ro.node_id
            ),
            deleted_register AS (
                DELETE FROM register r
                USING target_nodes tn
                WHERE r.node_id = tn.id
                RETURNING r.node_id
            )
            DELETE FROM nodes n
            USING target_nodes tn
            WHERE n.id = tn.id
            """,
            tuple(params)
        )
        conn.commit()

        print(
            f"Nodos y registros han sido eliminados. Filtros aplicados: "
            f"principal_symbol={principal_symbol}, symbol_cruce={symbol_cruce}, mercado={mercado}"
        )
    finally:
        release_connection(conn)


def promedio_correct_percentage(principal_symbol, symbol_cruce, mercado, label, modo):

    conn = get_connection()
    cursor = conn.cursor()

    if modo == "os":
        query = '''
        SELECT AVG(correct_percentage_os)
        FROM nodes
        WHERE label = %s AND mercado = %s AND principal_symbol = %s AND symbol_cruce = %s
        '''
    else:
        query = '''
        SELECT AVG(correct_percentage)
        FROM nodes
        WHERE label = %s AND mercado = %s AND principal_symbol = %s AND symbol_cruce = %s
        '''

    cursor.execute(query, (label, mercado, principal_symbol, symbol_cruce))
    resultado = cursor.fetchone()[0]

    release_connection(conn)
    return resultado  
    

def get_dates_by_label(principal_symbol, symbol_cruce, mercado, label, modo="is"):

    conn = get_connection()
    cursor = conn.cursor()

    # Obtener IDs de nodos con ese label
    cursor.execute(
        "SELECT id FROM nodes WHERE label = %s AND mercado = %s AND principal_symbol = %s AND symbol_cruce = %s", 
        (label, mercado, principal_symbol, symbol_cruce))
    node_ids = [row[0] for row in cursor.fetchall()]

    if not node_ids:
        release_connection(conn)
        return tuple()  # Tupla vacía si no hay nodos

    # Elegir tabla y columna según el modo
    if modo.lower() == "os":
        table = "register_os"
        column = "dates_os"
    else:
        table = "register"
        column = "dates"

    query = f"SELECT {column} FROM {table} WHERE node_id = ANY(%s)"
    cursor.execute(query, (node_ids,))

    # Obtener resultados como tupla
    results = tuple(row[0] for row in cursor.fetchall())
    release_connection(conn)
    return results


def get_nodes(principal_symbol, symbol_cruce, mercado=None, label=None):
    conn = get_connection()
    cursor = conn.cursor()

    if mercado is None:
        cursor.execute(
            'SELECT * FROM nodes WHERE label = %s AND principal_symbol = %s AND symbol_cruce = %s',
            (label, principal_symbol, symbol_cruce)
        )
    else:
        cursor.execute(
            'SELECT * FROM nodes WHERE label = %s AND principal_symbol = %s AND symbol_cruce = %s AND mercado = %s',
            (label, principal_symbol, symbol_cruce, mercado)
        )

    res = cursor.fetchall()
    release_connection(conn)
    return res if res else None


def get_nodes_by_label(principal_symbol, symbol_cruce, mercado=None, label=None):
    conn = get_connection()
    cursor = conn.cursor()

    # Buscar todos los registros de la tabla 'nodes'
    if mercado is None:
        cursor.execute(
            'SELECT conditions, file_in_db FROM nodes WHERE label = %s AND principal_symbol = %s AND symbol_cruce = %s',
            (label, principal_symbol, symbol_cruce)
        )
    else:
        cursor.execute(
            'SELECT conditions, file_in_db FROM nodes WHERE label = %s AND principal_symbol = %s AND symbol_cruce = %s AND mercado = %s',
            (label, principal_symbol, symbol_cruce, mercado)
        )
    resultados = cursor.fetchall()

    release_connection(conn)
    if not resultados:
        return None

    allowed_indicator_files = _get_allowed_indicator_files()
    if allowed_indicator_files is not None:
        resultados = [
            result for result in resultados
            if f"{str(result[1]).split('_')[0]}.csv" in allowed_indicator_files
        ]

    if not resultados:
        return None
    return resultados


# def get_node_by_id(name, id_node):
#     path = f'output/db/{name}.db'
#     conn = sqlite3.connect(path)
#     cursor = conn.cursor()

#     # Buscar el nodo con el id especificado
#     cursor.execute('SELECT * FROM nodes WHERE id = ?', (id_node,))
#     resultado = cursor.fetchone()
#     conn.close()

#     if resultado is None:
#         return None
#     return resultado


# def existe_registro(name, fecha_busqueda, label_busqueda, modo):
#     db_path = f'output/db/{name}.db'

#     if not os.path.exists(db_path):
#         print(f"La base de datos {db_path} no existe.")
#         return False

#     if modo not in ("os", "is"):
#         raise ValueError("El modo debe ser 'os' o 'is'")

#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()

#     if modo == "os":
#         query = '''
#         SELECT ro.id
#         FROM register_os ro
#         JOIN nodes n ON ro.node_id = n.id
#         WHERE n.label = ? AND ro.dates_os = ?
#         LIMIT 1
#         '''
#     else:
#         query = '''
#         SELECT r.id
#         FROM register r
#         JOIN nodes n ON r.node_id = n.id
#         WHERE n.label = ? AND r.dates = ?
#         LIMIT 1
#         '''

#     cursor.execute(query, (label_busqueda, fecha_busqueda))
#     resultado = cursor.fetchone()

#     conn.close()
#     return resultado is not None



# def sum_successful_operations(db_name, label):
#     db_path = f'output/db/{db_name}.db'
#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()

#     cursor.execute('''
#         SELECT SUM(successful_operations)
#         FROM nodes
#         WHERE label = ?
#     ''', (label,))
    
#     result = cursor.fetchone()[0]
#     conn.close()
    
#     # Si no hay registros, devuelve 0 en vez de None
#     return result if result is not None else 0

# def delete_nodes_by_label(db_name, label):
#     db_path = f'output/db/{db_name}.db'
#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()

#     # Paso 1: obtener todos los IDs de nodos con el label dado
#     cursor.execute("SELECT id FROM nodes WHERE label = ?", (label,))
#     node_ids = [row[0] for row in cursor.fetchall()]

#     if not node_ids:
#         print(f"No se encontraron nodos con label '{label}'.")
#         return

#     # Paso 2: eliminar registros relacionados en register y register_os
#     cursor.executemany("DELETE FROM register WHERE node_id = ?", [(nid,) for nid in node_ids])
#     cursor.executemany("DELETE FROM register_os WHERE node_id = ?", [(nid,) for nid in node_ids])

#     # Paso 3: eliminar nodos
#     cursor.execute("DELETE FROM nodes WHERE label = ?", (label,))

#     conn.commit()
#     conn.close()
#     print(f"Eliminados {len(node_ids)} nodos y sus registros asociados.")
    


# def get_nodes_label(name, label):
#     path = f'output/db/{name}.db'
#     conn = sqlite3.connect(path)
#     cursor = conn.cursor()

#     # Buscar todos los registros de la tabla 'nodes'
#     cursor.execute('SELECT conditions FROM nodes WHERE label = ?', (label,))
#     resultados = cursor.fetchall()

#     conn.close()
#     if not resultados:
#         return None
#     return resultados


