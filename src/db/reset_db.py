import psycopg2
import src.db.postgres as _pg

DB_NAME = "andionfa"

def reset_database():
    # 1. Cerrar pool (CRÍTICO)
    try:
        if _pg.POOL is not None:
            _pg.POOL.closeall()
            _pg.POOL = None
    except Exception:
        pass

    # 2. Conectarse a postgres (no a la DB a borrar)
    conn = psycopg2.connect(
        host="localhost",
        database="postgres",
        user="postgres",
        password="angel3020",
        port=5432
    )
    conn.autocommit = True
    cursor = conn.cursor()

    # 3. Matar conexiones activas
    cursor.execute(f"""
        SELECT pg_terminate_backend(pid)
        FROM pg_stat_activity
        WHERE datname = '{DB_NAME}'
        AND pid <> pg_backend_pid();
    """)

    # 4. Drop + Create
    cursor.execute(f"DROP DATABASE IF EXISTS {DB_NAME};")
    cursor.execute(f"CREATE DATABASE {DB_NAME};")

    cursor.close()
    conn.close()

    # 5. Crear estructura
    conn = psycopg2.connect(
        host="localhost",
        database=DB_NAME,
        user="postgres",
        password="angel3020",
        port=5432
    )
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE nodes (
        id SERIAL PRIMARY KEY,
        principal_symbol TEXT NOT NULL,
        symbol_cruce TEXT NOT NULL,
        label TEXT NOT NULL,
        mercado TEXT NOT NULL,
        file_in_db TEXT,
        conditions TEXT,
        correct_percentage FLOAT,
        successful_operations INT,
        total_operations INT,
        correct_percentage_os FLOAT,
        successful_operations_os INT,
        total_operations_os INT,
        expectancy FLOAT,
        profit_factor FLOAT,
        sharpe_like FLOAT,
        drawdown_ratio FLOAT,
        quality_score FLOAT,
        max_losing_streak INT,
        expectancy_os FLOAT,
        profit_factor_os FLOAT,
        sharpe_like_os FLOAT,
        drawdown_ratio_os FLOAT,
        quality_score_os FLOAT,
        max_losing_streak_os INT,
        UNIQUE (
            principal_symbol,
            symbol_cruce,
            mercado,
            label,
            conditions
        )
    );
    """)

    cursor.execute("""
    CREATE TABLE register (
        id SERIAL PRIMARY KEY,
        node_id INT REFERENCES nodes(id) ON DELETE CASCADE,
        dates TIMESTAMP,
        principal_symbol TEXT,
        symbol_cruce TEXT,
        mercado TEXT,
        veneficios FLOAT
    );
    """)

    cursor.execute("""
    CREATE TABLE register_os (
        id SERIAL PRIMARY KEY,
        node_id INT REFERENCES nodes(id) ON DELETE CASCADE,
        dates_os TIMESTAMP,
        principal_symbol TEXT,
        symbol_cruce TEXT,
        mercado TEXT,
        veneficios_os FLOAT
    );
    """)

    conn.commit()
    cursor.close()
    conn.close()

    # 6. Recrear pool → la API sigue viva sin errores
    _pg.init_pool()

    print("Base de datos reseteada completamente ⚡")