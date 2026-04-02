import psycopg2
from psycopg2.pool import SimpleConnectionPool

POOL: SimpleConnectionPool | None = None

_DB_KWARGS = dict(
    host="localhost",
    database="andionfa",
    user="postgres",
    password="angel3020",
    port=5432,
)


def init_pool() -> None:
    """Crea (o recrea) el pool de conexiones."""
    global POOL
    if POOL is not None:
        try:
            POOL.closeall()
        except Exception:
            pass
    POOL = SimpleConnectionPool(minconn=1, maxconn=100, **_DB_KWARGS)


# Inicializar al cargar el módulo
init_pool()


def get_connection():
    if POOL is None:
        init_pool()
    return POOL.getconn()  # type: ignore[union-attr]


def release_connection(conn):
    if POOL is not None:
        POOL.putconn(conn)