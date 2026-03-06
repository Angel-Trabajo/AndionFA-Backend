import psycopg2
from psycopg2.pool import SimpleConnectionPool

POOL = SimpleConnectionPool(
    minconn=1,
    maxconn=100,
    host="localhost",
    database="andionfa",
    user="postgres",
    password="angel3020",
    port=5432
)


def get_connection():
    return POOL.getconn()

def release_connection(conn):
    POOL.putconn(conn)