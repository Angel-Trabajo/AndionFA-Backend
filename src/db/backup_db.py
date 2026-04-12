import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from src.db.postgres import _DB_KWARGS

BACKUP_DIR = Path("backup")

# Buscar pg_dump: primero en PATH, luego en ubicaciones típicas de Windows
def _find_pg_tool(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    pg_dirs = sorted(Path("C:/Program Files/PostgreSQL").glob("*/bin"), reverse=True)
    for d in pg_dirs:
        candidate = d / f"{name}.exe"
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"No se encontró '{name}' en PATH ni en PostgreSQL/bin")

_DB_NAME = _DB_KWARGS["database"]
_DB_USER = _DB_KWARGS["user"]
_DB_PASSWORD = _DB_KWARGS["password"]
_DB_HOST = _DB_KWARGS["host"]
_DB_PORT = str(_DB_KWARGS["port"])

_SAFE_FILENAME = re.compile(r'^[a-zA-Z0-9_\-\.]+\.sql$')


def _safe_path(file_name: str) -> Path:
    """Valida que el nombre de archivo no contenga path traversal."""
    if not _SAFE_FILENAME.match(file_name):
        raise ValueError(f"Nombre de archivo no válido: {file_name!r}")
    return BACKUP_DIR / file_name


def _pg_env() -> dict:
    env = os.environ.copy()
    env["PGPASSWORD"] = _DB_PASSWORD
    return env


def create_backup() -> str:
    """Genera un archivo .sql con pg_dump. Devuelve el nombre del archivo creado."""
    BACKUP_DIR.mkdir(exist_ok=True)
    file_name = f"{_DB_NAME}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.sql"
    backup_path = BACKUP_DIR / file_name

    subprocess.run(
        [_find_pg_tool("pg_dump"), "-h", _DB_HOST, "-p", _DB_PORT, "-U", _DB_USER, "-d", _DB_NAME, "-f", str(backup_path)],
        check=True,
        env=_pg_env(),
        capture_output=True,
        text=True,
    )
    return file_name


def restore_backup(file_name: str) -> None:
    """Restaura la DB desde un .sql generado por esta misma app."""
    backup_path = _safe_path(file_name)
    if not backup_path.exists():
        raise FileNotFoundError(f"Backup no encontrado: {file_name}")

    subprocess.run(
        [_find_pg_tool("psql"), "-h", _DB_HOST, "-p", _DB_PORT, "-U", _DB_USER, "-d", _DB_NAME, "-f", str(backup_path)],
        check=True,
        env=_pg_env(),
        capture_output=True,
        text=True,
    )


def list_backups() -> list[str]:
    """Devuelve la lista de archivos .sql en la carpeta backup/, ordenados de más nuevo a más antiguo."""
    BACKUP_DIR.mkdir(exist_ok=True)
    files = sorted(BACKUP_DIR.glob("*.sql"), key=lambda f: f.stat().st_mtime, reverse=True)
    return [f.name for f in files]
