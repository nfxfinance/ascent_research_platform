import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, List

def get_db_path() -> Path:
    db_dir = Path("data")
    db_dir.mkdir(exist_ok=True)
    return db_dir / "data_sources.db"

def init_data_source_table():
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS data_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_name TEXT UNIQUE NOT NULL,
            type TEXT NOT NULL,
            ip TEXT,
            port INTEGER,
            path TEXT,
            username TEXT,
            password TEXT,
            created_at TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def insert_or_update_data_source(config: Dict[str, Any]):
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO data_sources
        (source_name, type, ip, port, path, username, password, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        config["source_name"], config["type"], config.get("ip"), config.get("port"),
        config.get("path"), config.get("username"), config.get("password"), config["created_at"]
    ))
    conn.commit()
    conn.close()

def get_all_data_sources() -> List[Dict[str, Any]]:
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute('SELECT source_name, type, ip, port, path, username, password, created_at FROM data_sources')
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "source_name": row[0],
            "type": row[1],
            "ip": row[2],
            "port": row[3],
            "path": row[4],
            "username": row[5],
            "password": row[6],
            "created_at": row[7],
        } for row in rows
    ]

def delete_data_source(source_name: str):
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute('DELETE FROM data_sources WHERE source_name = ?', (source_name,))
    conn.commit()
    conn.close()
