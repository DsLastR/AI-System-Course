import sqlite3
from datetime import datetime

DB_PATH = "../result/emotion_results.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS emotion_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_name TEXT NOT NULL,
        detected_faces INTEGER,
        emotion_label TEXT,
        confidence REAL,
        predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

def insert_result(file_name, detected_faces, label, confidence):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO emotion_results (file_name, detected_faces, emotion_label, confidence)
        VALUES (?, ?, ?, ?)
    """, (file_name, detected_faces, label, confidence))
    conn.commit()
    conn.close()
