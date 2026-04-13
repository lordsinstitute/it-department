import sqlite3
import os

DB_NAME = "attendance.db"


def get_connection():
    """Return a new SQLite connection"""
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    return conn


def init_db():
    """Create tables if not existing"""
    conn = get_connection()
    cursor = conn.cursor()

    # Students Table
    cursor.execute('''CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT UNIQUE,
            name TEXT,
            images_captured INTEGER DEFAULT 0
        )''')

    # Attendance Table
    cursor.execute('''CREATE TABLE IF NOT EXISTS attendance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_name TEXT NOT NULL,
                        date TEXT NOT NULL,
                        time TEXT NOT NULL
                    )''')

    # Admin Table
    cursor.execute('''CREATE TABLE IF NOT EXISTS admin (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL
                    )''')

    conn.commit()
    conn.close()


def add_admin(username, password_hash):
    """Insert admin account if not existing"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM admin WHERE username=?", (username,))
    admin = cursor.fetchone()
    if not admin:
        cursor.execute("INSERT INTO admin (username, password) VALUES (?, ?)", (username, password_hash))
        conn.commit()
    conn.close()
