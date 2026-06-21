"""
conversation_store.py
Handles saving and loading chat conversations to a local SQLite database
so conversation history persists even after closing and reopening the app.
"""

import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "jobsense_history.db")


def init_db():
    """Creates the conversations table if it doesn't already exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            chat_history TEXT,
            jd_text_raw TEXT,
            resume_text_raw TEXT,
            fit_result TEXT,
            tailoring_result TEXT,
            created_at TEXT,
            updated_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_conversation(conversation_id, title, chat_history, jd_text_raw, resume_text_raw, fit_result, tailoring_result):
    """
    Saves or updates a conversation. If conversation_id is None, creates a new row
    and returns its new id. Otherwise, updates the existing row.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.now().isoformat()

    chat_history_json = json.dumps(chat_history)

    if conversation_id is None:
        cursor.execute("""
            INSERT INTO conversations
            (title, chat_history, jd_text_raw, resume_text_raw, fit_result, tailoring_result, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (title, chat_history_json, jd_text_raw, resume_text_raw, fit_result, tailoring_result, now, now))
        new_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return new_id
    else:
        cursor.execute("""
            UPDATE conversations
            SET title = ?, chat_history = ?, jd_text_raw = ?, resume_text_raw = ?,
                fit_result = ?, tailoring_result = ?, updated_at = ?
            WHERE id = ?
        """, (title, chat_history_json, jd_text_raw, resume_text_raw, fit_result, tailoring_result, now, conversation_id))
        conn.commit()
        conn.close()
        return conversation_id


def list_conversations():
    """Returns all conversations ordered by most recently updated first, as a list of dicts."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, updated_at FROM conversations ORDER BY updated_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [{"id": row[0], "title": row[1], "updated_at": row[2]} for row in rows]


def load_conversation(conversation_id):
    """Loads a single conversation by id. Returns a dict or None if not found."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, title, chat_history, jd_text_raw, resume_text_raw, fit_result, tailoring_result
        FROM conversations WHERE id = ?
    """, (conversation_id,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return {
        "id": row[0],
        "title": row[1],
        "chat_history": json.loads(row[2]) if row[2] else [],
        "jd_text_raw": row[3],
        "resume_text_raw": row[4],
        "fit_result": row[5],
        "tailoring_result": row[6],
    }


def delete_conversation(conversation_id):
    """Deletes a conversation by id."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    conn.commit()
    conn.close()


def generate_title_from_history(chat_history):
    """
    Generates a short title for a conversation based on its first user message.
    Falls back to a timestamp-based title if no user message exists yet.
    """
    for msg in chat_history:
        if msg["role"] == "user" and not msg["content"].startswith("📎"):
            text = msg["content"].strip()
            return text[:40] + ("..." if len(text) > 40 else "")
    for msg in chat_history:
        if msg["role"] == "user":
            return msg["content"][:40]
    return "New Conversation - " + datetime.now().strftime("%b %d, %H:%M")
