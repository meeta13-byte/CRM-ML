"""Prediction and campaign-analysis history in SQLite (stdlib, no server).

Two tables:
- predictions        one row per lead-conversion score
- campaign_analyses  one row per campaign plan / benchmark run

Everything is stored as JSON plus a few indexable columns, so a record can be
reopened exactly as it was created.
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import HISTORY_DB


def _connect(path=None):
    path = Path(path or HISTORY_DB)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path=None):
    with _connect(path) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created TEXT NOT NULL,
            dataset TEXT NOT NULL,
            model TEXT NOT NULL,
            features TEXT NOT NULL,
            probability REAL,
            probability_supported INTEGER NOT NULL DEFAULT 1,
            decision_score REAL,
            prediction INTEGER NOT NULL,
            priority TEXT,
            actual INTEGER
        );
        CREATE TABLE IF NOT EXISTS campaign_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created TEXT NOT NULL,
            objective TEXT NOT NULL,
            context TEXT NOT NULL,
            result TEXT NOT NULL,
            methodology TEXT NOT NULL
        );
        """)


def save_prediction(dataset, model, features, prediction, probability=None,
                    probability_supported=True, decision_score=None, priority=None,
                    path=None):
    init_db(path)
    with _connect(path) as conn:
        cur = conn.execute(
            """INSERT INTO predictions
               (created, dataset, model, features, probability,
                probability_supported, decision_score, prediction, priority)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now().isoformat(timespec="seconds"), dataset, model,
             json.dumps(features, default=str), probability,
             int(probability_supported), decision_score, int(prediction), priority),
        )
        return int(cur.lastrowid)


def set_actual(prediction_id, actual, path=None):
    with _connect(path) as conn:
        conn.execute("UPDATE predictions SET actual = ? WHERE id = ?",
                     (None if actual is None else int(actual), int(prediction_id)))


def list_predictions(limit=200, path=None):
    init_db(path)
    with _connect(path) as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def get_prediction(prediction_id, path=None):
    with _connect(path) as conn:
        row = conn.execute("SELECT * FROM predictions WHERE id = ?",
                           (int(prediction_id),)).fetchone()
    if row is None:
        return None
    record = dict(row)
    record["features"] = json.loads(record["features"])
    return record


def save_campaign(context, result, objective, methodology, path=None):
    init_db(path)
    with _connect(path) as conn:
        cur = conn.execute(
            """INSERT INTO campaign_analyses
               (created, objective, context, result, methodology)
               VALUES (?, ?, ?, ?, ?)""",
            (datetime.now().isoformat(timespec="seconds"), objective,
             json.dumps(context, default=str), json.dumps(result, default=str),
             methodology),
        )
        return int(cur.lastrowid)


def list_campaigns(limit=200, path=None):
    init_db(path)
    with _connect(path) as conn:
        rows = conn.execute(
            "SELECT * FROM campaign_analyses ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def get_campaign(analysis_id, path=None):
    with _connect(path) as conn:
        row = conn.execute("SELECT * FROM campaign_analyses WHERE id = ?",
                           (int(analysis_id),)).fetchone()
    if row is None:
        return None
    record = dict(row)
    record["context"] = json.loads(record["context"])
    record["result"] = json.loads(record["result"])
    return record
