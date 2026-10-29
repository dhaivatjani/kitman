"""SQLite store. Every record points at what it came from, so any model can be
traced back to the frames (and the match and second) it was trained on.

  match (an added video, stored in data/videos) ─< sample_run ─< frame >─ batch_frame >─ batch
  label_set (pre-labels from a model, or corrections from Roboflow) ─< label >─ frame
  training_run (label sets in) ─> model (weights out) ─> next round's pre-labels
"""

import json
import sqlite3
import threading
from datetime import datetime

from config import DATA, DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS match (
  id TEXT PRIMARY KEY, file TEXT NOT NULL, path TEXT NOT NULL, description TEXT,
  split TEXT NOT NULL CHECK (split IN ('train', 'val', 'test')),
  duration_s REAL, frames INTEGER, avg_fps REAL, width INTEGER, height INTEGER,
  status TEXT NOT NULL, message TEXT, created TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS sample_run (
  id INTEGER PRIMARY KEY, match_id TEXT NOT NULL REFERENCES match(id),
  mode TEXT NOT NULL CHECK (mode IN ('custom', 'fixed')), params TEXT NOT NULL,
  status TEXT NOT NULL, n_frames INTEGER DEFAULT 0, created TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS frame (
  id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, match_id TEXT NOT NULL REFERENCES match(id),
  sample_run_id INTEGER NOT NULL REFERENCES sample_run(id), path TEXT NOT NULL,
  t REAL NOT NULL, frame_idx INTEGER NOT NULL, bucket TEXT, round INTEGER DEFAULT 1);

CREATE TABLE IF NOT EXISTS model (
  id INTEGER PRIMARY KEY, name TEXT NOT NULL, path TEXT NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('coco', 'kitman')), training_run_id INTEGER,
  created TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS batch (
  id INTEGER PRIMARY KEY, name TEXT NOT NULL, round INTEGER NOT NULL, notes TEXT,
  status TEXT NOT NULL, created TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS batch_frame (
  batch_id INTEGER NOT NULL REFERENCES batch(id), frame_id INTEGER NOT NULL REFERENCES frame(id),
  PRIMARY KEY (batch_id, frame_id));

CREATE TABLE IF NOT EXISTS label_set (
  id INTEGER PRIMARY KEY, batch_id INTEGER NOT NULL REFERENCES batch(id),
  kind TEXT NOT NULL CHECK (kind IN ('prelabel', 'corrected')), source TEXT NOT NULL,
  model_id INTEGER REFERENCES model(id), created TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS label (
  label_set_id INTEGER NOT NULL REFERENCES label_set(id), frame_id INTEGER NOT NULL REFERENCES frame(id),
  path TEXT NOT NULL, n_player INTEGER, n_goalkeeper INTEGER, n_referee INTEGER, n_ball INTEGER,
  PRIMARY KEY (label_set_id, frame_id));

CREATE TABLE IF NOT EXISTS training_run (
  id INTEGER PRIMARY KEY, name TEXT NOT NULL, base_model TEXT NOT NULL, params TEXT NOT NULL,
  label_set_ids TEXT NOT NULL, counts TEXT, status TEXT NOT NULL, metrics TEXT,
  test_metrics TEXT, run_dir TEXT, created TEXT NOT NULL, finished TEXT, pid INTEGER);

CREATE TABLE IF NOT EXISTS job (
  id INTEGER PRIMARY KEY, kind TEXT NOT NULL, title TEXT NOT NULL, status TEXT NOT NULL,
  progress REAL DEFAULT 0, message TEXT, log TEXT DEFAULT '', created TEXT NOT NULL, finished TEXT);
"""

_lock = threading.Lock()   # sqlite allows one writer; jobs write from background threads


def now():
    return datetime.now().isoformat(timespec="seconds")


def connect():
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def q(sql, args=()):
    with connect() as con:
        return [dict(r) for r in con.execute(sql, args).fetchall()]


def q1(sql, args=()):
    rows = q(sql, args)
    return rows[0] if rows else None


def x(sql, args=()):
    with _lock, connect() as con:
        return con.execute(sql, args).lastrowid


def xmany(sql, rows):
    with _lock, connect() as con:
        con.executemany(sql, rows)


def init():
    DATA.mkdir(parents=True, exist_ok=True)
    with _lock, connect() as con:
        con.executescript(SCHEMA)
        if "pid" not in [r[1] for r in con.execute("PRAGMA table_info(training_run)")]:   # databases made before Stop existed
            con.execute("ALTER TABLE training_run ADD COLUMN pid INTEGER")
        if "rf_project" not in [r[1] for r in con.execute("PRAGMA table_info(batch)")]:   # Roboflow target per batch
            from config import local_config
            cfg = local_config()   # batches made before this point at the one project Settings used to hold
            con.execute("ALTER TABLE batch ADD COLUMN rf_workspace TEXT")
            con.execute("ALTER TABLE batch ADD COLUMN rf_project TEXT")
            con.execute("UPDATE batch SET rf_workspace=?, rf_project=?",
                        (cfg.get("roboflow_workspace"), cfg.get("roboflow_project")))
        if "rf_sent_to" not in [r[1] for r in con.execute("PRAGMA table_info(batch)")]:   # where the frames really went
            con.execute("ALTER TABLE batch ADD COLUMN rf_sent_to TEXT")
            con.execute("UPDATE batch SET rf_sent_to=rf_project WHERE status IN ('sent to Roboflow','corrected')")
        from config import COCO_MODEL
        if not con.execute("SELECT 1 FROM model WHERE kind='coco'").fetchone():
            con.execute("INSERT INTO model (name, path, kind, created) VALUES (?,?,?,?)",
                        ("YOLO11m COCO (pretrained)", str(COCO_MODEL), "coco", now()))
        # Jobs that were running when the server stopped will never finish.
        # (Training runs are separate processes that report their own status, so they survive.)
        con.execute("UPDATE job SET status='failed', message='Server stopped while running' "
                    "WHERE status IN ('queued','running')")
        con.execute("UPDATE match SET status='failed', message='Server stopped while it was being prepared' "
                    "WHERE status IN ('queued','copying','analysing')")


def jdump(v):
    return json.dumps(v, separators=(",", ":"))
