"""The class list (Classes page). A class's id is its position in the name-sorted list, the way
Roboflow numbers them, so adding, merging or deleting a class renumbers every stored label file.
Merged and deleted names are remembered as aliases, so old Roboflow exports and models still read.
A change renumbers in the job queue (one job at a time, so no pre-label or import runs mid-way); the
class list itself switches over with the files when the job finishes."""

import colorsys
import random
import re
import shutil
import sqlite3
import threading
from datetime import datetime

import db
import jobs
import labels
from config import DATA, DB_PATH

_changing = threading.Lock()   # one change at a time: a double-clicked button must not renumber twice
pending = None   # while a change's job hasn't finished: {"what", "colors": [(name, colour)] the page shows}
NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")   # also used as a CSS variable name (--c-<name>)


def names():
    return [r["name"] for r in db.q("SELECT name FROM class ORDER BY name")]


def colors():
    return [(r["name"], r["color"]) for r in db.q("SELECT * FROM class ORDER BY name")]


def aliases():
    return {r["name"]: r["target"] for r in db.q("SELECT * FROM class_alias")}


def read_names(theirs, live=None, alias=None):
    """theirs: {their id: name} -> ({their id: our id or None when deleted}, [names we've never had])."""
    live = names() if live is None else live
    alias = aliases() if alias is None else alias
    ours, unknown = {}, []
    for i, n in theirs.items():
        if n in live:
            ours[i] = live.index(n)
        elif n in alias:
            ours[i] = live.index(alias[n]) if alias[n] else None
        else:
            unknown.append(n)
    return ours, unknown


def random_color():
    return "#%02x%02x%02x" % tuple(round(c * 255) for c in colorsys.hls_to_rgb(random.random(), 0.5, 0.75))


def busy():
    """Why classes can't change now, or None."""
    if pending:
        return f"labels are still being renumbered ({pending['what']})."
    return training()


def training():
    """A training run reads label files; they mustn't be renumbered under it."""
    if db.q1("SELECT 1 FROM training_run WHERE status IN ('queued','running')"):
        return "a training run is in progress."
    return None


def shown():
    """The list the Classes page shows: the pending change's result as soon as it's asked for."""
    return pending["colors"] if pending else colors()


def renumber(old, new, new_of):
    """{old id: new id} for classes that survive; old/new are name lists, new_of {old name: new name or None}."""
    return {i: new.index(new_of[n]) for i, n in enumerate(old) if new_of.get(n, n)}


def start(what, new_of, sql, added=None, color=None):
    """Queue the change and show its result on the Classes page straight away. Call holding _changing."""
    global pending
    if why := busy():
        raise ValueError(f"Classes can't change while {why}")
    shown_now = {new_of.get(n, n): c for n, c in colors() if new_of.get(n, n)} | ({added: color} if added else {})
    pending = {"what": what, "colors": sorted(shown_now.items())}
    try:
        jobs.submit("classes", what, run, what, new_of, sql, added)
    except BaseException:
        pending = None
        raise


def run(job, what, new_of, sql, added):
    global pending
    try:
        return f"{what}: {change(new_of, sql, added, job)} label files renumbered"
    finally:
        pending = None   # done or failed: the page shows the real list again


def change(new_of, sql, added=None, job=None):
    """Rewrite every stored label for the new class list, then run sql [(statement, args)] in one transaction.
    new_of: {old name: new name, or None to delete its boxes}; names not in it stay. added: a new class name."""
    if why := training():
        raise ValueError(f"Classes can't change while {why}")
    old = names()
    new = sorted(({new_of.get(n, n) for n in old} | {added}) - {None})
    new_of = {n: new_of.get(n, n) for n in old}
    ids = renumber(old, new, new_of)
    rows = db.q("SELECT * FROM label")
    files = []
    for k, r in enumerate(rows):   # read everything first: a bad file stops the change before anything is written
        boxes = [(ids[c], *v) for c, *v in labels.read(r) if c in ids]
        files.append((r, boxes))
        if job and k % 100 == 0:
            job.progress(0.5 * k / len(rows), f"Read {k} of {len(rows)} label files")
    if job:   # no progress inside the write below: it holds db._lock, which progress() also takes
        job.progress(0.5, f"Backing up and rewriting {len(rows)} label files")
    backup()
    with db._lock, db.connect() as con:   # commits only if every statement and file write succeeds
        for s, a in sql:   # first, so a refused statement stops the change before any file is touched
            con.execute(s, a)
        for r, boxes in files:
            con.execute("UPDATE label SET counts=? WHERE label_set_id=? AND frame_id=?",
                        (db.jdump(labels.counts(boxes, new)), r["label_set_id"], r["frame_id"]))
        for r, boxes in files:
            labels.write_file(DATA / r["path"], boxes)
    return len(files)


def backup():
    """data/backups/classes_<time>/: the database and every label file, as they were before the change."""
    dst = DATA / "backups" / f"classes_{datetime.now():%Y%m%d_%H%M%S_%f}"
    dst.mkdir(parents=True)
    with sqlite3.connect(DB_PATH) as src, sqlite3.connect(dst / "kitman.db") as out:
        src.backup(out)
    if (DATA / "labels").exists():
        shutil.copytree(DATA / "labels", dst / "labels")


def add(name, color=None):
    with _changing:
        return _add(name, color)


def merge(sources, target):
    with _changing:
        return _merge(sources, target)


def delete(gone):
    with _changing:
        return _delete(gone)


def _add(name, color):
    name = name.strip().lower()
    if not NAME.match(name):
        raise ValueError("A class name is lowercase letters, digits, - or _, starting with a letter or digit.")
    if name in names():
        raise ValueError(f"{name} already exists.")
    color = color or random_color()
    start(f"Add class {name}", {}, [("INSERT INTO class (name, color) VALUES (?,?)", (name, color)),
                                     ("DELETE FROM class_alias WHERE name=?", (name,))],   # the name is live again
          added=name, color=color)


def _merge(sources, target):
    """Every box of sources becomes target; the source names live on as aliases of target."""
    sources = [s for s in sources if s != target]
    live = names()
    if target not in live or not sources or any(s not in live for s in sources):
        raise ValueError("Pick two or more classes and which one they become.")
    q = ",".join("?" * len(sources))
    start(f"Merge {', '.join(sources)} into {target}", {s: target for s in sources},
                  [(f"DELETE FROM class WHERE name IN ({q})", tuple(sources)),
                   (f"UPDATE class_alias SET target=? WHERE target IN ({q})", (target, *sources))]
                  + [("INSERT OR REPLACE INTO class_alias VALUES (?,?)", (s, target)) for s in sources])


def _delete(gone):
    """Removes the classes and all their boxes; their names are remembered so imports drop them too."""
    live = names()
    if not gone or any(g not in live for g in gone):
        raise ValueError("Pick the classes to delete.")
    if set(gone) >= set(live):
        raise ValueError("Keep at least one class.")
    q = ",".join("?" * len(gone))
    start(f"Delete {', '.join(gone)}", {g: None for g in gone},
                  [(f"DELETE FROM class WHERE name IN ({q})", tuple(gone)),
                   (f"UPDATE class_alias SET target=NULL WHERE target IN ({q})", tuple(gone))]
                  + [("INSERT OR REPLACE INTO class_alias VALUES (?,NULL)", (g,)) for g in gone])


def recolor(name, color):
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", color or ""):
        raise ValueError("Colors are #rrggbb.")
    db.x("UPDATE class SET color=? WHERE name=?", (color, name))
    if pending:   # a class still being added has no row yet; its colour lands with the job's INSERT
        pending["colors"] = [(n, color if n == name else c) for n, c in pending["colors"]]
