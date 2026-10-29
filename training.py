"""Training runs. Kitman builds the YOLO dataset from corrected labels using the fixed
match split (never a percentage split), then trains in a separate process
(train_worker.py) that writes its own status, so a run survives a server restart."""

import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import db
from config import CLASSES, DATA, KITMAN, SAMPLING_DIR

# COCO-pretrained starting weights, as Ultralytics names them (v8 keeps the "v"; 11, 12 and 26 don't).
VERSIONS = {"26": "yolo26", "12": "yolo12", "11": "yolo11", "v8": "yolov8"}
SIZES = {"n": "nano", "s": "small", "m": "medium", "l": "large", "x": "extra large"}
DEFAULT_BASE = ("26", "m")


def base_weights(version, size):
    if version not in VERSIONS or size not in SIZES:
        raise ValueError("Pick a YOLO version and a size from the lists.")
    return f"{VERSIONS[version]}{size}.pt"


def parse_base(name):
    """'yolo26m.pt' -> ('26', 'm'); anything else -> the default."""
    for v, prefix in VERSIONS.items():
        if name.startswith(prefix) and name[len(prefix):] in {f"{s}.pt" for s in SIZES}:
            return v, name[len(prefix)]
    return DEFAULT_BASE
DEFAULTS = {"imgsz": 1280, "epochs": 100, "batch": 4, "patience": 25, "seed": 42}   # same seed + data = comparable runs

# Advanced settings: (group, key, kind, Ultralytics 8.4 default, football starting value, choices or hint).
# Only values that differ from the default are passed to Ultralytics; workers is always passed (Kitman's own default).
OPTIMIZERS = ["auto", "SGD", "Adam", "Adamax", "AdamW", "NAdam", "RAdam", "MuSGD"]
ADVANCED = [
    ("Optimiser and learning rate", "optimizer", "choice", "auto", "auto", OPTIMIZERS),
    ("Optimiser and learning rate", "lr0", "float", 0.01, 0.01, "ignored when optimizer is auto"),
    ("Optimiser and learning rate", "lrf", "float", 0.01, 0.01, "final rate, as a fraction of lr0"),
    ("Optimiser and learning rate", "momentum", "float", 0.937, 0.937, ""),
    ("Optimiser and learning rate", "weight_decay", "float", 0.0005, 0.0005, ""),
    ("Optimiser and learning rate", "warmup_epochs", "float", 3.0, 3.0, ""),
    ("Optimiser and learning rate", "cos_lr", "bool", False, False, "cosine schedule"),
    ("Augmentation", "fliplr", "float", 0.5, 0.5, "a mirrored pitch is still a pitch"),
    ("Augmentation", "flipud", "float", 0.0, 0.0, "the camera is never upside down"),
    ("Augmentation", "degrees", "float", 0.0, 0.0, "the camera doesn't tilt"),
    ("Augmentation", "shear", "float", 0.0, 0.0, ""),
    ("Augmentation", "perspective", "float", 0.0, 0.0, ""),
    ("Augmentation", "scale", "float", 0.5, 0.3, "the ball is ~13 px; heavy zoom-out loses it"),
    ("Augmentation", "translate", "float", 0.1, 0.1, ""),
    ("Augmentation", "hsv_h", "float", 0.015, 0.015, ""),
    ("Augmentation", "hsv_s", "float", 0.7, 0.5, "kit colours matter for teams later"),
    ("Augmentation", "hsv_v", "float", 0.4, 0.4, ""),
    ("Augmentation", "mosaic", "float", 1.0, 1.0, ""),
    ("Augmentation", "close_mosaic", "int", 10, 10, "mosaic off for the last N epochs"),
    ("Augmentation", "mixup", "float", 0.0, 0.0, ""),
    ("Augmentation", "cutmix", "float", 0.0, 0.0, ""),
    ("Augmentation", "copy_paste", "float", 0.0, 0.0, ""),
    ("Augmentation", "erasing", "float", 0.4, 0.4, ""),
    ("Run", "device", "text", "", "", "blank picks your GPU"),
    ("Run", "workers", "int", 2, 2, "Kitman's default; Windows is slow with more"),
    ("Run", "cache", "choice", "off", "off", ["off", "ram", "disk"]),
    ("Run", "amp", "bool", True, True, "mixed precision"),
    ("Run", "freeze", "text", "", "", "e.g. 10 = keep first 10 layers fixed"),
    ("Run", "rect", "bool", False, False, ""),
    ("Run", "dropout", "float", 0.0, 0.0, ""),
    ("Run", "multi_scale", "float", 0.0, 0.0, ""),
    ("Run", "save_period", "int", -1, -1, "save every N epochs; -1 = off"),
]
SPEC = {a[1]: a for a in ADVANCED}
FOOTBALL = {a[1]: a[4] for a in ADVANCED}
# Set by Kitman or by the basic fields; refused in Extra arguments so nothing overrides them silently.
RESERVED = {"data", "model", "project", "name", "exist_ok", "mode", "task", "resume", *DEFAULTS, *SPEC}


def cast(key, raw):
    """A form value as Ultralytics wants it; blank means the default. Raises ValueError on bad input."""
    _, _, kind, default, _, choices = SPEC[key]
    raw = (raw or "").strip() if kind != "bool" else raw
    if kind == "bool":
        return raw in ("on", "true", "True", True)
    if raw == "":
        return default
    if kind == "int":
        return int(raw)
    if kind == "float":
        return float(raw)
    if kind == "choice" and raw not in choices:
        raise ValueError(f"{key} must be one of {', '.join(choices)}")
    return raw


def to_ultralytics(key, value):
    """Kitman's form value -> the value Ultralytics expects (blank text -> None, cache 'off' -> False)."""
    if key in ("device", "freeze"):
        return None if value == "" else (int(value) if key == "freeze" and str(value).isdigit() else value)
    if key == "cache":
        return False if value == "off" else value
    return value


def parse_extra(text):
    """'key=value' lines -> dict, checked against Ultralytics' own settings list."""
    import yaml
    from ultralytics.cfg import DEFAULT_CFG_DICT
    out = {}
    for n, line in enumerate((text or "").splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"Extra arguments line {n}: write it as key=value")
        key, raw = (s.strip() for s in line.split("=", 1))
        if key not in DEFAULT_CFG_DICT:
            raise ValueError(f"Extra arguments line {n}: “{key}” isn't an Ultralytics setting")
        if key in RESERVED:
            raise ValueError(f"Extra arguments line {n}: set “{key}” with its own field above")
        out[key] = yaml.safe_load(raw) if raw else None
    return out


def settings_from_form(form):
    """Advanced fields (only those changed from the default, plus workers) and extra arguments."""
    adv = {}
    for key, spec in SPEC.items():
        try:
            v = cast(key, form.get(key))
        except ValueError as e:
            raise ValueError(f"{key}: {e}") from None
        if v != spec[3] or key == "workers":
            adv[key] = v
    return adv, parse_extra(form.get("extra", ""))


def labelled_frames(label_set_ids):
    """Latest label per frame among the chosen sets. Tracking-clip frames are left out:
    they are near-copies meant for tracking evaluation, not detector training."""
    if not label_set_ids:
        return []
    marks = ",".join("?" * len(label_set_ids))
    rows = db.q(f"""SELECT f.*, m.split, l.path AS label_path, l.label_set_id FROM label l
                    JOIN frame f ON f.id=l.frame_id JOIN match m ON m.id=f.match_id
                    WHERE l.label_set_id IN ({marks}) AND f.bucket != 'clip'
                    ORDER BY l.label_set_id""", label_set_ids)
    return list({r["id"]: r for r in rows}.values())


def link(src, dst):
    try:
        os.link(src, dst)          # same drive: no copy
    except OSError:
        shutil.copy2(src, dst)


def train_args(params):
    """Everything a run passes to model.train(), from its stored params."""
    args = {k: params[k] for k in DEFAULTS}
    args |= {k: to_ultralytics(k, v) for k, v in params.get("advanced", {}).items()}
    return args | params.get("extra", {})


def create(name, base_model, params, label_set_ids, advanced=None, extra=None, copy_to=""):
    params = DEFAULTS | {k: v for k, v in params.items() if v not in (None, "")}
    params |= {"advanced": advanced or {"workers": 2}, "extra": extra or {}, "copy_to": copy_to}
    frames = labelled_frames(label_set_ids)
    counts = {s: sum(f["split"] == s for f in frames) for s in ("train", "val", "test")}
    if not counts["train"]:
        raise ValueError(f"The chosen labels have no frames from train videos (have {counts['val']} val, "
                         f"{counts['test']} test). Pick a batch with train frames.")
    # No val frames: validate on the train frames. Scores are then optimistic, which the run page says.
    counts["val_from_train"] = not counts["val"]
    run_id = db.x("INSERT INTO training_run (name, base_model, params, label_set_ids, counts, status, created) "
                  "VALUES (?,?,?,?,?,?,?)",
                  (name, base_model, db.jdump(params), db.jdump(label_set_ids), db.jdump(counts), "queued", db.now()))
    run_dir = DATA / "runs" / f"run{run_id}"
    ds = run_dir / "dataset"
    for f in frames:
        for sub in ("images", "labels"):
            (ds / f["split"] / sub).mkdir(parents=True, exist_ok=True)
        link(SAMPLING_DIR / f["path"], ds / f["split"] / "images" / f["name"])
        shutil.copy2(DATA / f["label_path"], ds / f["split"] / "labels" / (f["name"].rsplit(".", 1)[0] + ".txt"))
    (ds / "data.yaml").write_text(
        f"path: {ds.as_posix()}\ntrain: train/images\nval: {'train' if counts['val_from_train'] else 'val'}/images\n"
        + ("test: test/images\n" if counts["test"] else "")
        + "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)), encoding="utf-8")
    with open(run_dir / "frames.csv", "w", newline="", encoding="utf-8") as fh:   # exact lineage of this run
        w = csv.writer(fh)
        w.writerow(["frame", "match", "split", "t", "label_set"])
        w.writerows([f["name"], f["match_id"], f["split"], f["t"], f["label_set_id"]] for f in frames)
    db.x("UPDATE training_run SET run_dir=? WHERE id=?", (str(run_dir), run_id))
    spawn(run_id, run_dir, "w")
    return run_id


def spawn(run_id, run_dir, log_mode, *flags):
    group = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    with open(Path(run_dir) / "train.log", log_mode, encoding="utf-8") as log:   # the worker keeps its own handle
        proc = subprocess.Popen([sys.executable, str(KITMAN / "train_worker.py"), str(run_id), *flags], cwd=KITMAN,
                                stdout=log, stderr=subprocess.STDOUT, creationflags=group)
    db.x("UPDATE training_run SET pid=? WHERE id=?", (getattr(proc, "pid", None), run_id))


def worker(run):
    """The run's training process if it is still alive, else None."""
    import psutil
    if not run["pid"]:
        return None
    try:
        proc = psutil.Process(run["pid"])
        # only ever Kitman's own worker for this run, never a reused process id
        if any("train_worker.py" in a for a in proc.cmdline()) and str(run["id"]) in proc.cmdline():
            return proc
    except psutil.Error:   # gone, or a reused id now owned by a system process we can't read
        pass
    return None


def last_weights(run):
    return Path(run["run_dir"] or "") / "train" / "weights" / "last.pt"


def resumable(run):
    """Paused, or crashed (failed, or 'running' with its process gone, e.g. the PC slept), and has a checkpoint."""
    crashed = run["status"] in ("failed", "paused") or (run["status"] in ("queued", "running") and not worker(run))
    return crashed and run["run_dir"] is not None and last_weights(run).exists()


def resume(run_id):
    """Carry on a crashed run from its last.pt, with the settings saved in it, in the same folder and log."""
    run = db.q1("SELECT * FROM training_run WHERE id=?", (run_id,))
    if not run or not resumable(run):
        raise ValueError(f"Run {run_id} can't be resumed: only a paused or crashed run with a saved last.pt can.")
    db.x("UPDATE training_run SET status='queued', finished=NULL WHERE id=?", (run_id,))
    spawn(run_id, run["run_dir"], "a", "--resume")


def pause(run_id):
    """End a run's training process but keep everything, to resume later from its last finished epoch."""
    run = db.q1("SELECT * FROM training_run WHERE id=?", (run_id,))
    if not run or run["status"] not in ("queued", "running"):
        raise ValueError(f"Run {run_id} isn't training, so there is nothing to pause.")
    kill(run)
    with open(Path(run["run_dir"]) / "train.log", "a", encoding="utf-8") as fh:   # the killed worker can't say it
        fh.write("\nKITMAN: paused; resume carries on from the last finished epoch\n")
    db.x("UPDATE training_run SET status='paused', finished=? WHERE id=?", (db.now(), run_id))


def stop(run_id):
    """Stop a run and leave no trace: end its training process (and data-loader helpers),
    then delete the run's folder and its database rows, as if it was never started."""
    run = db.q1("SELECT * FROM training_run WHERE id=?", (run_id,))
    if not run:
        raise ValueError(f"Run {run_id} doesn't exist.")
    if run["status"] == "done":
        raise ValueError("This run has finished; finished runs are kept.")
    kill(run)
    if run["run_dir"] and os.path.isdir(run["run_dir"]):
        import time
        for attempt in range(5):   # Windows may hold the log or weights for a moment after the process dies
            try:
                shutil.rmtree(run["run_dir"])
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(1)
    model_ids = [m["id"] for m in db.q("SELECT id FROM model WHERE training_run_id=?", (run_id,))]
    for mid in model_ids:   # a run stopped while finishing may have registered its model already
        db.x("UPDATE label_set SET model_id=NULL WHERE model_id=?", (mid,))
        db.x("DELETE FROM model WHERE id=?", (mid,))
    db.x("DELETE FROM training_run WHERE id=?", (run_id,))


def kill(run):
    """End the run's training process and its data-loader helpers."""
    import psutil
    proc = worker(run)
    tree = []
    if proc:
        try:
            tree = proc.children(recursive=True) + [proc]
        except psutil.Error:
            pass
    if run["pid"]:
        # data-loader helpers, found by the parent id in their command line: they outlive a killed worker
        # (then children() no longer sees them) and hold train.log open
        for p in psutil.process_iter():
            try:
                if f"parent_pid={run['pid']}" in " ".join(p.cmdline()):
                    tree.append(p)
            except psutil.Error:
                pass
    for p in tree:
        try:
            p.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(tree, timeout=15)


def epochs(run):
    """Per-epoch rows from Ultralytics' results.csv (live while training)."""
    if not run["run_dir"]:
        return []
    p = os.path.join(run["run_dir"], "train", "results.csv")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as fh:
        rows = [{k.strip(): float(v) for k, v in r.items() if v.strip()} for r in csv.DictReader(fh)]
    # A loss can go nan; NaN isn't valid JSON, so the browser would drop the whole update.
    return [{k: v for k, v in r.items() if math.isfinite(v)} for r in rows]


def log_tail(run, n=60):
    p = os.path.join(run["run_dir"] or "", "train.log")
    if not os.path.exists(p):
        return ""
    with open(p, encoding="utf-8", errors="replace") as fh:
        text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", fh.read()).replace("\r", "\n")   # colours; bars redraw with \r
    return "\n".join([ln for ln in text.splitlines() if ln.strip()][-n:])


def metrics_of(results, names):
    """Ultralytics DetMetrics -> plain dict, overall and per class."""
    box = results.box
    per = {}
    for k, ci in enumerate(box.ap_class_index):
        per[names[int(ci)]] = {"precision": float(box.p[k]), "recall": float(box.r[k]),
                               "map50": float(box.ap50[k]), "map50_95": float(box.ap[k])}
    return {"overall": {"precision": float(box.mp), "recall": float(box.mr),
                        "map50": float(box.map50), "map50_95": float(box.map)}, "per_class": per}


def evaluate_test(job, run_id):
    from ultralytics import YOLO
    run = db.q1("SELECT * FROM training_run WHERE id=?", (run_id,))
    if not json.loads(run["counts"])["test"]:
        raise ValueError("This run has no labelled test frames. Label the test matches first.")
    params = json.loads(run["params"])
    model = YOLO(os.path.join(run["run_dir"], "train", "weights", "best.pt"))
    job.progress(0.2, "Evaluating on test matches")
    res = model.val(data=os.path.join(run["run_dir"], "dataset", "data.yaml"), split="test",
                    imgsz=params["imgsz"], batch=params["batch"], project=run["run_dir"], name="test",
                    exist_ok=True, verbose=False)
    m = metrics_of(res, model.names)
    db.x("UPDATE training_run SET test_metrics=? WHERE id=?", (db.jdump(m), run_id))
    return f"Test mAP50 {m['overall']['map50']:.3f}"
