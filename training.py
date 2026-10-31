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
import threading
import time
import zipfile
from pathlib import Path

import db
import classes
from config import DATA, KITMAN, SAMPLING_DIR

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


def data_yaml(counts, root=None):
    """root: the absolute dataset folder, for Kitman's own runs. None (exported zip): Ultralytics then
    resolves train/val/test next to data.yaml, wherever the zip is unpacked."""
    return ((f"path: {root.as_posix()}\n" if root else "")
            + f"train: train/images\nval: {'train' if counts['val_from_train'] else 'val'}/images\n"
            + ("test: test/images\n" if counts["test"] else "")
            + "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(counts["classes"])))


def create(name, base_model, params, label_set_ids, advanced=None, extra=None, copy_to="", export=False):
    """export: build the dataset but don't train here; the run waits as 'exported' for a model trained elsewhere."""
    if classes.pending:   # its label files would be copied half renumbered
        raise ValueError("Labels are being renumbered after a class change. Start the run in a few seconds.")
    params = DEFAULTS | {k: v for k, v in params.items() if v not in (None, "")}
    params |= {"advanced": advanced or {"workers": 2}, "extra": extra or {}, "copy_to": copy_to}
    frames = labelled_frames(label_set_ids)
    counts = {s: sum(f["split"] == s for f in frames) for s in ("train", "val", "test")}
    if not counts["train"]:
        raise ValueError(f"The chosen labels have no frames from train videos (have {counts['val']} val, "
                         f"{counts['test']} test). Pick a batch with train frames.")
    # No val frames: validate on the train frames. Scores are then optimistic, which the run page says.
    counts["val_from_train"] = not counts["val"]
    counts["classes"] = classes.names()   # what the copied label files' ids mean, whatever the Classes page does later
    run_id = db.x("INSERT INTO training_run (name, base_model, params, label_set_ids, counts, status, created) "
                  "VALUES (?,?,?,?,?,?,?)",
                  (name, base_model, db.jdump(params), db.jdump(label_set_ids), db.jdump(counts),
                   "exported" if export else "queued", db.now()))
    run_dir = DATA / "runs" / f"run{run_id}"
    ds = run_dir / "dataset"
    for f in frames:
        for sub in ("images", "labels"):
            (ds / f["split"] / sub).mkdir(parents=True, exist_ok=True)
        link(SAMPLING_DIR / f["path"], ds / f["split"] / "images" / f["name"])
        shutil.copy2(DATA / f["label_path"], ds / f["split"] / "labels" / (f["name"].rsplit(".", 1)[0] + ".txt"))
    (ds / "data.yaml").write_text(data_yaml(counts, ds), encoding="utf-8")
    with open(run_dir / "frames.csv", "w", newline="", encoding="utf-8") as fh:   # exact lineage of this run
        w = csv.writer(fh)
        w.writerow(["frame", "match", "split", "t", "label_set"])
        w.writerows([f["name"], f["match_id"], f["split"], f["t"], f["label_set_id"]] for f in frames)
    db.x("UPDATE training_run SET run_dir=? WHERE id=?", (str(run_dir), run_id))
    if not export:
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
                    imgsz=params["imgsz"], batch=val_batch(params), project=run["run_dir"], name="test",
                    exist_ok=True, verbose=False, workers=0)   # see val_batch: loaders would re-import app.py
    m = metrics_of(res, model.names)
    db.x("UPDATE training_run SET test_metrics=? WHERE id=?", (db.jdump(m), run_id))
    return f"Test mAP50 {m['overall']['map50']:.3f}"


def val_batch(params):
    """A run trained elsewhere may have planned a batch only a bigger GPU holds; scoring here never needs more.
    Scoring runs inside the server, so it also uses workers=0: on Windows each data-loader process re-imports
    app.py, whose db.init() marks running jobs failed, and the scoring hangs."""
    return min(params["batch"], DEFAULTS["batch"])


# ---------------------------------------------------------------- trained elsewhere

def export_path(run):
    return Path(run["run_dir"]) / f"kitman_run{run['id']}.zip"


def train_command(run):
    """The yolo command that trains an exported run as Kitman would, run next to its data.yaml."""
    args = {k: v for k, v in train_args(json.loads(run["params"])).items() if v is not None}
    return (f"yolo train data=data.yaml model={Path(run['base_model']).name} project=runs name=train "
            + " ".join(f"{k}={v if isinstance(v, str) else json.dumps(v, separators=(',', ':'))}" for k, v in args.items()))


def export_zip(job, run_id):
    """kitman_run<N>.zip: the run's dataset in YOLO layout, a data.yaml that works wherever it is unpacked,
    frames.csv and kitman_run.json (settings, lineage and the command to train it)."""
    run = db.q1("SELECT * FROM training_run WHERE id=?", (run_id,))
    ds, out = Path(run["run_dir"]) / "dataset", export_path(run)
    files = [f for f in ds.rglob("*") if f.is_file() and f.name != "data.yaml" and f.suffix != ".cache"]   # Ultralytics' label caches hold this PC's paths
    meta = {"kitman_run": run_id, "name": run["name"], "base_model": Path(run["base_model"]).name,
            "params": json.loads(run["params"]), "label_set_ids": json.loads(run["label_set_ids"]),
            "counts": json.loads(run["counts"]), "classes": json.loads(run["counts"])["classes"], "command": train_command(run),
            "bring_back": "Zip the runs/train folder (or just runs/train/weights/best.pt) and import it on "
                          f"Kitman's page for run {run_id}."}
    part = out.with_suffix(".part")   # a half-written zip never looks ready
    with zipfile.ZipFile(part, "w", zipfile.ZIP_STORED) as z:   # JPEGs don't compress further
        z.writestr("data.yaml", data_yaml(meta["counts"]))
        z.writestr("kitman_run.json", json.dumps(meta, indent=2))
        z.write(Path(run["run_dir"]) / "frames.csv", "frames.csv")
        for k, f in enumerate(files):
            z.write(f, f.relative_to(ds).as_posix())
            if job and k % 20 == 0:
                job.progress((k + 1) / len(files), f"Packed {k + 1} of {len(files)} files")
    part.replace(out)
    return f"Zip ready: {out.name}, {out.stat().st_size / 2**20:.0f} MB"


ZIP_HOURS = 12
_zip_lock = threading.Lock()   # two downloads at once share one build


def dataset_zip(run):
    """The run's dataset zip: built on the first download, then the same file for ZIP_HOURS, after which
    sweep_zips deletes it (a later download builds it again; the dataset never changes)."""
    with _zip_lock:
        p = export_path(run)
        if not (p.exists() and time.time() - p.stat().st_mtime < ZIP_HOURS * 3600):
            export_zip(None, run["id"])
        return p


def sweep_zips(root=None):
    """Delete dataset zips older than ZIP_HOURS."""
    with _zip_lock:
        for p in Path(root or DATA / "runs").glob("run*/kitman_run*.zip"):
            if time.time() - p.stat().st_mtime >= ZIP_HOURS * 3600:
                try:
                    p.unlink()
                except OSError:   # still being downloaded (Windows holds it open); the next sweep gets it
                    pass


SAFE_NAME = re.compile(r"[\w\- ][\w.\- ]*")


def pick_train_files(names):
    """'/'-separated paths (zip members, or files under a chosen folder) -> {path: name inside run/train}.
    The training folder is the one holding weights/best.pt, and there must be exactly one. Taken from it:
    best/last weights and its own top-level files (results.csv, args.yaml, plots); nothing that could
    land outside run/train."""
    bests = [n for n in names if n == "weights/best.pt" or n.endswith("/weights/best.pt")]
    if not bests:
        raise ValueError("No weights/best.pt in it. Choose Ultralytics' training folder (runs/detect/train), "
                         "or a zip of it.")
    if len(bests) > 1:
        raise ValueError(f"It holds {len(bests)} trainings ({', '.join(b[:-16] or '.' for b in bests)}). "
                         "Choose or zip just one.")
    root = bests[0][:-len("weights/best.pt")]
    out = {}
    for n in names:
        rel = n[len(root):] if n.startswith(root) else None
        if rel in ("weights/best.pt", "weights/last.pt") or (
                rel and SAFE_NAME.fullmatch(rel) and rel.lower().endswith((".csv", ".yaml", ".png", ".jpg"))):
            out[n] = rel
    return out


IMPORTED_FROM = {"pt": "best.pt only", "zip": "a training folder zip", "folder": "a training folder"}


def import_model(job, run_id, source, kind):
    """Bring back a model trained elsewhere for an exported run, then score it here on the run's own
    dataset so its numbers compare with runs trained here. kind: pt or zip (an uploaded file, removed
    afterwards) or folder (a folder on this computer, left as it is)."""
    from ultralytics import YOLO
    run = db.q1("SELECT * FROM training_run WHERE id=?", (run_id,))
    if not run or run["status"] != "exported":   # e.g. imported twice before the first finished
        raise ValueError(f"Run {run_id} isn't waiting for a model.")
    rd = Path(run["run_dir"])
    stage = rd / "import"
    shutil.rmtree(stage, ignore_errors=True)
    (stage / "weights").mkdir(parents=True)
    try:
        if kind == "pt":
            shutil.copy2(source, stage / "weights" / "best.pt")
        elif kind == "zip":
            with zipfile.ZipFile(source) as z:   # read members as data, never extract paths
                for src, rel in pick_train_files(z.namelist()).items():
                    (stage / rel).write_bytes(z.read(src))
        else:
            root = Path(source)
            names = [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()]
            for src, rel in pick_train_files(names).items():
                shutil.copy2(root / src, stage / rel)
        job.progress(0.1, "Checking the model's classes")
        names = list(YOLO(str(stage / "weights" / "best.pt")).names.values())
        want = json.loads(run["counts"])["classes"]
        if names != want:
            raise ValueError(f"The model's classes ({', '.join(names[:8])}) aren't this run's "
                             f"({', '.join(want)}) in that order, so it wasn't trained on this export.")
        shutil.rmtree(rd / "train", ignore_errors=True)   # left by an earlier import that failed while scoring
        stage.rename(rd / "train")
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        if kind != "folder":
            Path(source).unlink(missing_ok=True)

    params = json.loads(run["params"])
    # The run now shows, scores at and copies what it was really trained with (its args.yaml); the plan is kept.
    base = run["base_model"]
    planned = {"model": Path(base).name, **train_args(params)}
    args_file = rd / "train" / "args.yaml"
    if args_file.is_file():
        import yaml
        used = yaml.safe_load(args_file.read_text(encoding="utf-8")) or {}
        params |= {k: used[k] for k in DEFAULTS if used.get(k) is not None}
        base = Path(str(used.get("model") or base)).name
    job.progress(0.2, "Scoring on this run's val frames")
    model = YOLO(str(rd / "train" / "weights" / "best.pt"))
    res = model.val(data=str(rd / "dataset" / "data.yaml"), split="val", imgsz=params["imgsz"],
                    batch=val_batch(params), project=str(rd), name="val", exist_ok=True, verbose=False, workers=0)
    m = metrics_of(res, model.names)
    rows = epochs(run)
    best = max(rows, key=lambda r: r.get("metrics/mAP50-95(B)", 0)) if rows else None   # best.pt's epoch
    params["imported"] = {"from": IMPORTED_FROM[kind], "at": db.now(), "planned": planned, "remote": best and {
        "epoch": int(best.get("epoch", 0)), "precision": best.get("metrics/precision(B)"),
        "recall": best.get("metrics/recall(B)"), "map50": best.get("metrics/mAP50(B)"),
        "map50_95": best.get("metrics/mAP50-95(B)")}}
    db.x("UPDATE training_run SET status='done', metrics=?, params=?, base_model=?, finished=? WHERE id=?",
         (db.jdump(m), db.jdump(params), base, db.now(), run_id))
    db.x("INSERT INTO model (name, path, kind, training_run_id, created) VALUES (?,?,?,?,?)",
         (f"Run {run_id}: {run['name']}", str(rd / "train" / "weights" / "best.pt"), "kitman", run_id, db.now()))
    return f"Run {run_id} imported: val mAP50 {m['overall']['map50']:.3f}, scored here"
