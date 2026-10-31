"""Self-checks for Kitman's pure logic: python kitman/test_kitman.py"""

import random

import numpy as np
import pandas as pd

from app import spread
from labels import class_map, frame_stem, parse
from sampling import CUSTOM_DEFAULTS, plan_custom, plan_fixed, snap

# Roboflow renaming is undone
assert frame_stem("train/labels/M05_frame_003704_jpg.rf.0a1b2c3d4e.txt") == "M05_frame_003704"
assert frame_stem("M04clip_frame_000120.txt") == "M04clip_frame_000120"

# YOLO label parsing rejects malformed lines
assert parse("0 0.5 0.5 0.1 0.2\n\n3 0.1 0.1 0.01 0.01\n") == [(0, .5, .5, .1, .2), (3, .1, .1, .01, .01)]
for bad in ("0 0.5 0.5 0.1", "0 1.5 0.5 0.1 0.1"):
    try:
        parse(bad)
        raise AssertionError(bad)
    except ValueError:
        pass

# Uploaded models: read by our class names, else as COCO, else refused
LIVE = ["ball", "goalkeeper", "player", "referee"]   # Roboflow's alphabetical ids
assert class_map({0: "player", 1: "ball"}, LIVE, {}) == {0: 2, 1: 0}
assert class_map({0: "person", 32: "sports ball"}, LIVE, {}) is None
try:
    class_map({0: "car"}, LIVE, {})
    raise AssertionError("car-only model accepted")
except ValueError:
    pass
# after merging goalkeeper into player and deleting referee: old names still read, deleted ones drop
import classes
after = ["ball", "player"]
alias = {"goalkeeper": "player", "referee": None}
assert classes.read_names({0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}, after, alias) == ({0: 0, 1: 1, 2: 1, 3: None}, [])
assert classes.read_names({0: "car"}, after, alias) == ({}, ["car"])
assert class_map({0: "goalkeeper", 1: "referee"}, after, alias) == {0: 1}
# a Roboflow project holding only player and referee numbers them 0, 1; they still land on 2, 3
assert classes.read_names({0: "player", 1: "referee"}, LIVE, {})[0] == {0: 2, 1: 3}
assert classes.renumber(LIVE, after, {"ball": "ball", "goalkeeper": "player", "player": "player", "referee": None}) == {0: 0, 1: 1, 2: 1}
assert classes.renumber(LIVE, sorted(LIVE + ["coach"]), {n: n for n in LIVE}) == {0: 0, 1: 2, 2: 3, 3: 4}

# Snapping to real frames on a variable-rate clock, with the freeze limit
pts = np.array([0.0, 0.03, 0.07, 0.10, 1.40, 1.43])          # freeze between 0.10 and 1.40
assert snap([0.06, 1.0], pts) == [2, 4]
assert snap([0.06, 0.8], pts, 0.25) == [2, None]

# Fixed interval: every N s by timestamp, nothing dropped, no repeats inside a freeze
fx = plan_fixed(pts, 1.5, 0.5)
assert [p["frame_idx"] for p in fx] == [0, 3, 4]               # 0.5 and 1.0 both snap to 0.10 / 1.40

# Custom plan: quota buckets on a synthetic all-live match
t = np.arange(0, 300, 0.5)
rng = np.random.default_rng(0)
samples = pd.DataFrame({"t": t, "state": "live", "flow_mag": rng.uniform(0, 30, len(t)),
                        "near_dup": 0, "scene_cut": 0,
                        "n_ball": np.where(np.arange(len(t)) % 2 == 0, rng.integers(0, 2, len(t)), np.nan)})
dets = pd.DataFrame(columns=["t", "cx", "cy"])
picks, stats = plan_custom("train", samples, dets, np.arange(0, 300, 1 / 30), CUSTOM_DEFAULTS, random.Random(1))
times = [p["t"] for p in picks]
assert stats["windows"] == 60 and len(picks) == 60
assert min(np.diff(times)) >= CUSTOM_DEFAULTS["min_gap_s"]
assert {p["bucket"] for p in picks} <= {"random", "fast_pan", "ball_missed", "edge"}
assert sum(p["bucket"] == "fast_pan" for p in picks) >= 10      # 20% of 60, minus any fallbacks
# Val/test: uniform windows plus a tracking clip
picks, stats = plan_custom("val", samples, dets, None, CUSTOM_DEFAULTS, random.Random(1))
assert len(picks) == 50 and stats["clip"]["end"] - stats["clip"]["start"] == 10

# Non-live frames: about one per live frame, split replay/menu, >= 1 s apart, shortfall reported
mixed = samples.copy()
mixed.loc[mixed.t >= 200, "state"] = "replay_or_cutscene"        # 100 s of replays
mixed.loc[mixed.t >= 270, "state"] = "other"                     # 30 s of menus
picks, stats = plan_custom("train", mixed, dets, np.arange(0, 300, 1 / 30), CUSTOM_DEFAULTS, random.Random(1))
live_n = sum(p["bucket"] not in ("replay", "menu") for p in picks)
nl = sorted(p["t"] for p in picks if p["bucket"] in ("replay", "menu"))
assert stats["nonlive"]["target"] == live_n and len(nl) == live_n - stats["nonlive"]["shortfall"]
assert stats["nonlive"]["replay"] > stats["nonlive"]["menu"] > 0
assert min(np.diff(nl)) >= CUSTOM_DEFAULTS["nonlive_gap_s"]
picks, stats = plan_custom("train", mixed, dets, np.arange(0, 300, 1 / 30),
                           CUSTOM_DEFAULTS | {"nonlive_ratio": 0}, random.Random(1))
assert not [p for p in picks if p["bucket"] in ("replay", "menu")]

# Extraction: two picks on the same frame (a live pick inside the tracking clip) both get their file
import tempfile
from pathlib import Path
import cv2
import sampling
with tempfile.TemporaryDirectory() as tmp:
    vid = Path(tmp) / "v.avi"
    w = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 32))
    for i in range(10):
        w.write(np.full((32, 64, 3), i * 20, np.uint8))
    w.release()
    real_dir, sampling.SAMPLING_DIR = sampling.SAMPLING_DIR, Path(tmp)
    try:
        sampling.extract(vid, [{"frame_idx": 4, "path": "X/a.jpg"}, {"frame_idx": 4, "path": "X/b.jpg"},
                               {"frame_idx": 7, "path": "X/c.jpg"}])
        assert all((Path(tmp) / "X" / n).is_file() for n in ("a.jpg", "b.jpg", "c.jpg"))
    finally:
        sampling.SAMPLING_DIR = real_dir

# Training settings: only changed values are passed; extras are checked against Ultralytics
import training
assert training.base_weights("v8", "x") == "yolov8x.pt" and training.base_weights("26", "m") == "yolo26m.pt"
assert training.parse_base("yolo12l.pt") == ("12", "l") and training.parse_base("best.pt") == training.DEFAULT_BASE
try:
    training.base_weights("27", "m")
    raise AssertionError("accepted YOLO27")
except ValueError:
    pass
untouched = {k: ("on" if s[3] is True else "") if s[2] == "bool" else str(s[3]) for k, s in training.SPEC.items()}
adv, extra = training.settings_from_form(untouched)
assert adv == {"workers": 2} and extra == {}
fb = dict(untouched, scale="0.3", hsv_s="0.5", amp="", optimizer="AdamW", cache="ram",
          extra="fraction=0.5\n# a comment\nsingle_cls=False")
adv, extra = training.settings_from_form(fb)
assert adv == {"scale": 0.3, "hsv_s": 0.5, "amp": False, "optimizer": "AdamW", "cache": "ram", "workers": 2}
assert extra == {"fraction": 0.5, "single_cls": False}
args = training.train_args({**training.DEFAULTS, "advanced": adv, "extra": extra})
assert args["scale"] == 0.3 and args["amp"] is False and args["epochs"] == 100 and args["fraction"] == 0.5
assert training.train_args({**training.DEFAULTS, "advanced": {"cache": "off", "device": "", "freeze": "10"}}) \
    | {} == {**training.DEFAULTS, "cache": False, "device": None, "freeze": 10}
for bad in ("lr0=0.1", "epochz=5", "data=x.yaml", "fraction 0.5"):
    try:
        training.parse_extra(bad)
        raise AssertionError(f"accepted {bad!r}")
    except ValueError:
        pass
try:
    training.settings_from_form(dict(untouched, optimizer="Lion"))
    raise AssertionError("accepted unknown optimizer")
except ValueError:
    pass

# Stop: ends the run's process, deletes its folder and row; finished runs are refused
import subprocess
import sys as _sys
import db

with tempfile.TemporaryDirectory() as tmp:
    real_db, db.DB_PATH = db.DB_PATH, Path(tmp) / "t.db"
    try:
        db.init()
        run_dir = Path(tmp) / "run9"
        (run_dir / "train").mkdir(parents=True)
        (run_dir / "train.log").write_text("x")
        # a stand-in worker: its command line looks like Kitman's training process for run 9
        fake = subprocess.Popen([_sys.executable, "-c", "import time; time.sleep(60)", "train_worker.py", "9"])
        # an orphaned data-loader helper whose worker already died: found by parent_pid=<run pid>
        orphan = subprocess.Popen([_sys.executable, "-c", "import time; time.sleep(60)", "parent_pid=424242"])
        db.x("INSERT INTO training_run (id, name, base_model, params, label_set_ids, status, run_dir, created, pid) "
             "VALUES (9,'t','yolo26n.pt','{}','[]','running',?,?,?)", (str(run_dir), db.now(), fake.pid))
        training.stop(9)
        assert fake.wait(timeout=10) is not None   # the stand-in training process was ended
        db.x("INSERT INTO training_run (id, name, base_model, params, label_set_ids, status, run_dir, created, pid) "
             "VALUES (10,'t','yolo26n.pt','{}','[]','failed',NULL,?,424242)", (db.now(),))
        training.stop(10)
        assert orphan.wait(timeout=10) is not None   # the orphan helper was ended too
        assert not run_dir.exists() and not db.q1("SELECT 1 FROM training_run WHERE id=9")
        db.x("INSERT INTO training_run (id, name, base_model, params, label_set_ids, status, created) "
             "VALUES (8,'t','yolo26n.pt','{}','[]','done',?)", (db.now(),))
        try:
            training.stop(8)
            raise AssertionError("removed a finished run")
        except ValueError:
            pass

        # Resume: only crashed runs (failed, or 'running' with no live worker) that saved a last.pt
        (run_dir / "train" / "weights").mkdir(parents=True)
        (run_dir / "train" / "weights" / "last.pt").write_text("x")
        alive = subprocess.Popen([_sys.executable, "-c", "import time; time.sleep(60)", "train_worker.py", "7"])
        for rid, status, pid in ((5, "failed", None), (6, "running", 999999), (7, "running", alive.pid), (4, "done", None)):
            db.x("INSERT INTO training_run (id, name, base_model, params, label_set_ids, status, run_dir, created, pid) "
                 "VALUES (?,'t','yolo26n.pt','{}','[]',?,?,?,?)", (rid, status, str(run_dir), db.now(), pid))
        got = {r["id"] for r in db.q("SELECT * FROM training_run") if training.resumable(r)}
        alive.kill()
        assert got == {5, 6}, got
        spawned, real_spawn = [], training.spawn
        training.spawn = lambda *a: spawned.append(a)
        try:
            training.resume(5)
            assert db.q1("SELECT status FROM training_run WHERE id=5")["status"] == "queued"
            assert spawned == [(5, str(run_dir), "a", "--resume")]
            try:
                training.resume(4)
                raise AssertionError("resumed a finished run")
            except ValueError:
                pass
        finally:
            training.spawn = real_spawn

        # Pause: ends the worker, keeps the folder and checkpoint, and the run becomes resumable
        worker11 = subprocess.Popen([_sys.executable, "-c", "import time; time.sleep(60)", "train_worker.py", "11"])
        db.x("INSERT INTO training_run (id, name, base_model, params, label_set_ids, status, run_dir, created, pid) "
             "VALUES (11,'t','yolo26n.pt','{}','[]','running',?,?,?)", (str(run_dir), db.now(), worker11.pid))
        assert not training.resumable(db.q1("SELECT * FROM training_run WHERE id=11"))   # still training
        training.pause(11)
        assert worker11.wait(timeout=10) is not None
        run = db.q1("SELECT * FROM training_run WHERE id=11")
        assert run["status"] == "paused" and training.resumable(run) and (run_dir / "train" / "weights" / "last.pt").exists()
        assert "paused" in (run_dir / "train.log").read_text()
        try:
            training.pause(11)
            raise AssertionError("paused a run that isn't training")
        except ValueError:
            pass

        # Export: a zip that trains anywhere (data.yaml with no absolute path), with lineage and the command
        import json
        import zipfile
        exp = Path(tmp) / "run12"
        for sub in ("train/images", "train/labels"):
            (exp / "dataset" / sub).mkdir(parents=True)
        (exp / "dataset" / "train/images/a.jpg").write_bytes(b"jpg")
        (exp / "dataset" / "train/labels/a.txt").write_text("0 0.5 0.5 0.1 0.1\n")
        (exp / "dataset" / "data.yaml").write_text("path: C:/somewhere\n")
        (exp / "frames.csv").write_text("frame\na.jpg\n")
        params = training.DEFAULTS | {"advanced": {"workers": 2, "device": ""}, "extra": {"fraction": 0.5}}
        db.x("INSERT INTO training_run (id, name, base_model, params, label_set_ids, counts, status, run_dir, created) "
             "VALUES (12,'t','yolo26m.pt',?,'[3]',?,'exported',?,?)",
             (db.jdump(params), db.jdump({"train": 1, "val": 0, "test": 0, "val_from_train": True, "classes": ["ball", "player"]}), str(exp), db.now()))
        training.export_zip(type("J", (), {"progress": lambda *a: None})(), 12)
        with zipfile.ZipFile(exp / "kitman_run12.zip") as z:
            assert set(z.namelist()) == {"data.yaml", "kitman_run.json", "frames.csv", "train/images/a.jpg", "train/labels/a.txt"}
            y = z.read("data.yaml").decode()
            assert "path:" not in y and "val: train/images" in y and "test:" not in y and "0: ball\n  1: player" in y   # the run's own classes
            cmd = json.loads(z.read("kitman_run.json"))["command"]
        assert cmd.startswith("yolo train data=data.yaml model=yolo26m.pt project=runs name=train imgsz=1280")
        assert "fraction=0.5" in cmd and "device" not in cmd   # blank device: let the remote machine pick
        assert training.resumable(db.q1("SELECT * FROM training_run WHERE id=12")) is False

        # Download zip: the same file within 12 hours, a fresh one after, and the sweep deletes old ones
        import os
        import time
        run12 = db.q1("SELECT * FROM training_run WHERE id=12")
        zp = training.dataset_zip(run12)
        old = time.time() - 11 * 3600
        os.utime(zp, (old, old))
        assert training.dataset_zip(run12).stat().st_mtime == old   # reused, not rebuilt
        old = time.time() - 13 * 3600
        os.utime(zp, (old, old))
        assert training.dataset_zip(run12).stat().st_mtime > old    # expired: rebuilt
        training.sweep_zips(tmp)
        assert zp.exists()                                          # fresh zips stay
        os.utime(zp, (old, old))
        training.sweep_zips(tmp)
        assert not zp.exists()
    finally:
        db.DB_PATH = real_db

# Import: the one training folder holding weights/best.pt, and nothing that could land outside run/train
pick = training.pick_train_files
assert pick(["weights/best.pt", "weights/last.pt", "results.csv", "args.yaml", "events.out"]) == {
    "weights/best.pt": "weights/best.pt", "weights/last.pt": "weights/last.pt",
    "results.csv": "results.csv", "args.yaml": "args.yaml"}
got = pick(["runs/detect/train/weights/best.pt", "runs/detect/train/results.csv", "runs/detect/train/val/x.png",
            "runs/detect/train/..\\..\\evil.csv", "other/results.csv", "runs/detect/train/weights/epoch5.pt"])
assert got == {"runs/detect/train/weights/best.pt": "weights/best.pt", "runs/detect/train/results.csv": "results.csv"}, got
for bad in (["results.csv"], ["a/weights/best.pt", "b/weights/best.pt"]):
    try:
        pick(bad)
        raise AssertionError(f"accepted {bad}")
    except ValueError:
        pass

# Batches spread round-robin across matches
c = [{"id": i, "match_id": m} for i, m in enumerate(["M01"] * 10 + ["M02"] * 10 + ["M03"] * 2)]
got = spread(c, 7, seed=0)
assert len(got) == 7 and {g["match_id"] for g in got} == {"M01", "M02", "M03"}

# Parallel Roboflow upload sends every frame once and surfaces a failure
import time

import requests

import roboflow_io


class FakeProject:
    def __init__(self, fail=None, conflict=None, drops=0):
        self.sent, self.fail, self.conflict, self.drops, self.ann = [], fail, conflict, drops, []

    def single_upload(self, image_path, **kw):
        time.sleep(0.02)   # like a network call, so cancellation has something to cancel
        if image_path.endswith(str(self.fail)):
            raise IOError("upload refused")
        if image_path.endswith("f5") and self.drops:
            self.drops -= 1
            try:
                raise requests.ConnectionError("Connection aborted.")
            except requests.ConnectionError as e:
                raise RuntimeError("AnnotationSaveError") from e   # how the SDK wraps a dropped connection
        self.sent.append(image_path)
        self.ann.append(kw.get("annotation_path"))
        if image_path.endswith(str(self.conflict)):
            raise AttributeError("'str' object has no attribute 'get'")   # how roboflow 1.2.11 reports a 409


class FakeJob:
    def progress(self, frac, message=None):
        self.last = frac


fake_frames = [{"id": i, "path": f"f{i}", "split": "train", "match_id": "M01"} for i in range(30)]
real = roboflow_io.db.q1, roboflow_io.db.x, roboflow_io.batch_frames, roboflow_io.db.q, roboflow_io.project, roboflow_io.sleep
try:
    roboflow_io.db.q1 = lambda *a: {"round": 1, "rf_workspace": "w", "rf_project": "p"}
    roboflow_io.db.x = lambda *a: None
    roboflow_io.batch_frames = lambda b: fake_frames
    roboflow_io.db.q = lambda *a: []   # no label set: images only
    fp, fj = FakeProject(), FakeJob()
    roboflow_io.project = lambda *a: fp
    roboflow_io.api_upload(fj, 1, None)
    assert sorted(fp.sent) == sorted(str(roboflow_io.SAMPLING_DIR / f["path"]) for f in fake_frames) and fj.last == 1
    assert fp.ann == [None] * 30
    # the chosen label set goes up; a frame with no boxes in it goes as an image only
    roboflow_io.db.q = lambda sql, args=(): [{"frame_id": 2, "path": "labels/9/f2.txt", "counts": '{"ball":1,"player":0}'},
                                             {"frame_id": 4, "path": "labels/9/f4.txt", "counts": '{"ball":0}'}] \
        if args == (9,) else []
    fp = FakeProject()
    roboflow_io.project = lambda *a: fp
    roboflow_io.api_upload(FakeJob(), 1, 9)
    assert [a for a in fp.ann if a] == [str(roboflow_io.DATA / "labels/9/f2.txt")]
    roboflow_io.db.q = lambda *a: []
    roboflow_io.sleep = lambda s: None
    fp = FakeProject(drops=3)   # dropped three times, then goes through
    roboflow_io.project = lambda *a: fp
    roboflow_io.api_upload(FakeJob(), 1, None)
    assert len(fp.sent) == 30
    roboflow_io.project = lambda *a: FakeProject(drops=4)
    try:
        roboflow_io.api_upload(FakeJob(), 1, None)
        raise AssertionError("gave up too late")
    except RuntimeError:
        pass
    roboflow_io.project = lambda *a: FakeProject(conflict="f3")
    assert "1 were already labelled" in roboflow_io.api_upload(FakeJob(), 1, None)
    fp = FakeProject(fail="f0")
    roboflow_io.project = lambda *a: fp
    try:
        roboflow_io.api_upload(FakeJob(), 1, None)
        raise AssertionError("failed upload not reported")
    except IOError:
        pass
    assert len(fp.sent) < 29   # the rest of the batch is cancelled, not uploaded unseen
finally:
    roboflow_io.db.q1, roboflow_io.db.x, roboflow_io.batch_frames, roboflow_io.db.q, roboflow_io.project, roboflow_io.sleep = real

# A batch with no Roboflow project chosen is refused before anything is sent
try:
    roboflow_io.project("w", None)
    raise AssertionError("upload without a project")
except ValueError:
    pass

# Classes page: add, merge and delete renumber the stored labels and remember old names
import json

import labels
with tempfile.TemporaryDirectory() as tmp:
    real = db.DB_PATH, classes.DATA, classes.DB_PATH, labels.DATA, classes.jobs.submit
    db.DB_PATH = classes.DB_PATH = Path(tmp) / "t.db"
    classes.DATA = labels.DATA = Path(tmp)
    queued = []   # the job queue, run by hand below; most checks run each change straight away
    classes.jobs.submit = lambda kind, title, fn, *a: fn(FakeJob(), *a)
    try:
        db.init()
        db.x("INSERT INTO batch (id, name, round, status, created) VALUES (1,'b',1,'new','x')")
        db.x("INSERT INTO label_set (id, batch_id, kind, source, created) VALUES (1,1,'corrected','t','x')")
        db.x("INSERT INTO match (id, file, path, split, status, created) VALUES ('M01','f','p','train','ready','x')")
        db.x("INSERT INTO sample_run (id, match_id, mode, params, status, created) VALUES (1,'M01','fixed','{}','done','x')")
        db.x("INSERT INTO frame (id, name, match_id, sample_run_id, path, t, frame_idx) VALUES (1,'a.jpg','M01',1,'a.jpg',0,0)")
        # one box per class: ball 0, goalkeeper 1, player 2, referee 3
        labels.save_rows([labels.write(1, {"id": 1, "name": "a.jpg"}, [(c, .5, .5, .1, .1) for c in range(4)])])
        ids = lambda: [b[0] for b in labels.read(db.q1("SELECT * FROM label"))]
        cnt = lambda: json.loads(db.q1("SELECT counts FROM label")["counts"])
        classes.add("coach")                                   # ball 0, coach 1, goalkeeper 2, player 3, referee 4
        assert classes.names() == ["ball", "coach", "goalkeeper", "player", "referee"] and ids() == [0, 2, 3, 4]
        classes.merge(["goalkeeper", "player"], "player")      # ball 0, coach 1, player 2, referee 3
        assert ids() == [0, 2, 2, 3] and cnt() == {"ball": 1, "coach": 0, "player": 2, "referee": 1}
        classes.delete(["referee"])
        assert ids() == [0, 2, 2] and classes.aliases() == {"goalkeeper": "player", "referee": None}
        classes.merge(["player", "coach"], "coach")             # goalkeeper now also reads as coach
        assert classes.aliases()["goalkeeper"] == "coach" and ids() == [0, 1, 1]
        classes.add("goalkeeper")                               # a name comes back to life
        assert "goalkeeper" not in classes.aliases() and ids() == [0, 1, 1]
        for bad in (lambda: classes.add("Bad Name"), lambda: classes.delete(classes.names()), lambda: classes.merge(["ball"], "ball")):
            try:
                bad()
                raise AssertionError("bad class change accepted")
            except ValueError:
                pass
        assert len(list((Path(tmp) / "backups").iterdir())) == 5   # one backup per change
        # the same add sent twice at once (a double click) renumbers once; the second is refused
        import threading as _th
        errs = []
        def twice():
            try:
                classes.add("aaa")
            except ValueError as e:
                errs.append(e)
        ts = [_th.Thread(target=twice) for _ in range(2)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        assert len(errs) == 1 and ids() == [1, 2, 2] and classes.names()[0] == "aaa"
        # queued: the Classes page shows the result at once; the real list and files switch when the job runs
        classes.jobs.submit = lambda kind, title, fn, *a: queued.append((fn, a))
        classes.delete(["aaa"])
        assert [n for n, _ in classes.shown()] == ["ball", "coach", "goalkeeper"] and classes.names()[0] == "aaa"
        try:
            classes.add("linesman")   # one change at a time
            raise AssertionError("second change queued behind a pending one")
        except ValueError:
            pass
        try:
            training.create("x", "yolo26n.pt", {}, [1])
            raise AssertionError("training started mid-renumber")
        except ValueError as e:
            assert "renumbered" in str(e)
        fn, a = queued.pop()
        fn(FakeJob(), *a)
        assert classes.pending is None and classes.names() == ["ball", "coach", "goalkeeper"] and ids() == [0, 1, 1]
        db.x("INSERT INTO training_run (name, base_model, params, label_set_ids, status, created) "
             "VALUES ('r','m','{}','[]','running','x')")
        try:
            classes.add("linesman")
            raise AssertionError("changed classes under a running training run")
        except ValueError:
            pass
        assert classes.pending is None
    finally:
        db.DB_PATH, classes.DATA, classes.DB_PATH, labels.DATA, classes.jobs.submit = real

print("All Kitman checks passed")
