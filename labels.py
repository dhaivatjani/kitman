"""Label files (YOLO txt) and pre-labelling with a registered model."""

import json
import re
from pathlib import Path

import cv2
import numpy as np

import classes
import db
from config import DATA, SAMPLING_DIR
from camera_benchmark import MIN_HULL_FRAC, UI_TOP_FRAC, on_pitch, pitch_mask  # copy of benchmark/camera_benchmark.py: same on-pitch rule as the EDA

PERSON_CONF, BALL_CONF = 0.30, 0.15
RF_SUFFIX = re.compile(r"_(jpg|jpeg|png)\.rf\.[0-9a-f]+$", re.I)


def label_dir(label_set_id):
    d = DATA / "labels" / str(label_set_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def frame_stem(filename):
    """'M05_frame_003704_jpg.rf.ab12.txt' -> 'M05_frame_003704' (Roboflow renames on export)."""
    return RF_SUFFIX.sub("", Path(filename).stem)


def parse(text):
    """YOLO lines -> list of (cls, cx, cy, w, h). Raises on malformed lines."""
    boxes = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"line {n}: expected 5 values, got {len(parts)}")
        c, *v = parts
        v = [float(a) for a in v]
        if not all(0.0 <= a <= 1.0 for a in v):
            raise ValueError(f"line {n}: coordinates must be between 0 and 1")
        boxes.append((int(c), *v))
    return boxes


def write(label_set_id, frame, boxes):
    """Store one frame's boxes (class ids in the Classes page's order) and its label row."""
    path = label_dir(label_set_id) / (Path(frame["name"]).stem + ".txt")
    write_file(path, boxes)
    return (label_set_id, frame["id"], str(path.relative_to(DATA)), db.jdump(counts(boxes, classes.names())))


def write_file(path, boxes):
    path.write_text("".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in boxes),
                    encoding="utf-8")


def counts(boxes, names):
    """{class name: boxes}, every class listed so an empty one shows as 0."""
    return {n: sum(b[0] == i for b in boxes) for i, n in enumerate(names)}


def save_rows(rows):
    db.xmany("INSERT OR REPLACE INTO label (label_set_id, frame_id, path, counts) VALUES (?,?,?,?)", rows)


def read(label_row):
    p = DATA / label_row["path"]
    return parse(p.read_text(encoding="utf-8")) if p.exists() else []


# ---------------------------------------------------------------- pre-labelling

def prelabel(job, batch_id, model_id):
    from ultralytics import YOLO

    model = db.q1("SELECT * FROM model WHERE id=?", (model_id,))
    frames = db.q("SELECT f.* FROM frame f JOIN batch_frame b ON b.frame_id=f.id WHERE b.batch_id=? ORDER BY f.name",
                  (batch_id,))
    if not frames:
        raise ValueError("This batch has no frames")
    imgsz = 1920
    if model["training_run_id"]:
        imgsz = json.loads(db.q1("SELECT params FROM training_run WHERE id=?", (model["training_run_id"],))["params"])["imgsz"]
    yolo = YOLO(model["path"])
    live = classes.names()
    to_ours = class_map(yolo.names, live)   # None = a COCO model, read person/sports ball
    ball = live.index("ball") if "ball" in live else None

    ls = db.x("INSERT INTO label_set (batch_id, kind, source, model_id, created) VALUES (?,?,?,?,?)",
              (batch_id, "prelabel", model["name"], model_id, db.now()))
    rows = []
    for k, f in enumerate(frames):
        img = cv2.imread(str(SAMPLING_DIR / f["path"]))
        if img is None:
            raise FileNotFoundError(f"Missing image {f['path']}")
        H, W = img.shape[:2]
        if to_ours is None:
            res = yolo.predict(img, imgsz=imgsz, conf=BALL_CONF, classes=[0, 32], verbose=False)[0]
            boxes = coco_boxes(img, res, live)
        else:
            res = yolo.predict(img, imgsz=imgsz, conf=BALL_CONF, verbose=False)[0]
            boxes = []
            for (x1, y1, x2, y2), c, p in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int),
                                              res.boxes.conf.cpu().numpy()):
                ours = to_ours.get(c)
                if ours is not None and p >= (BALL_CONF if ours == ball else PERSON_CONF):
                    boxes.append(xyxy_to_yolo(ours, x1, y1, x2, y2, W, H))
        rows.append(write(ls, f, boxes))
        if k % 10 == 0:
            job.progress((k + 1) / len(frames), f"Pre-labelled {k + 1} of {len(frames)}")
    save_rows(rows)
    db.x("UPDATE batch SET status='pre-labelled' WHERE id=? AND status='new'", (batch_id,))
    return f"Pre-labelled {len(rows)} frames with {model['name']}"


def class_map(names, live=None, alias=None):
    """How to read a model's classes: {its id: our id} when it knows any of our classes by name
    (a merged name reads as the class it became, a deleted one is dropped), None when it's a COCO
    model (person -> player, sports ball -> ball). Raises if neither."""
    live = classes.names() if live is None else live
    ours, _ = classes.read_names(names, live, alias)
    ours = {i: c for i, c in ours.items() if c is not None}
    if ours:
        return ours
    if "person" in names.values():
        return None
    raise ValueError(f"The model's classes ({', '.join(list(names.values())[:8])}) include neither "
                     f"{', '.join(live)} nor COCO's person, so it can't pre-label.")


def coco_boxes(img, res, live):
    """COCO person -> player, sports ball -> ball (when those classes exist), kept only when on the pitch.
    Frames without a pitch (replays, menus from fixed-interval sampling) keep every box."""
    player = live.index("player") if "player" in live else None
    ball = live.index("ball") if "ball" in live else None
    H, W = img.shape[:2]
    _, pitch, _, hull_frac = pitch_mask(img)
    has_pitch = hull_frac >= MIN_HULL_FRAC
    inner = cv2.erode(pitch, np.ones((7, 7), np.uint8))
    boxes = []
    for (x1, y1, x2, y2), c, p in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int),
                                      res.boxes.conf.cpu().numpy()):
        if has_pitch and y2 < H * UI_TOP_FRAC:
            continue
        if c == 0 and player is not None and p >= PERSON_CONF and (not has_pitch or on_pitch(pitch, (x1 + x2) / 2, y2)):
            boxes.append(xyxy_to_yolo(player, x1, y1, x2, y2, W, H))
        elif c == 32 and ball is not None and p >= BALL_CONF and (not has_pitch or on_pitch(inner, (x1 + x2) / 2, (y1 + y2) / 2)):
            boxes.append(xyxy_to_yolo(ball, x1, y1, x2, y2, W, H))
    return boxes


def xyxy_to_yolo(c, x1, y1, x2, y2, W, H):
    x1, x2 = max(0.0, float(x1)), min(float(W), float(x2))
    y1, y2 = max(0.0, float(y1)), min(float(H), float(y2))
    return (c, (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H)
