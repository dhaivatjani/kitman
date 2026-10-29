"""Adding a video: store Kitman's own copy, read its frame timestamps, then run the
EDA's live-play pass on it (video_eda.py, copied from eda/, so every video is judged by the same rules).

  timestamps  -> data/analysis/<id>/packets.npz     enough for Fixed interval sampling
  live pass   -> data/analysis/<id>/samples.csv     2 fps: state, camera pan, players, ball
                 data/analysis/<id>/detections.csv  1 fps YOLO11m COCO boxes
                 data/analysis/<id>/sheet.jpg       12 thumbnails across the video
Frames from any other resolution are stretched to 2400x1080 on the fly (the original file is untouched).
"""

import json
import os
import shutil
import subprocess

import cv2
import numpy as np
import pandas as pd

import db
from config import ANALYSIS_DIR, COCO_MODEL, FRAME_H, FRAME_W, VIDEOS_DIR
import video_eda  # copy of eda/video_eda.py: the same 2 fps pass that produced the EDA dashboard

PERSON_CONF = 0.30   # make_tables.py: on-pitch people at >= 0.30, on-pitch balls at >= 0.15


def probe(path):
    """Duration, frame count, size; raises ValueError when it isn't a readable video."""
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=width,height,nb_frames:format=duration", "-of", "json", str(path)],
                             capture_output=True, text=True, check=True).stdout
        js = json.loads(out)
        s = js["streams"][0]
        dur = float(js["format"]["duration"])
    except (subprocess.CalledProcessError, KeyError, IndexError, ValueError):
        raise ValueError(f"{path.name} is not a video ffprobe can read") from None
    frames = int(s.get("nb_frames") or 0)
    return {"duration_s": round(dur, 2), "frames": frames, "avg_fps": round(frames / dur, 1) if dur else 0,
            "width": int(s["width"]), "height": int(s["height"])}


def next_id():
    taken = {r["id"] for r in db.q("SELECT id FROM match")}
    n = 1
    while f"M{n:02d}" in taken:
        n += 1
    return f"M{n:02d}"


def store(src, vid):
    """Kitman's own copy in data/videos. A hard link on the same drive (instant, no extra
    space, survives the original being deleted); a real copy otherwise."""
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    dst = VIDEOS_DIR / f"{vid}{src.suffix.lower()}"
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst


def register(file_name, path, split, description):
    vid = next_id()
    db.x("INSERT INTO match (id, file, path, description, split, status, created) VALUES (?,?,?,?,?,?,?)",
         (vid, file_name, str(path), description, split, "queued", db.now()))
    return vid


def prepare(job, vid, source=None):
    """Background job for one video: (copy), probe, timestamps, live-play pass."""
    try:
        return _prepare(job, vid, source)
    except Exception as e:
        db.x("UPDATE match SET status='failed', message=? WHERE id=?", (f"{type(e).__name__}: {e}", vid))
        raise


def _prepare(job, vid, source):
    from pathlib import Path
    if source:
        db.x("UPDATE match SET status='copying' WHERE id=?", (vid,))
        job.progress(0.01, f"Storing {vid} in Kitman")
        path = store(Path(source), vid)
        db.x("UPDATE match SET path=? WHERE id=?", (str(path), vid))
    else:
        path = Path(db.q1("SELECT path FROM match WHERE id=?", (vid,))["path"])

    info = probe(path)
    note = None
    if (info["width"], info["height"]) != (FRAME_W, FRAME_H):
        note = f"Recorded at {info['width']}×{info['height']}; frames are stretched to {FRAME_W}×{FRAME_H}."
    db.x("UPDATE match SET duration_s=?, frames=?, avg_fps=?, width=?, height=?, status='analysing', message=? "
         "WHERE id=?", (info["duration_s"], info["frames"], info["avg_fps"], info["width"], info["height"], note, vid))

    out = ANALYSIS_DIR / vid
    out.mkdir(parents=True, exist_ok=True)
    job.progress(0.03, f"Reading {vid} frame timestamps")
    pts, size, key = video_eda.packets(path)
    np.savez_compressed(out / "packets.npz", pts=pts, size=size, key=key)
    job.log(f"{vid}: {len(pts)} frames, {info['duration_s']} s, {info['width']}x{info['height']}")

    from ultralytics import YOLO
    model = YOLO(str(COCO_MODEL))
    orig = video_eda.frames_at

    def tracked(p, fps):   # report progress from inside the EDA pass
        for t, frame in orig(p, fps):
            if int(t * fps) % 20 == 0:
                job.progress(0.05 + 0.93 * min(1.0, t / info["duration_s"]), f"Analysing {vid}: {t:.0f} of {info['duration_s']:.0f} s")
            yield t, frame

    video_eda.frames_at = tracked   # jobs run one at a time, so the swap can't leak into another job
    try:
        det_rows = []
        rows, _, _, _, sheet = video_eda.visual_pass(path, vid, model, det_rows)
    finally:
        video_eda.frames_at = orig

    df = video_eda.label_segments(pd.DataFrame(rows))
    dets = pd.DataFrame(det_rows, columns=["match_id", "t", "cls", "conf", "on_pitch", "cx", "cy", "w", "h"])
    kept = dets[dets.on_pitch & ((dets.cls == "ball") | (dets.conf >= PERSON_CONF))]
    counts = kept.assign(n_person=kept.cls.eq("person"), n_ball=kept.cls.eq("ball")) \
        .groupby("t")[["n_person", "n_ball"]].sum()
    det_t = df.loc[df.get("det", False) == True, ["t"]]  # noqa: E712  zero-fill samples YOLO saw but found nothing
    det_t = det_t.join(counts, on="t").fillna(0)
    df = df.merge(det_t, on="t", how="left")
    df["scene_cut"] = df.scene_cut.astype(int)
    df["near_dup"] = df.near_dup.astype(int)
    df[["t", "state", "flow_mag", "scene_cut", "near_dup", "n_person", "n_ball", "panel_frac", "green_frac",
        "sharpness"]].to_csv(out / "samples.csv", index=False)
    dets.to_csv(out / "detections.csv", index=False)
    cv2.imwrite(str(out / "sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])

    live = (df.state == "live").mean()
    db.x("UPDATE match SET status='ready' WHERE id=?", (vid,))
    return f"{vid} ready: {live:.0%} live play over {info['duration_s'] / 60:.1f} min"


def samples(vid):
    p = ANALYSIS_DIR / vid / "samples.csv"
    return pd.read_csv(p) if p.exists() else None


def detections(vid):
    p = ANALYSIS_DIR / vid / "detections.csv"
    return pd.read_csv(p) if p.exists() else None


def pts(vid):
    p = ANALYSIS_DIR / vid / "packets.npz"
    return np.load(p)["pts"] if p.exists() else None
