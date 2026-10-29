"""Copied into Kitman from benchmark/camera_benchmark.py so Kitman runs on its own;
Kitman uses its on-pitch rule (pitch_mask, on_pitch, UI_TOP_FRAC, MIN_HULL_FRAC). Keep the two in step.

Camera-angle benchmark for FC Mobile recordings.

Runs a pretrained COCO YOLO model over every clip in "Screen Recordings/Angle N"
and measures, per analysed frame:

  - gameplay      : frame passes the pitch filter (loading screens, replays,
                    cutscenes and full-screen UI cards fail it and are excluded)
  - n_players     : 'person' detections (conf >= 0.30) whose foot point is on the pitch
  - ball          : a 'sports ball' detection (conf >= 0.15) on the pitch
  - bbox size     : median player box height / area, normalised by frame size
  - camera motion : global translation and zoom between consecutive analysed
                    frames, from optical flow on pitch pixels (players masked),
                    converted to per-second rates using the real timestamps
  - track ids     : ByteTrack IDs for on-pitch players (situation clips only)

The metrics come from raw detections, not tracker output: the tracker only
returns confirmed tracks, which undercounts short-lived objects like the ball.

Situation clips are analysed at their full frame rate so ByteTrack's IoU
matching works (players are ~25 px wide; at 10 fps a camera pan moves them
further than their own width). Full-match recordings are sampled at 10 fps
and get no tracking statistics.

Outputs (results/): frames.csv, clips.csv, situations.csv, angles_clips.csv,
angles_fullmatch.csv, and annotated sample frames for visual checks.

Usage:
    python camera_benchmark.py                               # everything
    python camera_benchmark.py --only "Camera 1 - KickOff"   # substring filter
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace, YAML
from ultralytics.utils.checks import check_yaml

ROOT = Path(__file__).resolve().parent
REC_DIR = ROOT.parent / "Screen Recordings"
OUT_DIR = ROOT / "results"

FULLMATCH_FPS = 10.0       # sampling rate for the long full-match recordings
IMGSZ = 1920               # native width is 2400; the ball needs high resolution
PERSON_CONF = 0.30         # counted as a visible player
TRACK_CONF = 0.10          # lower bound fed to ByteTrack (it uses low boxes for 2nd association)
BALL_CONF = 0.15
FULL_PLAYERS = 22          # PVR denominator; referee/linesmen can push n above 22, so capped
UI_TOP_FRAC = 0.09         # scoreboard / menu strip at the top of the screen
MIN_TRACK_S = 2.0          # a "stable" track lasts at least this long

# HSV range for the grass. FC Mobile grass is a saturated mid green.
GREEN_LO = np.array([30, 60, 40], np.uint8)
GREEN_HI = np.array([90, 255, 255], np.uint8)
MIN_GREEN_FRAC = 0.30      # below this the frame is not live gameplay
MIN_HULL_FRAC = 0.40

SITUATIONS = {
    "kickoff": "KickOff",
    "counter": "CounterAttack",
    "goalkick": "GoalKick",
    "corner": "CornerKick",
    "freekick": "FreeKick",
    "shrotfreekick": "FreeKick",
    "adressingconcern": "SustainedAttack",
    "attackinmotion": "SustainedAttack",
    "inboxdrama": "SustainedAttack",
    "neargoalline": "SustainedAttack",
    "changingsideofattack": "SwitchOfPlay",
}


def situation_of(name: str) -> str:
    if " - " not in name:
        return "FullMatch"
    tag = re.sub(r"[^a-z&]", "", name.split(" - ", 1)[1].lower())
    for key, label in SITUATIONS.items():
        if tag.startswith(key):
            return label
    return "Other"


def pitch_mask(frame):
    """Return (hull mask, pitch mask, green fraction, hull fraction), masks at 1/4 scale.

    The pitch mask is the filled outline of the largest connected grass region: filling
    the outline closes the holes players make, while the advertising boards keep the
    grass strips where photographers and stewards stand as separate regions. The convex
    hull is only used for the gameplay test and the camera-motion mask.
    """
    small = cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    h = small.shape[0]
    small[: int(h * UI_TOP_FRAC)] = 0
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, GREEN_LO, GREEN_HI)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    green_frac = float((green > 0).mean())
    hull_mask = np.zeros_like(green)
    pitch = np.zeros_like(green)
    contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return hull_mask, pitch, green_frac, 0.0
    big = max(contours, key=cv2.contourArea)
    cv2.fillPoly(hull_mask, [cv2.convexHull(big)], 255)
    cv2.fillPoly(pitch, [big], 255)
    return hull_mask, pitch, green_frac, float((hull_mask > 0).mean())


def on_pitch(pitch, x, y):
    hh, hw = pitch.shape
    hx, hy = int(x * 0.25), int(y * 0.25)
    if not (0 <= hx < hw and 0 <= hy < hh):
        return False
    return pitch[hy, hx] > 0


def camera_motion(prev_gray, gray, hull_mask, boxes, scale):
    """Similarity transform between consecutive analysed frames on pitch pixels.

    Returns (tx, ty, zoom): tx/ty in fractions of frame width, zoom as log scale
    change. NaNs when too few features survive (e.g. a plain grass close-up).
    """
    mask = cv2.resize(hull_mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(mask, (int(x1 * scale) - 3, int(y1 * scale) - 3),
                      (int(x2 * scale) + 3, int(y2 * scale) + 3), 0, -1)
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=400, qualityLevel=0.005,
                                  minDistance=8, mask=mask)
    if pts is None or len(pts) < 20:
        return np.nan, np.nan, np.nan
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, winSize=(21, 21), maxLevel=3)
    good = st.ravel() == 1
    if good.sum() < 20:
        return np.nan, np.nan, np.nan
    m, inliers = cv2.estimateAffinePartial2D(pts[good], nxt[good], method=cv2.RANSAC,
                                             ransacReprojThreshold=2.0)
    if m is None or inliers.sum() < 15:
        return np.nan, np.nan, np.nan
    w = gray.shape[1]
    return float(m[0, 2]) / w, float(m[1, 2]) / w, float(np.log(np.hypot(m[0, 0], m[1, 0])))


def draw_sample(frame, hull_mask, players, balls, path):
    vis = frame.copy()
    contours, _ = cv2.findContours(hull_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, [(c * 4).astype(np.int32) for c in contours], -1, (255, 255, 0), 3)
    for x1, y1, x2, y2, tid in players:
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        if tid >= 0:
            cv2.putText(vis, str(tid), (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    for x1, y1, x2, y2 in balls:
        cv2.rectangle(vis, (int(x1) - 8, int(y1) - 8), (int(x2) + 8, int(y2) + 8), (0, 255, 255), 3)
    cv2.putText(vis, f"players={len(players)} ball={len(balls) > 0}", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.imwrite(str(path), cv2.resize(vis, None, fx=0.5, fy=0.5))


def new_tracker(fps):
    cfg = IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml")))
    return BYTETracker(cfg, frame_rate=int(round(fps)))


def process_clip(model, path: Path, angle: int, samples_dir: Path):
    situation = situation_of(path.stem)
    full_match = situation == "FullMatch"
    cap = cv2.VideoCapture(str(path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    native_fps = n_frames / max(1e-6, _duration(path))
    tracker = None if full_match else new_tracker(native_fps)
    save_every_s = 45.0 if full_match else 3.0

    rows, next_t, next_save = [], 0.0, 1.0
    prev_gray, prev_t = None, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if full_match:
            if t + 1e-6 < next_t:
                continue
            next_t += 1.0 / FULLMATCH_FPS
        H, W = frame.shape[:2]

        hull_mask, pitch, green_frac, hull_frac = pitch_mask(frame)
        gameplay = green_frac >= MIN_GREEN_FRAC and hull_frac >= MIN_HULL_FRAC
        row = dict(angle=angle, clip=path.stem, situation=situation, t=round(t, 3),
                   gameplay=gameplay, green_frac=round(green_frac, 3))
        if not gameplay:
            prev_gray = None
            rows.append(row)
            continue

        res = model.predict(frame, imgsz=IMGSZ, conf=TRACK_CONF, classes=[0, 32], verbose=False)[0]
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        conf = res.boxes.conf.cpu().numpy()

        # the ball must sit well inside the pitch: board logos are a common false positive
        pitch_inner = cv2.erode(pitch, np.ones((7, 7), np.uint8))
        cand, balls, ball_conf = [], [], 0.0      # cand: on-pitch persons >= TRACK_CONF
        for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf):
            if y2 < H * UI_TOP_FRAC:
                continue
            if c == 0 and on_pitch(pitch, (x1 + x2) / 2, y2):
                cand.append([x1, y1, x2, y2, p, 0])
            elif c == 32 and p >= BALL_CONF and on_pitch(pitch_inner, (x1 + x2) / 2, (y1 + y2) / 2):
                balls.append((x1, y1, x2, y2))
                ball_conf = max(ball_conf, float(p))

        ids = -np.ones(len(cand), int)
        if tracker is not None:
            # ByteTrack's own det_idx refers to its internal high/low-score subsets, so the
            # original detection index is carried through the (unused) class column instead.
            data = np.array(cand, np.float32).reshape(-1, 6)
            data[:, 5] = np.arange(len(data))
            out = tracker.update(Boxes(data, (H, W)), frame)
            for tr in out:                       # [x1, y1, x2, y2, id, score, cls, det_idx]
                ids[int(tr[6])] = int(tr[4])
        players = [(*b[:4], tid) for b, tid in zip(cand, ids) if b[4] >= PERSON_CONF]
        tracked = [str(tid) for b, tid in zip(cand, ids) if b[4] >= PERSON_CONF and tid >= 0]

        heights = [(y2 - y1) / H for x1, y1, x2, y2, _ in players]
        areas = [(x2 - x1) * (y2 - y1) / (W * H) for x1, y1, x2, y2, _ in players]
        row.update(n_players=len(players), ball=len(balls) > 0, ball_conf=round(ball_conf, 3),
                   bbox_h_med=float(np.median(heights)) if heights else np.nan,
                   bbox_area_med=float(np.median(areas)) if areas else np.nan,
                   track_ids=" ".join(tracked))

        scale = 640.0 / W
        gray = cv2.cvtColor(cv2.resize(frame, (640, int(H * scale))), cv2.COLOR_BGR2GRAY)
        if prev_gray is not None and t > prev_t:
            tx, ty, zoom = camera_motion(prev_gray, gray, hull_mask, [p[:4] for p in players], scale)
            dt = t - prev_t
            row.update(cam_vx=tx / dt, cam_vy=ty / dt, cam_zoom_rate=zoom / dt, dt=round(dt, 4))
        prev_gray, prev_t = gray, t
        rows.append(row)

        if t >= next_save:
            draw_sample(frame, hull_mask, players, balls, samples_dir / f"{path.stem}_t{t:05.1f}.jpg")
            next_save = t + save_every_s
    cap.release()
    return rows


def _duration(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n - 1))
    cap.read()
    d = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    cap.release()
    return d if d > 0 else n / 30.0


def _weighted_quantile(x, w, q):
    x, w = np.asarray(x), np.asarray(w)
    keep = ~np.isnan(x)
    x, w = x[keep], w[keep]
    if not len(x):
        return np.nan
    order = np.argsort(x)
    cw = np.cumsum(w[order])
    return float(x[order][np.searchsorted(cw, q * cw[-1])])


def summarise(df: pd.DataFrame, keys):
    """Aggregate per group. Frames are weighted by their duration so clips at
    different frame rates (and the 10 fps full matches) count per second of play."""
    out = []
    for key, d_all in df.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        d = d_all[d_all.gameplay].copy()
        if d.empty:
            continue
        # duration each analysed frame stands for
        d["w"] = d.groupby("clip").t.diff().shift(-1)
        d["w"] = d.w.where(d.w.between(0, 0.5)).fillna(d.w.median() if d.w.notna().any() else 0.1)
        w = d.w.to_numpy()
        secs = w.sum()

        def wmean(s):
            s = s.astype(float).to_numpy()
            k = ~np.isnan(s)
            return float((s[k] * w[k]).sum() / w[k].sum()) if k.any() else np.nan

        speed = np.hypot(d.cam_vx, d.cam_vy)            # frame widths per second
        zoom = d.cam_zoom_rate.abs()                     # |log scale| per second
        row = dict(zip(keys, key)) | dict(
            clips=d["clip"].nunique(),
            gameplay_s=round(secs, 1),
            gameplay_share=round(d_all.gameplay.mean(), 3),
            players_mean=round(wmean(d.n_players), 2),
            PVR=round(wmean(d.n_players.clip(upper=FULL_PLAYERS) / FULL_PLAYERS), 3),
            share_ge18=round(wmean(d.n_players >= 18), 3),
            players_p10=_weighted_quantile(d.n_players, w, 0.10),
            BVR=round(wmean(d.ball), 3),
            bbox_h_med_pct=round(100 * _weighted_quantile(d.bbox_h_med, w, 0.5), 2),
            bbox_area_med_x1e4=round(1e4 * _weighted_quantile(d.bbox_area_med, w, 0.5), 2),
            cam_speed_mean=round(wmean(speed), 4),
            cam_speed_p95=round(_weighted_quantile(speed, w, 0.95), 4),
            cam_zoom_mean=round(wmean(zoom), 4),
            cam_zoom_p95=round(_weighted_quantile(zoom, w, 0.95), 4),
            motion_valid=round(wmean(d.cam_vx.notna()), 3),
        )
        if d.track_ids.notna().any() and (d.situation != "FullMatch").all():
            spans, dets = {}, {}
            for clip, dc in d.groupby("clip"):
                for t, ids in zip(dc.t, dc.track_ids.fillna("").astype(str)):
                    for i in ids.split():
                        k = (clip, i)
                        a, b = spans.get(k, (t, t))
                        spans[k] = (min(a, t), max(b, t))
                        dets[k] = dets.get(k, 0) + 1
            life = {k: b - a for k, (a, b) in spans.items()}
            total = sum(dets.values())
            row |= dict(
                track_len_med_s=round(_weighted_quantile([life[k] for k in dets], [dets[k] for k in dets], 0.5), 2)
                if dets else np.nan,
                dets_in_stable_tracks=round(sum(n for k, n in dets.items() if life[k] >= MIN_TRACK_S) / total, 3)
                if total else np.nan,
                ids_per_player_min=round(len(life) / (secs / 60) / max(row["players_mean"], 1), 2),
            )
        out.append(row)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="substring filter on clip name")
    ap.add_argument("--model", default=str(ROOT / "yolo11m.pt"))
    ap.add_argument("--summarise-only", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    samples_dir = OUT_DIR / "samples"
    samples_dir.mkdir(exist_ok=True)
    frames_csv = OUT_DIR / "frames.csv"

    if not args.summarise_only:
        model = YOLO(args.model)
        clips = []
        for d in sorted(REC_DIR.glob("Angle *")):
            angle = int(d.name.split()[-1])
            for f in sorted(d.glob("*.mp4")):
                if args.only is None or args.only.lower() in f.stem.lower():
                    clips.append((angle, f))
        done = set(pd.read_csv(frames_csv, usecols=["clip"])["clip"].unique()) if frames_csv.exists() else set()
        for angle, f in clips:
            if f.stem in done:
                print(f"skip (already in frames.csv) {f.name}")
                continue
            print(f"[angle {angle}] {f.name}", flush=True)
            rows = process_clip(model, f, angle, samples_dir)
            pd.DataFrame(rows).to_csv(frames_csv, mode="a", header=not frames_csv.exists(), index=False)

    df = pd.read_csv(frames_csv, low_memory=False)
    df["gameplay"] = df.gameplay.astype(bool)
    df["ball"] = df.ball.fillna(False).astype(bool)
    clips = df[df.situation != "FullMatch"]
    summarise(df, ["angle", "clip", "situation"]).to_csv(OUT_DIR / "clips.csv", index=False)
    summarise(clips, ["angle", "situation"]).to_csv(OUT_DIR / "situations.csv", index=False)
    summarise(clips, ["angle"]).to_csv(OUT_DIR / "angles_clips.csv", index=False)
    summarise(df[df.situation == "FullMatch"], ["angle"]).to_csv(OUT_DIR / "angles_fullmatch.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
