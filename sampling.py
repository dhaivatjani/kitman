"""Frame sampling, two modes (PLAN.md):

  custom  per-match plan from the EDA: windows, train quota buckets, live play only,
          >= 2 s apart, near-duplicates avoided, plus a 10 s tracking clip for val/test.
  fixed   one frame every N seconds over the whole video, nothing filtered.

Both pick by timestamp and snap to the real frame (the videos are variable frame
rate), then decode only the chosen frames.
"""

import csv
import json
import random

from pathlib import Path

import cv2
import numpy as np

import analysis
import db
from config import FRAME_H, FRAME_W, SAMPLING_DIR

CUSTOM_DEFAULTS = {
    "seed": 42, "train_window_s": 5.0, "eval_window_s": 6.0, "min_gap_s": 2.0,
    "quota": {"random": 0.5, "fast_pan": 0.2, "ball_missed": 0.2, "edge": 0.1},
    "clip_s": 10.0, "max_snap_s": 0.25, "freeze_s": 0.5, "crowd_min": 7,
    "nonlive_ratio": 1.0, "nonlive_gap_s": 1.0,
}
NONLIVE = {"replay": ("replay_or_cutscene",), "menu": ("other", "black")}   # bucket -> analysis states
FIXED_DEFAULTS = {"interval_s": 5.0}


# ---------------------------------------------------------------- planning (pure)

def snap(times, pts, max_snap=None):
    """Nearest real frame index for each time; None where it is further than max_snap."""
    pts = np.asarray(pts)
    out = []
    for t in times:
        i = int(np.searchsorted(pts, t))
        cands = [j for j in (i - 1, i) if 0 <= j < len(pts)]
        j = min(cands, key=lambda j: abs(pts[j] - t))
        out.append(j if max_snap is None or abs(pts[j] - t) <= max_snap else None)
    return out


def plan_fixed(pts, duration, interval_s):
    if interval_s <= 0:
        raise ValueError("Interval must be more than 0 seconds")
    times = np.arange(0.0, duration, interval_s)
    seen, picks = set(), []
    for t, idx in zip(times, snap(times, pts)):
        if idx not in seen:              # a long freeze can map two times to one frame
            seen.add(idx)
            picks.append({"t": float(pts[idx]), "frame_idx": idx, "bucket": "fixed"})
    return picks


def edge_times(live, pts, dets, p):
    """Times that count as edge cases: first live sample after a freeze, or players crowded together."""
    pts = np.asarray(pts)
    gaps = np.where(np.diff(pts) > p["freeze_s"])[0]
    after_freeze = set()
    lt = live.t.to_numpy()
    for g in gaps:
        k = np.searchsorted(lt, pts[g + 1])
        if k < len(lt) and lt[k] - pts[g + 1] <= 1.0:
            after_freeze.add(float(lt[k]))
    crowded = set()
    for t, d in dets.groupby("t"):
        xy = d[["cx", "cy"]].to_numpy()
        near = ((np.abs(xy[:, None, 0] - xy[None, :, 0]) < 0.08) &
                (np.abs(xy[:, None, 1] - xy[None, :, 1]) < 0.15)).sum(1)
        if len(near) and near.max() >= p["crowd_min"]:
            crowded.add(float(t))
    # shortcut: GOAL/HALFTIME overlays are not detected (no EDA column for them); add a panel test if edge frames look thin.
    return after_freeze | crowded


def plan_custom(split, samples, dets, pts, p, rng):
    """samples: this match's 2 fps EDA rows. Returns frame picks (and clip picks for val/test)."""
    live = samples[samples.state == "live"].sort_values("t").reset_index(drop=True)
    if live.empty:
        return [], {"windows": 0}
    window = p["train_window_s"] if split == "train" else p["eval_window_s"]
    live["window"] = (live.t // window).astype(int)
    windows = sorted(live.window.unique())

    is_bucket = {"random": np.ones(len(live), bool)}
    if split == "train":
        thr = np.nanpercentile(live.flow_mag, 75)
        is_bucket["fast_pan"] = (live.flow_mag >= thr).to_numpy()
        is_bucket["ball_missed"] = (live.n_ball == 0).to_numpy()   # NaN rows had no detector pass
        edges = edge_times(live, pts, dets, p)
        is_bucket["edge"] = live.t.isin(edges).to_numpy()
        # Scarcest bucket first, each given only to windows that contain such a frame; the rest are random.
        win_of = live.window.to_numpy()
        assigned = {}
        for b in sorted((b for b in p["quota"] if b != "random"), key=lambda b: is_bucket[b].sum()):
            has = sorted({int(w) for w in win_of[is_bucket[b]]} - set(assigned))
            rng.shuffle(has)
            for w in has[:int(round(p["quota"][b] * len(windows)))]:
                assigned[w] = b
        wanted = [assigned.get(int(w), "random") for w in windows]
    else:
        wanted = ["random"] * len(windows)

    picks, last_t, fell_back = [], -1e9, 0
    near_dup = live.near_dup.fillna(0).to_numpy() == 1
    for w, bucket in zip(windows, wanted):
        rows = np.where((live.window == w).to_numpy() & (live.t >= last_t + p["min_gap_s"]).to_numpy())[0]
        if not len(rows):
            continue
        fresh = rows[~near_dup[rows]]
        rows = fresh if len(fresh) else rows
        hit = rows[is_bucket[bucket][rows]]
        if not len(hit):
            hit, fell_back, bucket = rows, fell_back + 1, "random"
        i = int(rng.choice(hit))
        picks.append({"t": float(live.t[i]), "bucket": bucket, "window": int(w)})
        last_t = picks[-1]["t"]

    stats = {"windows": len(windows), "fell_back_to_random": fell_back}
    extra, stats["nonlive"] = plan_nonlive(samples, round(p["nonlive_ratio"] * len(picks)), p, rng)
    picks += extra
    if split != "train":
        clip = best_clip(samples, p["clip_s"])
        if clip is not None:
            stats["clip"] = clip
    return picks, stats


def plan_nonlive(samples, target, p, rng):
    """About `target` distinct frames that aren't live play: replays/cutscenes (labelled like any
    frame) and menus/loading/black (empty-label negatives), shared by how much of each the video has."""
    s = samples.sort_values("t").reset_index(drop=True)
    fresh = s.near_dup.fillna(0).to_numpy() != 1
    pools = {b: s.t[s.state.isin(states).to_numpy() & fresh].tolist() for b, states in NONLIVE.items()}
    total = sum(len(v) for v in pools.values())
    picks, taken = [], []

    def take(b, pool, want):
        for t in pool:
            if want <= 0:
                return
            if t not in taken and all(abs(t - u) >= p["nonlive_gap_s"] for u in taken):
                taken.append(t)
                picks.append({"t": float(t), "bucket": b})
                want -= 1

    for b, pool in pools.items():
        rng.shuffle(pool)
        take(b, pool, round(target * len(pool) / total) if total else 0)
    for b, pool in pools.items():   # one kind ran short: let the other make up the difference
        take(b, pool, target - len(picks))
    stats = {"target": target, **{b: sum(x["bucket"] == b for x in picks) for b in pools},
             "shortfall": target - len(picks)}
    return picks, stats


def best_clip(samples, length):
    """Start time of the clip_s stretch of uninterrupted live play with the fastest camera."""
    s = samples.sort_values("t").reset_index(drop=True)
    n = int(round(length / 0.5)) + 1
    ok = ((s.state == "live") & (s.scene_cut.fillna(0) == 0)).to_numpy()
    flow = s.flow_mag.fillna(0).to_numpy()
    best, best_v = None, -1.0
    for i in range(0, len(s) - n + 1):
        if ok[i:i + n].all() and np.allclose(np.diff(s.t[i:i + n]), 0.5):
            v = flow[i:i + n].mean()
            if v > best_v:
                best, best_v = float(s.t[i]), v
    return None if best is None else {"start": best, "end": best + length, "mean_flow_px": round(best_v, 2)}


# ---------------------------------------------------------------- extraction

def extract(video_path, picks, job=None):
    """Decode the picked frame indices in one pass and write them as JPEG."""
    todo = {}   # frame index -> every file it should become (a live pick can fall inside the tracking clip)
    for p in picks:
        if not (SAMPLING_DIR / p["path"]).exists():
            todo.setdefault(p["frame_idx"], []).append(p["path"])
    if not todo:
        return
    last = max(todo)
    cap = cv2.VideoCapture(str(video_path))
    idx, done = -1, 0
    while idx < last:
        if not cap.grab():
            break
        idx += 1
        if idx in todo:
            ok, frame = cap.retrieve()
            if not ok:
                raise RuntimeError(f"Could not decode frame {idx} of {video_path.name}")
            if frame.shape[:2] != (FRAME_H, FRAME_W):   # other phones: same stretch as the analysis pass
                frame = cv2.resize(frame, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
            for path in todo[idx]:
                out = SAMPLING_DIR / path
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            done += 1
            if job and done % 10 == 0:
                job.progress(done / len(todo), f"Decoded {done} of {len(todo)} frames")
    cap.release()
    missing = [p for paths in todo.values() for p in paths if not (SAMPLING_DIR / p).exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} frames past the end of {video_path.name}")


def run(job, run_id):
    """Background job: plan, extract and register frames for one sample_run."""
    try:
        return _run(job, run_id)
    except Exception:
        db.x("UPDATE sample_run SET status='failed' WHERE id=?", (run_id,))
        raise


def _run(job, run_id):
    r = db.q1("SELECT * FROM sample_run WHERE id=?", (run_id,))
    m = db.q1("SELECT * FROM match WHERE id=?", (r["match_id"],))
    params = json.loads(r["params"])
    db.x("UPDATE sample_run SET status='running' WHERE id=?", (run_id,))
    pts = analysis.pts(m["id"])
    if pts is None:
        raise ValueError(f"{m['id']} has no frame timestamps yet. Wait for it to finish preparing.")
    rng = random.Random(f"{params.get('seed', 42)}-{m['id']}-{r['mode']}")

    if r["mode"] == "fixed":
        picks, stats = plan_fixed(pts, m["duration_s"], float(params["interval_s"])), {}
    else:
        samples, dets = analysis.samples(m["id"]), analysis.detections(m["id"])
        if samples is None:
            raise ValueError(f"{m['id']} hasn't been analysed yet, so the custom plan can't find live play.")
        dets = dets[(dets.cls == "person") & dets.on_pitch]
        picks, stats = plan_custom(m["split"], samples, dets, pts, params, rng)
        idxs = snap([p["t"] for p in picks], pts, params["max_snap_s"])
        stats["dropped_in_freeze"] = sum(i is None for i in idxs)
        picks = [p | {"frame_idx": i, "t": float(pts[i])} for p, i in zip(picks, idxs) if i is not None]
        if "clip" in stats:
            c = stats["clip"]
            clip_idx = np.where((pts >= c["start"]) & (pts < c["end"]))[0]
            picks += [{"t": float(pts[i]), "frame_idx": int(i), "bucket": "clip"} for i in clip_idx]

    for p in picks:
        stem = "clip_frame" if p["bucket"] == "clip" else "_frame"
        p["name"] = f"{m['id']}{stem}_{p['frame_idx']:06d}.jpg"   # M05_frame_003704 / M04clip_frame_…
        p["path"] = f"{m['id']}/images/{p['name']}"
    job.log(f"{m['id']} {r['mode']}: {len(picks)} frames planned {stats}")

    extract(Path(m["path"]), picks, job)

    existing = {row["name"] for row in db.q("SELECT name FROM frame WHERE match_id=?", (m["id"],))}
    new = [p for p in picks if p["name"] not in existing]
    db.xmany("INSERT INTO frame (name, match_id, sample_run_id, path, t, frame_idx, bucket) VALUES (?,?,?,?,?,?,?)",
             [(p["name"], m["id"], run_id, p["path"], p["t"], p["frame_idx"], p["bucket"]) for p in new])

    # Plain-file record next to the images, as PLAN.md lays out.
    out = SAMPLING_DIR / m["id"]
    (out / f"plan_run{run_id}.json").write_text(json.dumps(
        {"match": m["id"], "split": m["split"], "mode": r["mode"], "params": params, "stats": stats,
         "frames": len(picks), "new_frames": len(new)}, indent=2), encoding="utf-8")
    frames_csv = out / "frames.csv"
    first = not frames_csv.exists()
    with open(frames_csv, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if first:
            w.writerow(["name", "t", "frame_idx", "bucket", "mode", "run", "round"])
        w.writerows([p["name"], round(p["t"], 3), p["frame_idx"], p["bucket"], r["mode"], run_id, 1] for p in new)

    db.x("UPDATE sample_run SET status='done', n_frames=? WHERE id=?", (len(picks), run_id))
    dup = len(picks) - len(new)
    return f"{m['id']}: {len(new)} new frames" + (f" ({dup} were already sampled)" if dup else "")
