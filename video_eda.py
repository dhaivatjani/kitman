"""Copied into Kitman from eda/video_eda.py so Kitman runs on its own; Kitman uses
packets, frames_at, visual_pass and label_segments. Keep the two in step.

Exploratory data analysis of the Final Recordings (FC Mobile screen captures).

Per video it collects:
  1. Container / stream metadata               (ffprobe: codec, profile, bitrate, colour, audio)
  2. Frame timing and compression              (ffprobe packets, no decode: VFR, gaps, GOP, frame sizes)
  3. Audio levels                              (ffmpeg volumedetect + silencedetect)
  4. Visual statistics on frames sampled at SAMPLE_FPS (decoded once, by timestamp):
       luma, contrast, saturation, colourfulness, sharpness, entropy, grass fraction,
       score-panel presence (live play vs replay/cutscene), black frames,
       frame-to-frame change, scene cuts, camera pan (optical flow), near-duplicates,
       mean / std image (static HUD map) and hue histogram
  5. Object content on every DET_EVERY-th sample (YOLO11m COCO person / sports ball):
       counts, box sizes and positions

Output (eda/out/):
  metadata.csv  timing.csv  audio.csv  frames.csv  detections.csv
  packets/<match>.npz   images/<match>_{mean,std,sheet}.png   hue/<match>.npy

Usage:
    python eda/video_eda.py               # all matches (skips finished ones)
    python eda/video_eda.py --only M01
    python eda/video_eda.py --no-det      # skip the YOLO pass
"""

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
from camera_benchmark import GREEN_HI, GREEN_LO, UI_TOP_FRAC, on_pitch, pitch_mask  # noqa: E402  same rules

VIDEO_DIR = ROOT / "Screen Recordings" / "Final Recordings"
SPLITS_CSV = ROOT / "dataset" / "splits.csv"
OUT = HERE / "out"
MODEL = ROOT / "benchmark" / "yolo11m.pt"

SAMPLE_FPS = 2.0           # visual statistics
DET_EVERY = 2              # YOLO on every 2nd sample -> 1 fps
W, H = 2400, 1080
SMALL = (600, 270)         # analysis resolution for colour / motion stats
FLOW = (240, 108)          # optical-flow resolution
SHEET_N = 12               # thumbnails per contact sheet

# Same score-panel rule as dataset/sample_frames.py (live play 0.83-0.96, replays ~0).
SCORE_PANEL = (slice(40, 95), slice(440, 610))
MIN_SCORE_PANEL = 0.5
MIN_GREEN_FRAC = 0.30
BLACK_LUMA = 16
CUT_HIST_CORR = 0.6        # HSV histogram correlation below this between samples = cut
DUP_DIFF = 2.0             # mean abs diff of 64x32 thumbnails below this = near-duplicate

IMGSZ = 1920
CONF = 0.15                # = BALL_CONF in sample_frames.py; persons are cut at 0.30 later


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


# --------------------------------------------------------------------------- 1. metadata
def probe(path):
    js = json.loads(run(["ffprobe", "-v", "error", "-print_format", "json",
                         "-show_format", "-show_streams", str(path)]).stdout)
    v = next(s for s in js["streams"] if s["codec_type"] == "video")
    a = next((s for s in js["streams"] if s["codec_type"] == "audio"), {})
    f = js["format"]
    num, den = map(int, v["avg_frame_rate"].split("/"))
    return dict(
        size_mb=round(int(f["size"]) / 2**20, 1),
        container=f["format_name"], duration_s=round(float(f["duration"]), 2),
        total_bitrate_mbps=round(int(f["bit_rate"]) / 1e6, 2),
        creation_time=f.get("tags", {}).get("creation_time", ""),
        android_version=f.get("tags", {}).get("com.android.version", ""),  # the only device tag Android writes
        vcodec=v["codec_name"], profile=v.get("profile", ""), level=v.get("level", ""),
        width=v["width"], height=v["height"], pix_fmt=v["pix_fmt"],
        color_range=v.get("color_range", ""), color_space=v.get("color_space", ""),
        color_transfer=v.get("color_transfer", ""), color_primaries=v.get("color_primaries", ""),
        nb_frames=int(v.get("nb_frames", 0)), avg_fps=round(num / den, 2) if den else 0,
        video_bitrate_mbps=round(int(v.get("bit_rate", 0)) / 1e6, 2),
        has_b_frames=v.get("has_b_frames", 0),
        acodec=a.get("codec_name", "none"), sample_rate=int(a.get("sample_rate", 0)),
        channels=a.get("channels", 0),
        audio_bitrate_kbps=round(int(a.get("bit_rate", 0)) / 1e3, 1),
    )


# --------------------------------------------------------------------------- 2. timing
def packets(path):
    out = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
               "packet=pts_time,size,flags", "-of", "csv=p=0", str(path)]).stdout
    pts, size, key = [], [], []
    for line in out.splitlines():
        p = line.split(",")
        if len(p) < 3 or p[0] in ("", "N/A"):
            continue
        pts.append(float(p[0])); size.append(int(p[1])); key.append("K" in p[2])
    order = np.argsort(pts)
    return (np.array(pts)[order], np.array(size)[order], np.array(key)[order])


def timing_stats(pts, size, key):
    dt = np.diff(pts)
    dt = dt[dt > 0]
    inst = 1.0 / dt
    kf = pts[key]
    gop = np.diff(kf)
    per_s = np.bincount(np.floor(pts - pts[0]).astype(int))
    bits_s = np.bincount(np.floor(pts - pts[0]).astype(int), weights=size) * 8 / 1e6
    return dict(
        n_packets=len(pts), first_pts=round(pts[0], 3),
        fps_mean=round(len(pts) / (pts[-1] - pts[0]), 2),
        fps_p05=round(np.percentile(inst, 5), 1), fps_median=round(np.median(inst), 1),
        fps_p95=round(np.percentile(inst, 95), 1),
        fps_per_sec_min=int(per_s[1:-1].min()), fps_per_sec_max=int(per_s[1:-1].max()),
        dt_ms_median=round(np.median(dt) * 1000, 2), dt_ms_p99=round(np.percentile(dt, 99) * 1000, 1),
        dt_ms_max=round(dt.max() * 1000, 1),
        gaps_over_100ms=int((dt > 0.1).sum()), gaps_over_500ms=int((dt > 0.5).sum()),
        stalled_time_s=round(dt[dt > 0.1].sum(), 2),
        n_keyframes=int(key.sum()), gop_s_median=round(np.median(gop), 2) if len(gop) else 0,
        gop_frames_median=int(np.median(np.diff(np.flatnonzero(key)))) if key.sum() > 1 else 0,
        kb_per_frame_mean=round(size.mean() / 1024, 1),
        kb_keyframe_mean=round(size[key].mean() / 1024, 1),
        kb_interframe_mean=round(size[~key].mean() / 1024, 1),
        mbps_per_sec_p05=round(np.percentile(bits_s, 5), 2),
        mbps_per_sec_p95=round(np.percentile(bits_s, 95), 2),
    ), per_s, bits_s


# --------------------------------------------------------------------------- 3. audio
def audio_stats(path):
    err = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-vn",
                          "-af", "silencedetect=n=-45dB:d=1,volumedetect", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    num = lambda pat: float(m.group(1)) if (m := re.search(pat, err)) else float("nan")
    silence = sum(float(x) for x in re.findall(r"silence_duration: ([\d.]+)", err))
    return dict(mean_volume_db=num(r"mean_volume: ([-\d.]+) dB"),
                max_volume_db=num(r"max_volume: ([-\d.]+) dB"),
                silence_s=round(silence, 1),
                n_silences=len(re.findall(r"silence_end", err)))


# --------------------------------------------------------------------------- 4. visual
def frames_at(path, fps):
    """Yield (t, bgr frame) sampled by timestamp at a constant rate, stretched to W x H
    (a no-op for 2400x1080; other phones' recordings get the HUD where the fixed-pixel tests expect it)."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps={fps},scale={W}:{H}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=W * H * 3 * 2)
    n = W * H * 3
    i = 0
    while True:
        buf = proc.stdout.read(n)
        if len(buf) < n:
            break
        yield i / fps, np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        i += 1
    proc.wait()


def colourfulness(bgr):
    b, g, r = [c.astype(np.float32) for c in cv2.split(bgr)]
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def entropy(gray):
    p = np.bincount(gray.ravel(), minlength=256) / gray.size
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def visual_pass(path, mid, model, det_rows):
    rows = []
    hue_hist = np.zeros(180)
    acc = acc2 = None
    sheet = []
    prev_hist = prev_flow = prev_thumb = None
    for k, (t, frame) in enumerate(frames_at(path, SAMPLE_FPS)):
        small = cv2.resize(frame, SMALL, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        panel = cv2.cvtColor(frame[SCORE_PANEL], cv2.COLOR_BGR2HSV)
        panel_frac = float(((panel[..., 2] > 170) & (panel[..., 1] < 40)).mean())
        play = hsv[int(SMALL[1] * UI_TOP_FRAC):]
        green_frac = float((cv2.inRange(play, GREEN_LO, GREEN_HI) > 0).mean())

        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        hue_hist += np.bincount(hsv[..., 0].ravel(), weights=(hsv[..., 1].ravel() > 40), minlength=180)

        fgray = cv2.resize(gray, FLOW, interpolation=cv2.INTER_AREA)
        thumb = cv2.resize(gray, (64, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        row = dict(
            match_id=mid, t=round(t, 2),
            luma=float(gray.mean()), contrast=float(gray.std()),
            saturation=float(hsv[..., 1].mean()), colourfulness=colourfulness(small),
            sharpness=float(cv2.Laplacian(cv2.cvtColor(frame[200:880, 600:1800], cv2.COLOR_BGR2GRAY),
                                          cv2.CV_64F).var()),
            entropy=entropy(gray), green_frac=green_frac, panel_frac=panel_frac,
            black=bool(gray.mean() < BLACK_LUMA),
            mad=np.nan, hist_corr=np.nan, flow_dx=np.nan, flow_dy=np.nan, flow_mag=np.nan,
            thumb_diff=np.nan,
        )
        if prev_hist is not None:
            row["hist_corr"] = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL))
            row["mad"] = float(np.abs(fgray.astype(np.int16) - prev_flow).mean())
            flow = cv2.calcOpticalFlowFarneback(prev_flow.astype(np.uint8), fgray, None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)
            # median flow ~ camera pan (players are a minority of pixels); px/sample at full res
            sx = W / FLOW[0]
            row["flow_dx"] = float(np.median(flow[..., 0]) * sx)
            row["flow_dy"] = float(np.median(flow[..., 1]) * sx)
            row["flow_mag"] = float(np.median(np.hypot(flow[..., 0], flow[..., 1])) * sx)
            row["thumb_diff"] = float(np.abs(thumb - prev_thumb).mean())
        prev_hist, prev_flow, prev_thumb = hist, fgray.astype(np.int16), thumb

        f32 = cv2.resize(frame, (480, 216), interpolation=cv2.INTER_AREA).astype(np.float64)
        acc = f32 if acc is None else acc + f32
        acc2 = f32 ** 2 if acc2 is None else acc2 + f32 ** 2
        sheet.append((t, cv2.resize(frame, (480, 216), interpolation=cv2.INTER_AREA)))
        if len(sheet) > 4 * SHEET_N:          # keep memory bounded, thin later
            sheet = sheet[::2]

        if model is not None and k % DET_EVERY == 0:
            res = model.predict(frame, imgsz=IMGSZ, conf=CONF, classes=[0, 32], verbose=False)[0]
            _, pitch, _, _ = pitch_mask(frame)
            pitch_inner = cv2.erode(pitch, np.ones((7, 7), np.uint8))
            b = res.boxes
            xyxy = b.xyxy.cpu().numpy(); cls = b.cls.cpu().numpy().astype(int)
            conf = b.conf.cpu().numpy()
            for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf):
                if y2 < H * UI_TOP_FRAC:      # HUD strip
                    continue
                # same on-pitch rule as sample_frames.py: feet for people, centre for the ball
                inside = on_pitch(pitch, (x1 + x2) / 2, y2) if c == 0 else \
                    on_pitch(pitch_inner, (x1 + x2) / 2, (y1 + y2) / 2)
                det_rows.append(dict(match_id=mid, t=round(t, 2), cls="person" if c == 0 else "ball",
                                     conf=round(float(p), 3), on_pitch=bool(inside), cx=(x1 + x2) / 2 / W, cy=(y1 + y2) / 2 / H,
                                     w=(x2 - x1) / W, h=(y2 - y1) / H))
            row["det"] = True
        rows.append(row)

    n = len(rows)
    mean = acc / n
    std = np.sqrt(np.maximum(acc2 / n - mean ** 2, 0))
    idx = np.linspace(0, len(sheet) - 1, SHEET_N).astype(int)
    tiles = []
    for i in idx:
        t, im = sheet[i]
        im = im.copy()
        cv2.putText(im, f"{mid} t={t:.0f}s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(im, f"{mid} t={t:.0f}s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        tiles.append(im)
    grid = np.vstack([np.hstack(tiles[r * 4:(r + 1) * 4]) for r in range(SHEET_N // 4)])
    return rows, hue_hist, mean.astype(np.uint8), std, grid


def label_segments(df):
    """live = score panel shown + grass; replay = grass but no panel; other = cutscene/menu/black."""
    live = (df.panel_frac >= MIN_SCORE_PANEL) & (df.green_frac >= MIN_GREEN_FRAC)
    replay = (df.panel_frac < MIN_SCORE_PANEL) & (df.green_frac >= MIN_GREEN_FRAC)
    df["state"] = np.select([df.black, live, replay], ["black", "live", "replay_or_cutscene"], "other")
    df["scene_cut"] = df.hist_corr < CUT_HIST_CORR
    df["near_dup"] = df.thumb_diff < DUP_DIFF
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--no-det", action="store_true")
    args = ap.parse_args()

    for d in ("packets", "images", "hue", "per_match"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    splits = pd.read_csv(SPLITS_CSV)
    if args.only:
        splits = splits[splits.match_id == args.only]

    model = None
    if not args.no_det:
        from ultralytics import YOLO
        model = YOLO(str(MODEL))

    for r in splits.itertuples():
        done = OUT / "per_match" / f"{r.match_id}.json"
        if done.exists():
            print(r.match_id, "done, skipping"); continue
        path = VIDEO_DIR / r.file
        print(r.match_id, r.file, flush=True)
        meta = dict(match_id=r.match_id, file=r.file, session=r.session, split=r.split, **probe(path))
        pts, size, key = packets(path)
        tstats, per_s, bits_s = timing_stats(pts, size, key)
        np.savez_compressed(OUT / "packets" / f"{r.match_id}.npz", pts=pts, size=size, key=key,
                            frames_per_s=per_s, mbps_per_s=bits_s)
        aud = audio_stats(path)
        det_rows = []
        rows, hue, mean, std, sheet = visual_pass(path, r.match_id, model, det_rows)
        cv2.imwrite(str(OUT / "images" / f"{r.match_id}_mean.png"), mean)
        cv2.imwrite(str(OUT / "images" / f"{r.match_id}_std.png"),
                    cv2.normalize(std.mean(axis=2), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
        cv2.imwrite(str(OUT / "images" / f"{r.match_id}_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        np.save(OUT / "hue" / f"{r.match_id}.npy", hue)
        df = label_segments(pd.DataFrame(rows))
        df.to_csv(OUT / "per_match" / f"{r.match_id}_frames.csv", index=False)
        pd.DataFrame(det_rows).to_csv(OUT / "per_match" / f"{r.match_id}_dets.csv", index=False)
        done.write_text(json.dumps(dict(meta=meta, timing=dict(match_id=r.match_id, **tstats),
                                        audio=dict(match_id=r.match_id, **aud)), default=float))
        print(f"  {len(df)} samples, live {(df.state == 'live').mean():.0%}, {len(det_rows)} dets", flush=True)

    # combine
    js = [json.loads(p.read_text()) for p in sorted((OUT / "per_match").glob("M??.json"))]
    pd.DataFrame([j["meta"] for j in js]).to_csv(OUT / "metadata.csv", index=False)
    pd.DataFrame([j["timing"] for j in js]).to_csv(OUT / "timing.csv", index=False)
    pd.DataFrame([j["audio"] for j in js]).to_csv(OUT / "audio.csv", index=False)
    pd.concat([pd.read_csv(p) for p in sorted((OUT / "per_match").glob("M??_frames.csv"))]) \
        .to_csv(OUT / "frames.csv", index=False)
    dets = [pd.read_csv(p) for p in sorted((OUT / "per_match").glob("M??_dets.csv")) if p.stat().st_size > 2]
    if dets:
        pd.concat(dets).to_csv(OUT / "detections.csv", index=False)
    print("combined ->", OUT)


if __name__ == "__main__":
    main()
