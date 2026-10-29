# Frame sampling plan

Status: **v1 approved 2026-10-25. Implemented in Kitman (`kitman/`) on 2026-10-27; not yet run on the real data.**
Videos are added in Kitman (Videos page) and analysed there with the same EDA pass; the candidate pool below
comes from `kitman/data/analysis/<id>/` rather than the `eda/` tables.
Update this file whenever the plan changes (add a line to the changelog at the bottom).

Inputs this plan relies on: the EDA in `eda/` (dashboard: https://claude.ai/artifact/8aDLJq3NWEeGi4KTbu1o2r),
and the fixed match split in `dataset/splits.csv`. Frames, labels and training runs are tracked by **Kitman**,
our own web app (built fresh; kalydo-marker was only an inspiration and is not used in this project).
Labelling is done in Roboflow.

## Why it looks like this (EDA facts)

- Variable frame rate: per-video mean 17.6–36.4 fps, bursts to ~63, 7–22 freezes > 0.5 s per video.
  Sample by timestamp, never by frame index.
- About 93 live-play minutes in total (train 64.7, val 13.8, test 14.5). Labelling time is the limit, not footage.
- In 0.5 s the camera moves a median 11.6 px (0.5% of the width): 2 fps frames are near-copies.
  Label one frame per ~5 s and spend the rest on distinct hard moments.
- Hard cases: fast pans (fps drops to ~25, blur), ball missed by COCO in ~30% of live frames,
  ball ~13 px, players ~74 px tall at 1080p.
- Sharp group (Laplacian ~50 vs ~38): M05 M08 M10 M14 M19. None in test; M08 in val.

## Naming

Files are named `M05_frame_003704.jpg` (match id + decoded frame index): readable, unique across
matches, and sorted in time order. The exact timestamp lives in `frames.csv`.

## Sampling modes in Kitman

When frames are sampled from a video, Kitman offers two modes:

| | **Custom** (this plan) | **Fixed interval** (as in kalydo-marker) |
|---|---|---|
| What it does | Per-video `plan.json`: windows, train quota buckets, live-play filter, near-duplicate skip, tracking clip | One frame every N seconds, start to end of the whole video |
| Settings | Derived from the video's split and EDA numbers | Just the interval N (seconds) |
| Timing | By timestamp | By timestamp, not kalydo's `sec × round(fps)` frame stepping, which drifts on these variable-frame-rate videos |
| Filtering | Live play only, duplicates skipped | None: every frame at the interval is kept, replays, cutscenes and menus included |

Both write the same output layout and `frames.csv`, with a `mode` column, so later steps don't care which was used.

## Kitman and Roboflow

- Kitman stores the sampled frames and pre-labels, and owns the match-level split (`splits.csv`);
  it builds train/val/test itself, never by per-video percentage.
- Two ways to move a batch to Roboflow, both kept:
  - **A — one click (API):** Kitman uploads the frames with their pre-labels straight to the Roboflow
    project, and pulls the corrected labels back. Needs the owner's Roboflow API key in a local config.
  - **B — one zip:** Kitman produces a single zip in the structure Roboflow imports (YOLO format,
    images + labels + data.yaml); the owner uploads it by hand. Corrected labels come back as a Roboflow
    YOLO export zip, and Kitman strips Roboflow's `_jpg.rf.<hash>` renaming to match each label to its frame.

## Per-video plan

Each video gets its own `plan.json`, derived from its split and its own EDA numbers
(live time, pan-speed quartiles, freeze list, ball-missed rate, sharp group), with a fixed seed.

| | Train (14) | Val + test (3 + 3) |
|---|---|---|
| Window | 5 s | 6 s |
| Budget | live seconds ÷ 5 | live seconds ÷ 6 |
| Pick | by quota | one random live frame |
| Tracking clip | none | one continuous 10 s clip, every frame |

Train quota (scarcest bucket first, given only to windows that contain such a frame; a window that
ends up with no usable frame of its bucket falls back to random):

| Share | Bucket | Rule |
|---|---|---|
| 50% | random | any live frame |
| 20% | fast_pan | top quarter of this video's pan speed |
| 20% | ball_missed | live frame where COCO found no on-pitch ball |
| 10% | edge | first live frame after a freeze, or ≥ 7 players crowded together (GOAL/HALFTIME overlays not detected yet) |

All splits: frames ≥ 2 s apart, near-duplicates skipped.

Tracking clip (val/test): the 10 s live stretch with the fastest camera and no state change.

### Non-live frames (all splits)

On top of the live frames, the custom plan takes about as many non-live frames per video
(ratio 1.0 by default, set on the Sampling page), split between the two kinds by how much of
each the video has:

| Bucket | From | Labelled as |
|---|---|---|
| `replay` | replays and cutscenes (grass, no score panel) | normal frames: box the players, they are still players |
| `menu` | menus, loading screens, full-screen cards, black frames | negatives: empty label file, nothing to box |

Picked ≥ 1 s apart with near-duplicates skipped. Menus barely change, so a video may not have
enough distinct non-live frames to reach the ratio; the plan then takes what exists and records
the shortfall in `plan_run<N>.json` instead of padding with copies.
Decided 2026-10-27 against the 0–10% background guideline, knowingly: watch player and ball
recall on val, and lower the ratio if they drop.

## Candidate pool and frame extraction

1. Candidates = the 2 fps timestamp grid from the EDA (`eda/tables/samples.csv`), live play only
   (score panel + grass), with pan speed and ball counts already computed.
2. Each chosen time snaps to the nearest real frame using `eda/out/packets/<match>.npz`.
   If the nearest frame is > 0.25 s away (inside a freeze), drop it.
3. Decode only the chosen frames, full 2400×1080, JPEG.
4. Pre-label with YOLO11m COCO using the same on-pitch rules as `dataset/sample_frames.py`.

## Output layout

```
kitman/data/sampled/              (this file stays in dataset/sampling/)
  rules.yaml                       global rules + version
  M05/
    frames.csv                     name, t, frame_idx, bucket, mode, run, round
    images/M05_frame_003704.jpg
    plan_run<N>.json               one per sampling run (settings, stats)
  M04/                             val/test: clip frames are M04clip_frame_*.jpg in images/
```

Labels are not stored here: every label set (pre-labels from a model, corrections from Roboflow)
lives in `kitman/data/labels/<set>/`, so several rounds of labels for the same frame can coexist.
Roboflow zips and training runs are in `kitman/data/` too.

## Budget

| | Matches | Frames |
|---|---|---|
| Train | 14 | ~780 |
| Val | 3 | ~140 |
| Test | 3 | ~145 |
| Tracking clips | 6 × 10 s | ~1,500 (label later, with interpolation) |

Detector total ~1,065 frames (was 637).

## Decisions (approved defaults)

1. Existing 637 frames in `dataset/frames/` (uncorrected pre-labels): **regenerate from scratch**.
2. Match-level split: **kept in our repo**; Kitman assembles train/val/test from the corrected labels + `splits.csv`.
3. Tracking clips: **generate now**, label when tracking work starts.

## Later rounds

- Round 2 (after the first fine-tune): run the model at 1 fps over **train matches only**; pick frames with low or
  flickering ball confidence and clustered players; 1–3 frames per failure stretch, ≥ 1 s apart; ~200 frames.
  Appended to the same video's `frames.csv` with `round=2`.
- Report val mAP for M08 (sharp) and M03+M17 separately. If more matches are recorded, put a sharp one in test.

## Changelog

- 2026-10-25 v1: plan approved, defaults accepted. Not implemented; waiting for go-ahead.
- 2026-10-25 v1.1: kalydo-marker removed (inspiration only). Kitman, a fresh web app, tracks frames,
  labels and training; Roboflow transfer by one-click API (A) or a single Roboflow-format zip (B).
  `M05_frame_<idx>` naming kept.
- 2026-10-27 v1.2: two sampling modes in Kitman — Custom (this plan) and Fixed interval (one frame
  every N s over the whole video, by timestamp). Fixed interval applies no filtering: replays,
  cutscenes and menus are kept.
- 2026-10-27 v1.3: implemented in Kitman. Labels stored per label set in kitman/data/labels/ (not a
  prelabels/ folder); per-run plan_run<N>.json instead of one plan.json; quota buckets assigned to
  windows that contain them; edge bucket = after-freeze or crowded players.
- 2026-10-27 v1.4: videos are added through Kitman (folder or upload; split chosen per video, editable until
  labelled); Kitman stores its own copy and runs the EDA pass per video. Non-2400×1080 videos are stretched
  to 2400×1080 frame by frame. Checked: Kitman's analysis of 01.mp4 matches the EDA on all 606 samples.
- 2026-10-27 v1.5: custom plan adds non-live frames, about one per live frame: `replay` (replays and
  cutscenes, labelled normally) and `menu` (menus, loading, black; empty-label negatives).
- 2026-10-27 v1.6: sampled frames moved from dataset/sampling/<id>/ to kitman/data/sampled/<id>/,
  so everything Kitman owns is under kitman/data/. Frame paths in the database are relative, so unchanged.
- 2026-10-27 v1.7: class ids switched to Roboflow's alphabetical order: 0 ball, 1 goalkeeper, 2 player, 3 referee.
