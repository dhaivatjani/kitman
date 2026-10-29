# Kitman

Kitman is a local web app that runs the labelling and training loop for the FC Mobile player/ball detector.
It tracks every video, frame, label and model, so you can trace any model back to its training run, the
batches and label sets that run used, and the exact match and second of every frame.

```
add videos → sample frames → batch → pre-label → Roboflow → corrections back → train → next round's pre-labels
```

Detector classes (fixed): `ball`, `goalkeeper`, `player`, `referee`.

---

## Contents

1. [Setup](#setup)
2. [Run it](#run-it)
3. [The pages](#the-pages)
4. [A round, start to finish](#a-round-start-to-finish)
5. [Feature reference](#feature-reference)
6. [Where things are stored](#where-things-are-stored)
7. [Where the numbers come from](#where-the-numbers-come-from)
8. [Code map](#code-map)
9. [Troubleshooting](#troubleshooting)
10. [Not built yet](#not-built-yet)

---

## Setup

This repo is self-contained: clone it anywhere and it runs. It doesn't need anything else from the course project.

### Requirements

- **Python 3.10+**
- **FFmpeg** (`ffmpeg` and `ffprobe` on your PATH), which video analysis uses to read frame timestamps and
  decode frames. On Windows: `winget install Gyan.FFmpeg`, or unzip a build and add its `bin/` to PATH.
  Check with `ffprobe -version`.
- **An NVIDIA GPU with CUDA PyTorch** for training. Analysis and pre-labelling also run on CPU, just slowly.
- **Internet on first use**, to download model weights and to reach Roboflow.

### Install on a new machine

```bash
git clone <this repo> kitman
cd kitman
python -m venv .venv
.venv\Scripts\activate          # Windows (macOS/Linux: source .venv/bin/activate)
```

Install PyTorch with CUDA first, using the command from https://pytorch.org for your CUDA version, e.g.:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Then install the rest:

```bash
pip install -r requirements.txt
```

Check the install:

```bash
python test_kitman.py
```

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

The second command should print `True` if training is going to use your GPU.

### Files git doesn't track

`.gitignore` leaves out secrets, model files and generated data. On a new machine:

| What | How |
|---|---|
| `config.local.json` | Created for you when you save a Roboflow key under **Settings**. Never commit it. |
| `weights/yolo11m.pt` | The COCO model for round-1 pre-labels and video analysis (~40 MB). Ultralytics downloads it the first time it's used. |
| Base weights (`yolo26m.pt` etc.) | Ultralytics downloads these into `kitman/` when a training run starts. Weights you upload go to `data/models/`. |
| `data/` | Created empty on first start. To bring your work along, see *Moving to another machine* under [Where things are stored](#where-things-are-stored). |

---

## Run it

From the `kitman` folder (with the venv active):

```bash
python app.py
```

Open http://127.0.0.1:5050. It only listens on your own machine.

Kitman starts empty. Long jobs (video analysis, sampling, pre-labelling, Roboflow transfers, test evaluation)
run one at a time in a background queue. The bar at the top of the page and the **Activity** page show their
progress. Each training run gets its own process, so you can keep using the app while it trains.

Self-checks for the core logic:

```bash
python test_kitman.py
```

---

## The pages

| Page | What it's for |
|---|---|
| **Squad sheet** (home) | Every match in one view: split, length, live-play timeline, sampled frames as ticks (coloured by status), corrected count. Totals and the best run so far are shown on top. Click a match to see its sample runs and frames. |
| **Videos** | Add videos (folder path or upload), set each one's split, retry a failed analysis, remove a video, open its contact sheet. |
| **Sampling** | Sample frames from all matches, one split, or one video, using the Custom plan or a Fixed interval. Lists recent sample runs. |
| **Frames** | Gallery of every sampled frame (600 per page), filtered by match, split, batch and status. Opening a frame shows its boxes from every label set, the sample run it came from, its batches, the training runs that used it, and prev/next within the match. |
| **Batches** | Make a batch of frames to label. A batch page handles pre-labelling, the Roboflow upload, pulling corrections back, and its label sets. |
| **Training** | Start runs and list past ones. A run's page shows live epochs and the log, plus pause/resume/stop, the settings it used, metrics, and the test evaluation. |
| **Models** | Every model: trained runs, COCO, and uploaded `.pt` files, with val/test metrics and how often each was used for pre-labels. Upload weights here. |
| **Activity** | The last 200 background jobs with status, progress and full log. Look here first when something fails. |
| **Settings** | Roboflow API keys, one per workspace. |

---

## A round, start to finish

### 0. Add videos (Videos)

- **Add a folder**: paste its path and pick a split (train / val / test) for each video. You can also **upload** files.
- Kitman keeps its own copy as `data/videos/M01.mp4`, `M02.mp4`, … That's a hard link on the same drive (no extra
  space), or a real copy otherwise.
- Each video is analysed for live play at 2 fps, about 2 minutes per video. It shows as ready when done.
- Splits for the original 20 recordings are filled in from `splits.csv`. Anything else defaults to train.
- **A split can only change until the video has corrected labels.** Splits are always per whole video, so frames
  from one match never end up in both train and test.
- Videos that aren't 2400×1080 (e.g. Galaxy A35, 2340×1080) are stretched to 2400×1080 frame by frame. The stored
  file is left as it is.

### 1. Sample frames (Sampling)

Pick a scope (all, a split, or one video) and a mode:

- **Custom plan** (the default; see [PLAN.md](PLAN.md)):
  - live play only, at least 2 s apart, near-duplicates avoided
  - train frames are split into quota buckets: 50 % random, 20 % fast camera pan, 20 % ball missed by COCO,
    10 % edge cases (after freezes, crowded players)
  - val/test matches also get a 10 s **tracking clip**, which is kept out of detector training
  - **Non-live frames per live frame** (0–5) adds replays/menus as negatives. **Seed** makes it repeatable.
- **Fixed interval**: one frame every N seconds over the whole video, with nothing filtered out.

The videos have a variable frame rate, so both modes pick timestamps, snap each one to the nearest real frame,
and decode only those frames. Each run writes `images/`, `frames.csv` and `plan_run<N>.json` under
`data/sampled/<match>/`.

### 2. Make a batch (Batches)

Create a batch and say **how many frames** it should hold. Kitman spreads that number evenly across matches.
A frame that's already in a batch is never picked again, so batches never overlap.

### 3. Pre-label (batch page)

**Pre-label** with the COCO model in round 1, or with your newest Kitman model after that. The result is
stored as a new *pre-label* label set on the batch.

### 4. Send to Roboflow (batch page)

1. Pick the Roboflow **workspace** and **project**, or create a new project there. The batch remembers your choice.
2. **Labels** picks which label set goes up with the images. The default is the newest pre-labels.
3. **Send to Roboflow**. This needs that workspace's API key in **Settings**.
   Or click **Build zip for Roboflow** and upload the zip by hand.

You can delete label sets you don't want from the batch page, except ones a training run used.

### 5. Correct in Roboflow

Fix the boxes in Roboflow. The project's classes **must** be named `ball`, `goalkeeper`, `player`, `referee`.
Kitman uses Roboflow's alphabetical ids for them: 0 ball, 1 goalkeeper, 2 player, 3 referee.

### 6. Bring corrections back (batch page)

Generate a version in Roboflow, then either:

- **Pull from Roboflow**: pick the version from the project's version list, or
- export the version as YOLO and use **Import zip**.

Roboflow's `_jpg.rf.<hash>` file renaming is undone and classes are matched by name. The result becomes a
*corrected* label set on the batch.

### 7. Train (Training)

1. Tick the batches to train on and choose each batch's label set. Corrected sets are ticked by default.
2. Pick a **Start from** model (YOLO 26 / 12 / 11 / v8 in sizes n–x; default YOLO26m) and set the basic
   settings: `imgsz` 1280, `epochs` 100, `batch` 4, `patience` 25, `seed` 42.
3. **Advanced settings** (optimiser and learning rate, augmentation, run options, plus free `key=value` extras
   checked against Ultralytics) only pass on what you change.
   **Use football defaults** sets `scale` 0.3 (the ball is about 13 px, so heavy zoom-out loses it) and
   `hsv_s` 0.5 (kit colours matter).
4. Optionally **Copy results to a folder**. When the run finishes, its weights, results, plots and log are copied there too.

Frames go to train, val or test by **the split of the video they came from**, never by percentage.
Tracking-clip frames are left out.

While a run is going:

- **Pause** ends the process and keeps everything. **Resume** carries on from the last finished epoch.
- **Resume** also works on a run that crashed or was cut off (e.g. the PC slept), as long as it saved a `last.pt`.
- **Stop** ends the run and **deletes it completely**: folder, database rows, everything. Finished runs can't be stopped.

A run's page lists every setting it actually used (from Ultralytics' `args.yaml`) and has
**Copy settings to a new run**.

### 8. Next round

When a run finishes, its `best.pt` shows up under **Models** and in the batch **Pre-label** menu. Make the next
batch, pre-label it with that model, and go back to step 4.

### 9. Final number

**Evaluate on test matches** on the run's page. Do this **once, at the end**. Don't tune on the test result.

---

## Feature reference

**Traceability.** Every record points at what it came from: model → training run → label sets → batches →
frames → sample run → match and timestamp. You can follow those links both ways in the UI.

**Frame status.** Each frame is *sampled*, *in batch*, *pre-labelled* or *corrected*. The Squad sheet ticks and
the Frames filters use this.

**Label sets.** A batch can have several label sets: one per pre-label pass and one per correction pull.
Training always uses exactly one set per batch, the one you choose.

**Roboflow.**
- You can have several workspaces. A Roboflow API key only reaches one workspace, so add one key per workspace.
- From inside Kitman you can list projects, create a project, and list a project's versions.
- Keys are saved in `config.local.json`, which git ignores.

**Uploading weights (Models).**
- Upload any YOLO `.pt` file. Kitman loads it once to check it.
- A model whose classes match Kitman's is treated as a Kitman model. Any other model (e.g. COCO) is treated as COCO,
  and its `person` / `sports ball` classes are mapped onto Kitman's.
- Uploaded models appear in both the pre-label list and the training **Start from** list.

**Background jobs.**
- One worker runs them in order, since they all want the GPU or the disk.
- A failed job keeps its full traceback, which you can read on **Activity**.

---

## Where things are stored

Everything Kitman owns is in `kitman/data/` (git ignores it):

| Path | What |
|---|---|
| `kitman.db` | SQLite database that links everything |
| `videos/` | Kitman's copy of every added video, named by id (`M01.mp4`) |
| `analysis/<id>/` | frame timestamps (`packets.npz`), the 2 fps live-play analysis (`samples.csv`), a contact sheet |
| `sampled/<id>/` | sampled frames (`images/`), `frames.csv`, and `plan_run<N>.json` per sampling run |
| `thumbs/` | 480 px gallery thumbnails (a cache, safe to delete) |
| `labels/` | label sets: pre-labels and pulled corrections, as YOLO txt |
| `exports/` | zips built for Roboflow |
| `runs/run<N>/` | a training run: `dataset/` (data.yaml + links), `train/` (Ultralytics output, weights), `train.log`, `frames.csv` |
| `models/` | uploaded `.pt` files |

**Backups:** copy the whole `data/` folder while Kitman is stopped. The database and the files have to stay together.

**Moving to another machine:** the database stores absolute paths (videos, run folders, model files). So copy
`data/` into a Kitman checkout at **the same folder path** as before, or start fresh with an empty `data/`.

---

## Where the numbers come from

A run's metrics are Ultralytics' own final validation of `best.pt`, i.e. the last results table in
`data/runs/run<N>/train.log`. Every frame a run used is listed, with its split, in `data/runs/run<N>/frames.csv`.
Test metrics come from a separate evaluation on the test matches (step 9).

---

## Code map

| File | Role |
|---|---|
| `app.py` | Flask routes and pages |
| `config.py` | paths, classes, splits, frame size, and the local config (Roboflow keys) |
| `db.py` | SQLite schema and query helpers |
| `jobs.py` | the background job queue |
| `analysis.py` | per-video timestamps and live-play analysis (wraps `video_eda.py`) |
| `sampling.py` | Custom plan and Fixed interval sampling |
| `labels.py` | reading/writing YOLO labels, class mapping, pre-labelling |
| `roboflow_io.py` | Roboflow upload, pull, zip build/import, projects and versions |
| `training.py` | building the dataset, train settings, run control (pause/resume/stop), metrics, test evaluation |
| `train_worker.py` | runs one training run in its own process |
| `templates/`, `static/` | HTML templates, CSS and JS |
| `video_eda.py` | the 2 fps live-play analysis; a copy of the project's `eda/video_eda.py` |
| `camera_benchmark.py` | the on-pitch rule used to filter COCO pre-labels; a copy of the project's `benchmark/camera_benchmark.py` |
| `splits.csv` | suggested splits for the original 20 recordings |
| `PLAN.md` | the sampling plan the Custom mode follows |
| `requirements.txt` | pip packages (PyTorch is installed separately) |
| `test_kitman.py` | self-checks |

---

## Troubleshooting

- **A video is stuck or failed.** Check **Activity** for the error, then click **Retry** on the Videos page.
- **"Skipped … still being prepared".** That video's analysis hasn't finished yet. Custom sampling needs the full
  analysis; Fixed interval only needs the timestamps.
- **Send to Roboflow fails.** Make sure that workspace has a key under **Settings**, and that the project's
  classes are named exactly `ball`, `goalkeeper`, `player`, `referee`.
- **A training run shows "running" but nothing happens.** The process probably died (the PC slept or was
  restarted). If the run saved a `last.pt`, it can be resumed.
- **After upgrading Ultralytics,** restart Kitman.
- **Torch/torchvision version warning at startup.** Install the matching `torchvision` for your `torch`.

---

## Not built yet

- Round-2 sampling of the model's failure frames (PLAN.md, "Later rounds"). For now, sample and batch by hand.
- The Custom plan's "edge" bucket uses after-freeze and crowded-player frames. GOAL/HALFTIME overlays aren't
  detected yet.
