"""Paths and constants shared by every Kitman module."""

import json
from pathlib import Path

KITMAN = Path(__file__).resolve().parent
SPLITS_CSV = KITMAN / "splits.csv"   # only used to suggest splits for the original 20 recordings
COCO_MODEL = KITMAN / "weights" / "yolo11m.pt"   # downloaded by Ultralytics on first use

DATA = KITMAN / "data"            # everything Kitman owns
VIDEOS_DIR = DATA / "videos"      # Kitman's own copy of every added video
ANALYSIS_DIR = DATA / "analysis"  # per-video timestamps + live-play analysis
SAMPLING_DIR = DATA / "sampled"   # sampled frames: <id>/images/, frames.csv, plan_run<N>.json
DB_PATH = DATA / "kitman.db"
LOCAL_CONFIG = KITMAN / "config.local.json"   # Roboflow key etc.; never committed

CLASSES = ["ball", "goalkeeper", "player", "referee"]   # alphabetical, the ids Roboflow assigns: 0 ball … 3 referee
SPLITS = ["train", "val", "test"]
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
FRAME_W, FRAME_H = 2400, 1080     # every frame is stretched to this before analysis and sampling


def local_config():
    try:
        return json.loads(LOCAL_CONFIG.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_local_config(values):
    cfg = local_config() | values
    LOCAL_CONFIG.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg


def roboflow_keys():
    """{workspace: API key}. A Roboflow key reaches exactly one workspace, so each needs its own."""
    cfg = local_config()
    keys = dict(cfg.get("roboflow_keys", {}))
    if cfg.get("roboflow_api_key") and cfg.get("roboflow_workspace"):   # settings saved before several workspaces
        keys.setdefault(cfg["roboflow_workspace"], cfg["roboflow_api_key"])
    return keys


def save_roboflow_keys(keys):
    cfg = {k: v for k, v in local_config().items() if k not in ("roboflow_api_key", "roboflow_workspace")}
    LOCAL_CONFIG.write_text(json.dumps(cfg | {"roboflow_keys": keys}, indent=2), encoding="utf-8")
