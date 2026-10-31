"""Moving batches between Kitman and Roboflow.

  A (API):  upload frames + pre-labels straight into the Roboflow project;
            pull a dataset version back as YOLO and import it.
  B (zip):  one zip in Roboflow's YOLO layout to upload by hand; import the
            YOLO export zip Roboflow gives back.
Both imports go through import_files(), which undoes Roboflow's renaming and
maps class names (not ids, which Roboflow may reorder) onto ours.
"""

import io
import json
import tempfile
from time import sleep
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import yaml

import classes
import db
import labels
from config import DATA, SAMPLING_DIR, roboflow_keys


RF_SPLIT = {"train": "train", "val": "valid", "test": "test"}   # Kitman split -> Roboflow folder / split name


def batch_frames(batch_id):
    return db.q("SELECT f.*, m.split FROM frame f JOIN batch_frame b ON b.frame_id=f.id "
                "JOIN match m ON m.id=f.match_id WHERE b.batch_id=? ORDER BY f.name", (batch_id,))


def latest_labels(batch_id, kind):
    ls = db.q1("SELECT * FROM label_set WHERE batch_id=? AND kind=? ORDER BY id DESC", (batch_id, kind))
    if not ls:
        return None, {}
    return ls, {r["frame_id"]: r for r in db.q("SELECT * FROM label WHERE label_set_id=?", (ls["id"],))}


def data_yaml(splits):
    """Only the splits the batch actually has (a train-only batch gets no val/test entries)."""
    keys = {"train": "train", "valid": "val", "test": "test"}
    return yaml.safe_dump({keys[s]: f"{s}/images" for s in ("train", "valid", "test") if s in splits}
                          | {"nc": len(names := classes.names()), "names": names}, sort_keys=False)


# ---------------------------------------------------------------- B: zip out

def export_zip(job, batch_id):
    b = db.q1("SELECT * FROM batch WHERE id=?", (batch_id,))
    frames = batch_frames(batch_id)
    ls, by_frame = latest_labels(batch_id, "prelabel")
    out = DATA / "exports" / f"kitman_batch{batch_id}_round{b['round']}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    splits = {RF_SPLIT[f["split"]] for f in frames}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:   # JPEGs don't compress further
        z.writestr("data.yaml", data_yaml(splits))
        for k, f in enumerate(frames):
            folder = RF_SPLIT[f["split"]]   # each frame goes where its video's split says
            z.write(SAMPLING_DIR / f["path"], f"{folder}/images/{f['name']}")
            row = by_frame.get(f["id"])
            text = (DATA / row["path"]).read_text(encoding="utf-8") if row else ""
            z.writestr(f"{folder}/labels/{Path(f['name']).stem}.txt", text)
            if k % 20 == 0:
                job.progress((k + 1) / len(frames), f"Packed {k + 1} of {len(frames)}")
    db.x("UPDATE batch SET status='sent to Roboflow' WHERE id=? AND status IN ('new','pre-labelled')", (batch_id,))
    with_labels = f"with pre-labels from {ls['source']}" if ls else "without pre-labels"
    per = ", ".join(f"{sum(RF_SPLIT[f['split']] == s for f in frames)} {s}" for s in ("train", "valid", "test") if s in splits)
    return f"Zip ready: {len(frames)} frames ({per}) {with_labels}"


# ---------------------------------------------------------------- import (A and B)

def import_files(batch_id, files, source):
    """files: {path inside export: bytes}. Creates a 'corrected' label set for the batch."""
    yamls = [p for p in files if p.endswith((".yaml", ".yml"))]
    if not yamls:
        raise ValueError("No data.yaml in the export. Export from Roboflow in a YOLO format.")
    names = yaml.safe_load(files[min(yamls, key=lambda p: p.count("/"))]).get("names", [])
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else names
    remap, unknown = classes.read_names(dict(enumerate(names)))   # merged names land on their new class
    if unknown:
        raise ValueError(f"Roboflow classes not in Kitman: {', '.join(unknown)}. "
                         f"Rename them in Roboflow to one of: {', '.join(classes.names())}, or add them on the Classes page.")

    frames = {Path(f["name"]).stem: f for f in batch_frames(batch_id)}
    ls = db.x("INSERT INTO label_set (batch_id, kind, source, created) VALUES (?,?,?,?)",
              (batch_id, "corrected", source, db.now()))
    rows, outside, seen = [], [], set()
    for path, data in files.items():
        if "/labels/" not in "/" + path or not path.endswith(".txt"):
            continue
        stem = labels.frame_stem(path)
        f = frames.get(stem)
        if f is None:
            outside.append(stem)
            continue
        try:
            boxes = labels.parse(data.decode("utf-8"))
        except ValueError as e:
            raise ValueError(f"{Path(path).name}: {e}") from None
        bad = {c for c, *_ in boxes if c not in remap}
        if bad:
            raise ValueError(f"{Path(path).name}: class ids {sorted(bad)} are not in data.yaml")
        # boxes of a deleted class (remap None) are dropped on arrival
        rows.append(labels.write(ls, f, [(remap[c], *v) for c, *v in boxes if remap[c] is not None]))
        seen.add(stem)
    if not rows:
        db.x("DELETE FROM label_set WHERE id=?", (ls,))
        raise ValueError("No label in the export matches a frame in this batch. "
                         "Check you exported the right Roboflow project or version.")
    labels.save_rows(rows)
    db.x("UPDATE batch SET status='corrected' WHERE id=?", (batch_id,))
    missing = len(frames) - len(seen)
    msg = f"Imported {len(rows)} corrected frames"
    if missing:
        msg += f"; {missing} batch frames were not in the export"
    if outside:
        msg += f"; ignored {len(outside)} labels for frames outside this batch"
    return msg


def import_zip(job, batch_id, zip_bytes, filename):
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:   # read members as data, never extract paths
        files = {i.filename: z.read(i) for i in z.infolist()
                 if not i.is_dir() and i.filename.endswith((".txt", ".yaml", ".yml"))}
    return import_files(batch_id, files, f"Roboflow zip ({filename})")


# ---------------------------------------------------------------- A: API

API = "https://api.roboflow.com"


def key_for(ws):
    key = roboflow_keys().get(ws)
    if not key:
        raise ValueError(f"No API key saved for workspace {ws}. Add it in Settings.")
    return key


def refusal(r):
    try:
        err = r.json()["error"]
        return ValueError(f"Roboflow said: {err['message'] if isinstance(err, dict) else err}")
    except (ValueError, KeyError, TypeError):
        return ValueError(f"Roboflow said: {r.text[:300]}")


def get(path, key):
    r = requests.get(f"{API}/{path}", params={"api_key": key}, timeout=30)
    if r.status_code != 200:
        raise refusal(r)
    return r.json()


def workspace_of(key):
    """The one workspace an API key belongs to."""
    return get("", key)["workspace"]


def list_projects(ws):
    projects = get(ws, key_for(ws))["workspace"]["projects"]
    return [{"id": p["id"].split("/")[-1], "name": p["name"], "images": p.get("images", 0)}
            for p in projects if p.get("type") == "object-detection"]


def list_versions(ws, proj):
    vs = get(f"{ws}/{proj}", key_for(ws)).get("versions", [])
    return sorted(({"n": int(v["id"].rsplit("/", 1)[1]), "name": v.get("name", ""), "images": v.get("images", 0),
                    "ready": not v.get("generating")} for v in vs), key=lambda v: -v["n"])


def create_project(ws, name):
    url = f"{API}/{ws}/projects"
    for license in ("Private", "CC BY 4.0"):   # Private is a paid feature; free workspaces only take public licences
        r = requests.post(url, params={"api_key": key_for(ws)}, timeout=30,
                          json={"name": name, "type": "object-detection", "license": license, "annotation": name})
        if r.ok and "error" not in r.json():
            return {"id": r.json()["id"].split("/")[-1], "name": r.json()["name"], "images": 0}
    raise refusal(r)


def project(ws, proj):
    if not (ws and proj):
        raise ValueError("Choose a Roboflow workspace and project for this batch first.")
    from roboflow import Roboflow
    return Roboflow(api_key=key_for(ws)).workspace(ws).project(proj)


def dropped(e):
    """A dropped connection or timeout anywhere in the chain (the SDK wraps them in its own errors)."""
    while e:
        if isinstance(e, (ConnectionError, requests.ConnectionError, requests.Timeout)):
            return True
        e = e.__cause__ or e.__context__
    return False


def api_upload(job, batch_id, label_set_id):
    """label_set_id: which of the batch's label sets goes up with the images; None sends images only."""
    b = db.q1("SELECT * FROM batch WHERE id=?", (batch_id,))
    frames = batch_frames(batch_id)
    by_frame = {r["frame_id"]: r for r in db.q("SELECT * FROM label WHERE label_set_id=?", (label_set_id,))}
    proj = project(b["rf_workspace"], b["rf_project"])
    labelmap = dict(enumerate(classes.names()))

    def send(f):
        row = by_frame.get(f["id"])
        ann = DATA / row["path"] if row and sum(json.loads(row["counts"]).values()) else None
        # The SDK never retries label saves, so retry the whole frame on a dropped connection;
        # Roboflow keeps one copy of a re-sent image, and labels already there come back as a 409.
        for attempt in range(4):
            try:
                proj.single_upload(image_path=str(SAMPLING_DIR / f["path"]),
                                   annotation_path=str(ann) if ann else None,
                                   annotation_labelmap=labelmap if ann else None,
                                   split=RF_SPLIT[f["split"]],
                                   batch_name=f"Kitman batch {batch_id} (round {b['round']})",
                                   tag_names=[f"kitman-batch-{batch_id}", f"round-{b['round']}", f["match_id"]],
                                   num_retry_uploads=2)
                return True
            except AttributeError:
                # shortcut: roboflow 1.2.11 crashes this way only on a 409 (image already annotated, e.g. a
                # re-sent batch); we keep Roboflow's labels rather than overwrite corrections. Drop when the SDK is fixed.
                return False
            except Exception as e:  # anything but a dropped connection, or the 4th drop, fails the job
                if attempt == 3 or not dropped(e):
                    raise
                sleep(2 ** attempt)   # 1, 2, 4 s

    # each upload is mostly waiting on Roboflow, so send several at once
    pool = ThreadPoolExecutor(max_workers=8)
    try:
        skipped = 0
        for k, fut in enumerate(as_completed([pool.submit(send, f) for f in frames])):
            skipped += not fut.result()   # re-raise a failed upload so the job reports it
            job.progress((k + 1) / len(frames), f"Uploaded {k + 1} of {len(frames)}")
    finally:
        pool.shutdown(cancel_futures=True)   # on a failure, stop instead of uploading the rest unseen
    db.x("UPDATE batch SET status='sent to Roboflow' WHERE id=? AND status IN ('new','pre-labelled')", (batch_id,))
    db.x("UPDATE batch SET rf_sent_to=? WHERE id=?", (b["rf_project"], batch_id))
    msg = f"Uploaded {len(frames)} frames to Roboflow project {b['rf_workspace']}/{b['rf_project']}"
    return msg + (f"; {skipped} were already labelled there and kept their Roboflow labels" if skipped else "")


def api_pull(job, batch_id, version):
    b = db.q1("SELECT * FROM batch WHERE id=?", (batch_id,))
    proj = project(b["rf_workspace"], b["rf_project"])
    job.progress(0.1, f"Downloading version {version} from Roboflow")
    with tempfile.TemporaryDirectory() as tmp:
        proj.version(int(version)).download("yolov8", location=tmp, overwrite=True)
        root = Path(tmp)
        files = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*")
                 if p.is_file() and p.suffix in (".txt", ".yaml", ".yml")}
    return import_files(batch_id, files, f"Roboflow API ({b['rf_project']} version {version})")
