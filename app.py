"""Kitman: tracks every frame, label and model in the FC Mobile detector loop.

    python kitman/app.py      then open http://127.0.0.1:5050
"""

import csv
import json
import random
import shutil
from pathlib import Path

from flask import Flask, abort, flash, jsonify, redirect, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename

import analysis
import db
import jobs
import labels
import roboflow_io
import sampling
import training
from config import (ANALYSIS_DIR, CLASSES, DATA, SAMPLING_DIR, SPLITS, SPLITS_CSV, VIDEO_EXTS, VIDEOS_DIR,
                    roboflow_keys, save_roboflow_keys)

app = Flask(__name__)
app.secret_key = "kitman-local"   # only signs flash messages; the app listens on localhost only
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024**3   # video uploads (~1 GB each) and Roboflow zips


@app.template_filter("fromjson")
def fromjson(s):
    return json.loads(s) if s else None


@app.template_filter("mmss")
def mmss(t):
    t = float(t or 0)
    return f"{int(t // 60)}:{t % 60:04.1f}"


@app.context_processor
def globals_():
    active = db.q("SELECT * FROM job WHERE status IN ('queued','running') ORDER BY id")
    # an edited file gets a new URL, so browsers never keep running a stale cached copy
    asset = lambda f: url_for("static", filename=f, v=int((Path(app.static_folder) / f).stat().st_mtime))
    return {"CLASSES": CLASSES, "active_jobs": active, "nav": request.endpoint, "asset": asset}


# ---------------------------------------------------------------- shared queries

_segments = {}


def segments(mid):
    """Run-length states from a video's 2 fps analysis: [(start, end, state)], cached until it changes."""
    p = ANALYSIS_DIR / mid / "samples.csv"
    if not p.exists():
        return []
    key = (mid, p.stat().st_mtime)
    if key not in _segments:
        d = analysis.samples(mid).sort_values("t")
        segs, start, state = [], None, None
        for t, st in zip(d.t, d.state):
            if st != state:
                if state is not None:
                    segs.append((start, t, state))
                start, state = t, st
        segs.append((start, float(d.t.iloc[-1]) + 0.5, state))
        _segments[key] = segs
    return _segments[key]


FRAME_STATUS = """
  CASE WHEN EXISTS (SELECT 1 FROM label l JOIN label_set s ON s.id=l.label_set_id
                    WHERE l.frame_id=f.id AND s.kind='corrected') THEN 'corrected'
       WHEN EXISTS (SELECT 1 FROM label l WHERE l.frame_id=f.id) THEN 'pre-labelled'
       WHEN EXISTS (SELECT 1 FROM batch_frame b WHERE b.frame_id=f.id) THEN 'in batch'
       ELSE 'sampled' END"""


def frames_where(args):
    where, vals = ["1=1"], []
    for col, key in (("f.match_id", "match"), ("m.split", "split"), ("r.mode", "mode"), ("f.bucket", "bucket")):
        if args.get(key):
            where.append(f"{col}=?")
            vals.append(args[key])
    if args.get("batch"):
        where.append("f.id IN (SELECT frame_id FROM batch_frame WHERE batch_id=?)")
        vals.append(args["batch"])
    sql = f"""SELECT f.*, m.split, r.mode, {FRAME_STATUS} AS status FROM frame f
              JOIN match m ON m.id=f.match_id JOIN sample_run r ON r.id=f.sample_run_id
              WHERE {' AND '.join(where)}"""
    # Always wrapped, so callers can filter on status and sort by plain column names.
    sql = f"SELECT * FROM ({sql})" + (" WHERE status=?" if args.get("status") else "")
    if args.get("status"):
        vals.append(args["status"])
    return sql, vals


def match_rows():
    rows = db.q(f"""SELECT m.*,
        (SELECT COUNT(*) FROM frame f WHERE f.match_id=m.id) AS n_frames,
        (SELECT COUNT(*) FROM frame f WHERE f.match_id=m.id AND ({FRAME_STATUS})='corrected') AS n_corrected
        FROM match m ORDER BY m.id""")
    ticks = db.q(f"SELECT f.match_id, f.t, f.bucket, {FRAME_STATUS} AS status FROM frame f")
    by = {}
    for t in ticks:
        by.setdefault(t["match_id"], []).append(t)
    for r in rows:
        r["segments"], r["ticks"] = segments(r["id"]), by.get(r["id"], [])
        r["live_s"] = sum(e - s for s, e, st in r["segments"] if st == "live")
        r["has_pts"] = (ANALYSIS_DIR / r["id"] / "packets.npz").exists()
        r["split_locked"] = r["n_corrected"] > 0
    return rows


# ---------------------------------------------------------------- pages

@app.route("/")
def overview():
    totals = db.q1(f"""SELECT COUNT(*) AS frames,
        SUM(({FRAME_STATUS})='corrected') AS corrected,
        (SELECT COUNT(*) FROM batch) AS batches,
        (SELECT COUNT(*) FROM training_run WHERE status='done') AS runs FROM frame f""")
    best = db.q1("SELECT * FROM training_run WHERE status='done' ORDER BY "
                 "json_extract(metrics,'$.overall.map50') DESC LIMIT 1")
    return render_template("overview.html", matches=match_rows(), totals=totals, best=best)


@app.route("/matches/<mid>")
def match(mid):
    m = next((r for r in match_rows() if r["id"] == mid), None) or abort(404)
    runs = db.q("SELECT * FROM sample_run WHERE match_id=? ORDER BY id", (mid,))
    sql, vals = frames_where({"match": mid})
    frames = db.q(sql + " ORDER BY t", vals)
    return render_template("match.html", m=m, runs=runs, frames=frames)


@app.route("/sampling")
def sampling_page():
    runs = db.q("SELECT * FROM (SELECT * FROM sample_run ORDER BY id DESC LIMIT 100) ORDER BY id")   # latest 100, oldest first
    return render_template("sampling.html", matches=match_rows(), runs=runs, pick=request.args.get("match", "all"),
                           custom=sampling.CUSTOM_DEFAULTS, fixed=sampling.FIXED_DEFAULTS)


@app.post("/matches/sample")
def sample():
    mode = request.form["mode"]
    scope = request.form.get("scope", "all")
    if scope == "all":
        ids = [r["id"] for r in db.q("SELECT id FROM match ORDER BY id")]
    elif scope in SPLITS:
        ids = [r["id"] for r in db.q("SELECT id FROM match WHERE split=? ORDER BY id", (scope,))]
    else:   # a single video
        ids = [r["id"] for r in db.q("SELECT id FROM match WHERE id=?", (scope,))]
    # Fixed interval only needs frame timestamps; the custom plan needs the full analysis.
    ready = [i for i in ids if (ANALYSIS_DIR / i / ("samples.csv" if mode == "custom" else "packets.npz")).exists()]
    if len(ready) < len(ids):
        flash(f"Skipped {', '.join(sorted(set(ids) - set(ready)))}: still being prepared on the Videos page.", "error")
    ids = ready
    if not ids:
        flash("No videos ready to sample. Add videos and wait until they show as ready.", "error")
        return redirect(url_for("sampling_page", match=scope))
    if mode == "fixed":
        try:
            interval = float(request.form["interval_s"])
            assert interval > 0
        except (ValueError, AssertionError):
            flash("Interval must be a number of seconds above 0.", "error")
            return redirect(url_for("sampling_page", match=scope))
        params = {"interval_s": interval}
    else:
        try:
            ratio = float(request.form.get("nonlive_ratio") or sampling.CUSTOM_DEFAULTS["nonlive_ratio"])
            assert 0 <= ratio <= 5
        except (ValueError, AssertionError):
            flash("Non-live frames per live frame must be a number from 0 to 5.", "error")
            return redirect(url_for("sampling_page", match=scope))
        params = sampling.CUSTOM_DEFAULTS | {"seed": int(request.form.get("seed") or 42), "nonlive_ratio": ratio}
    for mid in ids:
        rid = db.x("INSERT INTO sample_run (match_id, mode, params, status, created) VALUES (?,?,?,?,?)",
                   (mid, mode, db.jdump(params), "queued", db.now()))
        how = "custom plan" if mode == "custom" else f"every {params['interval_s']:g} s"
        jobs.submit("sample", f"Sample {mid} ({how})", sampling.run, rid)
    flash(f"Sampling queued for {len(ids)} match{'es' if len(ids) > 1 else ''}.", "ok")
    return redirect(url_for("sampling_page", match=scope))


@app.route("/frames")
def frames():
    args = request.args
    sql, vals = frames_where(args)
    rows = db.q(sql + " ORDER BY match_id, t LIMIT 600", vals)
    total = db.q1(f"SELECT COUNT(*) AS n FROM ({sql})", vals)["n"]
    return render_template("frames.html", frames=rows, total=total, args=args,
                           matches=db.q("SELECT id, split FROM match ORDER BY id"),
                           batches=db.q("SELECT id, name FROM batch ORDER BY id DESC"))


@app.route("/frames/<int:fid>")
def frame(fid):
    sql, vals = frames_where({})
    f = db.q1(f"SELECT * FROM ({sql}) WHERE id=?", vals + [fid]) or abort(404)
    sets = db.q("""SELECT l.*, s.kind, s.source, s.created, s.batch_id FROM label l
                   JOIN label_set s ON s.id=l.label_set_id WHERE l.frame_id=? ORDER BY s.id DESC""", (fid,))
    for s in sets:
        s["boxes"] = labels.read(s)
    run = db.q1("SELECT * FROM sample_run WHERE id=?", (f["sample_run_id"],))
    batches = db.q("SELECT b.* FROM batch b JOIN batch_frame bf ON bf.batch_id=b.id WHERE bf.frame_id=?", (fid,))
    used_in = [r for r in db.q("SELECT id, name, label_set_ids FROM training_run")
               if set(json.loads(r["label_set_ids"])) & {s["label_set_id"] for s in sets}]
    nav = db.q("SELECT id FROM frame WHERE match_id=? ORDER BY t", (f["match_id"],))
    ids = [n["id"] for n in nav]
    k = ids.index(fid)
    return render_template("frame.html", f=f, sets=sets, run=run, batches=batches, used_in=used_in,
                           prev_id=ids[k - 1] if k else None, next_id=ids[k + 1] if k + 1 < len(ids) else None)


@app.route("/img/<path:p>")
def img(p):
    path = (SAMPLING_DIR / p).resolve()
    if SAMPLING_DIR.resolve() not in path.parents or not path.is_file():
        abort(404)
    return send_file(path, max_age=86400)


@app.route("/thumb/<path:p>")
def thumb(p):
    """480 px wide cached copy for galleries."""
    import cv2
    src = (SAMPLING_DIR / p).resolve()
    if SAMPLING_DIR.resolve() not in src.parents or not src.is_file():
        abort(404)
    out = DATA / "thumbs" / p
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        im = cv2.imread(str(src))
        cv2.imwrite(str(out), cv2.resize(im, (480, round(480 * im.shape[0] / im.shape[1])),
                                         interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return send_file(out, max_age=86400)


# ---------------------------------------------------------------- videos

def suggested(file_name):
    """Split for a file from the original 20-recording split, else train."""
    if SPLITS_CSV.exists():
        with open(SPLITS_CSV, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["file"] == file_name:
                    return r["split"]
    return "train"


@app.route("/videos")
def videos():
    return render_template("videos.html", matches=match_rows(), scan=None, folder="")


@app.post("/videos/pick")
def videos_pick():
    """Open the Windows folder picker on this computer (a browser can't reveal local folder paths)."""
    import tkinter
    from tkinter import filedialog
    root = tkinter.Tk()
    root.withdraw()
    root.attributes("-topmost", True)   # otherwise it opens behind the browser
    folder = filedialog.askdirectory(parent=root, title=request.args.get("title", "Choose the folder with your match recordings"))
    root.destroy()
    return jsonify(folder=folder or "")


@app.post("/videos/scan")
def videos_scan():
    folder = request.form.get("folder", "").strip().strip('"')
    p = Path(folder)
    if not folder or not p.is_dir():
        flash(f"“{folder}” isn't a folder on this computer. Use Choose folder to pick it again.", "error")
        return redirect(url_for("videos"))
    have = {(r["file"], r["frames"]) for r in db.q("SELECT file, frames FROM match")}
    scan = []
    for f in sorted(x for x in p.iterdir() if x.is_file() and x.suffix.lower() in VIDEO_EXTS):
        try:
            info = analysis.probe(f)
        except ValueError as e:
            scan.append({"file": f.name, "error": str(e)})
            continue
        split = suggested(f.name)
        scan.append(info | {"file": f.name, "size_mb": round(f.stat().st_size / 2**20), "split": split,
                            "added": (f.name, info["frames"]) in have})
    if not scan:
        flash("No video files in that folder.", "error")
        return redirect(url_for("videos"))
    return render_template("videos.html", matches=match_rows(), scan=scan, folder=str(p))


@app.post("/videos/folder")
def videos_add_folder():
    folder = Path(request.form["folder"])
    files = request.form.getlist("file")
    if not files:
        flash("Tick at least one video to add.", "error")
        return redirect(url_for("videos"))
    added = []
    for name in files:
        src = folder / name
        split = request.form.get(f"split::{name}")
        if src.parent != folder or not src.is_file() or split not in SPLITS:
            flash(f"Skipped {name}: file or split not valid.", "error")
            continue
        vid = analysis.register(name, "", split, "")
        jobs.submit("video", f"Add {vid} ({name})", analysis.prepare, vid, str(src))
        added.append(vid)
    if added:
        flash(f"Adding {len(added)} video{'s' if len(added) > 1 else ''} ({added[0]}–{added[-1]}). "
              "Each is stored in Kitman, then analysed (about 2 minutes per video).", "ok")
    return redirect(url_for("videos"))


@app.post("/videos/upload")
def videos_upload():
    files = [f for f in request.files.getlist("video") if f.filename]
    split = request.form.get("split")
    if not files or split not in SPLITS:
        flash("Choose at least one video and a split.", "error")
        return redirect(url_for("videos"))
    bad = [f.filename for f in files if Path(f.filename).suffix.lower() not in VIDEO_EXTS]
    if bad:
        flash(f"Not a supported video type: {', '.join(bad)}. Use {', '.join(sorted(VIDEO_EXTS))}.", "error")
        return redirect(url_for("videos"))
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    added = []
    for f in files:
        name = secure_filename(f.filename) or "video.mp4"
        vid = analysis.register(name, "", split, request.form.get("description", "").strip())
        dst = VIDEOS_DIR / f"{vid}{Path(name).suffix.lower()}"
        f.save(dst)
        db.x("UPDATE match SET path=? WHERE id=?", (str(dst), vid))
        jobs.submit("video", f"Analyse {vid} ({name})", analysis.prepare, vid)
        added.append(vid)
    flash(f"Uploaded {', '.join(added)}. Analysis is running; Custom plan sampling unlocks when it's done.", "ok")
    return redirect(url_for("videos"))


@app.post("/videos/<mid>/split")
def video_split(mid):
    m = next((r for r in match_rows() if r["id"] == mid), None) or abort(404)
    split = request.form.get("split")
    if split not in SPLITS:
        abort(400)
    if m["split_locked"]:
        flash(f"{mid} has corrected labels, so its split is locked. Moving it would leak labelled frames "
              "between train and test.", "error")
    else:
        db.x("UPDATE match SET split=? WHERE id=?", (split, mid))
        flash(f"{mid} is now in {split}.", "ok")
    return redirect(request.referrer or url_for("videos"))


@app.post("/videos/<mid>/retry")
def video_retry(mid):
    m = db.q1("SELECT * FROM match WHERE id=? AND status='failed'", (mid,)) or abort(404)
    if not Path(m["path"] or "").is_file():
        flash(f"{mid}'s stored file is missing. Remove it and add the video again.", "error")
        return redirect(url_for("videos"))
    db.x("UPDATE match SET status='queued', message=NULL WHERE id=?", (mid,))
    jobs.submit("video", f"Analyse {mid} again", analysis.prepare, mid)
    return redirect(url_for("videos"))


@app.post("/videos/<mid>/remove")
def video_remove(mid):
    m = db.q1("SELECT * FROM match WHERE id=?", (mid,)) or abort(404)
    if db.q1("SELECT 1 FROM frame WHERE match_id=? LIMIT 1", (mid,)) or db.q1("SELECT 1 FROM sample_run WHERE match_id=? LIMIT 1", (mid,)):
        flash(f"{mid} already has sampled frames, so it stays: removing it would break their history.", "error")
        return redirect(url_for("videos"))
    if m["status"] in ("queued", "copying", "analysing"):
        flash(f"{mid} is still being prepared. Remove it once that finishes.", "error")
        return redirect(url_for("videos"))
    if m["path"] and Path(m["path"]).parent == VIDEOS_DIR:   # only ever delete Kitman's own copy
        Path(m["path"]).unlink(missing_ok=True)
    shutil.rmtree(ANALYSIS_DIR / mid, ignore_errors=True)
    db.x("DELETE FROM match WHERE id=?", (mid,))
    flash(f"Removed {mid} ({m['file']}). The original file wasn't touched.", "ok")
    return redirect(url_for("videos"))


@app.route("/videos/<mid>/sheet.jpg")
def video_sheet(mid):
    p = ANALYSIS_DIR / secure_filename(mid) / "sheet.jpg"
    return send_file(p, max_age=3600) if p.exists() else abort(404)


# ---------------------------------------------------------------- batches

@app.route("/batches")
def batches():
    rows = db.q(f"""SELECT b.*, COUNT(bf.frame_id) AS n,
        SUM(CASE WHEN m.split='train' THEN 1 ELSE 0 END) AS n_train,
        SUM(CASE WHEN m.split='val' THEN 1 ELSE 0 END) AS n_val,
        SUM(CASE WHEN m.split='test' THEN 1 ELSE 0 END) AS n_test
        FROM batch b LEFT JOIN batch_frame bf ON bf.batch_id=b.id
        LEFT JOIN frame f ON f.id=bf.frame_id LEFT JOIN match m ON m.id=f.match_id
        GROUP BY b.id ORDER BY b.id DESC""")
    free = db.q1(f"""SELECT COUNT(*) AS n FROM frame f WHERE f.id NOT IN (SELECT frame_id FROM batch_frame)""")["n"]
    next_round = (db.q1("SELECT MAX(round) AS r FROM batch")["r"] or 0) + 1
    return render_template("batches.html", batches=rows, free=free, next_round=next_round,
                           matches=db.q("SELECT id, split, status FROM match ORDER BY id"))


@app.post("/batches/new")
def batch_new():
    form = request.form
    name = form.get("name", "").strip()
    if not name:
        flash("Give the batch a name.", "error")
        return redirect(url_for("batches"))
    picked = form.getlist("match")
    if not picked:
        flash("Tick at least one video in the Videos list.", "error")
        return redirect(url_for("batches"))
    where = ["f.id NOT IN (SELECT frame_id FROM batch_frame)", f"f.match_id IN ({','.join('?' * len(picked))})"]
    vals = list(picked)
    if form.get("mode"):
        where.append("r.mode=?")
        vals.append(form["mode"])
    if not form.get("include_clips"):
        where.append("f.bucket != 'clip'")
    cands = db.q(f"""SELECT f.id, f.match_id FROM frame f JOIN match m ON m.id=f.match_id
                     JOIN sample_run r ON r.id=f.sample_run_id WHERE {' AND '.join(where)} ORDER BY f.id""", vals)
    if not cands:
        flash("No unbatched frames match those filters. Sample more frames first.", "error")
        return redirect(url_for("batches"))
    limit = int(form.get("limit") or 0)
    picked = spread(cands, limit, seed=int(form.get("seed") or 0)) if limit else cands
    bid = db.x("INSERT INTO batch (name, round, notes, status, created) VALUES (?,?,?,?,?)",
               (name, int(form.get("round") or 1), form.get("notes", ""), "new", db.now()))
    db.xmany("INSERT INTO batch_frame VALUES (?,?)", [(bid, c["id"]) for c in picked])
    flash(f"Batch “{name}” created with {len(picked)} frames.", "ok")
    return redirect(url_for("batch", bid=bid))


def spread(cands, limit, seed):
    """Up to `limit` frames, taken round-robin across matches so every match is represented."""
    rng = random.Random(seed)
    by = {}
    for c in cands:
        by.setdefault(c["match_id"], []).append(c)
    for v in by.values():
        rng.shuffle(v)
    out, queues = [], [by[k] for k in sorted(by)]
    while len(out) < limit and any(queues):
        for qq in queues:
            if qq and len(out) < limit:
                out.append(qq.pop())
    return out


@app.route("/batches/<int:bid>")
def batch(bid):
    b = db.q1("SELECT * FROM batch WHERE id=?", (bid,)) or abort(404)
    sql, vals = frames_where({"batch": bid})
    rows = db.q(sql + " ORDER BY match_id, t", vals)
    sets = db.q("""SELECT s.*, COUNT(l.frame_id) AS n, SUM(l.n_player) AS n_player, SUM(l.n_goalkeeper) AS n_goalkeeper,
                   SUM(l.n_referee) AS n_referee, SUM(l.n_ball) AS n_ball FROM label_set s
                   LEFT JOIN label l ON l.label_set_id=s.id WHERE s.batch_id=? GROUP BY s.id ORDER BY s.id""", (bid,))
    exports = sorted((DATA / "exports").glob(f"kitman_batch{bid}_*.zip")) if (DATA / "exports").exists() else []
    by_match = {}
    for r in rows:
        by_match[r["match_id"]] = by_match.get(r["match_id"], 0) + 1
    trained = {i for r in db.q("SELECT label_set_ids FROM training_run") for i in json.loads(r["label_set_ids"])}
    return render_template("batch.html", b=b, frames=rows, sets=sets, exports=exports, by_match=by_match, trained=trained,
                           models=db.q("SELECT * FROM model ORDER BY id DESC"),
                           workspaces=sorted(roboflow_keys()))


@app.post("/batches/<int:bid>/prelabel")
def batch_prelabel(bid):
    mid = int(request.form["model_id"])
    name = db.q1("SELECT name FROM model WHERE id=?", (mid,))["name"]
    jobs.submit("prelabel", f"Pre-label batch {bid} with {name}", labels.prelabel, bid, mid)
    flash("Pre-labelling queued.", "ok")
    return redirect(url_for("batch", bid=bid))


@app.post("/batches/<int:bid>/zip")
def batch_zip(bid):
    jobs.submit("export", f"Build Roboflow zip for batch {bid}", roboflow_io.export_zip, bid)
    flash("Building the zip. It appears under Roboflow when ready.", "ok")
    return redirect(url_for("batch", bid=bid))


@app.route("/batches/<int:bid>/zip/<name>")
def batch_zip_download(bid, name):
    p = (DATA / "exports" / name).resolve()
    if p.parent != (DATA / "exports").resolve() or not p.name.startswith(f"kitman_batch{bid}_") or not p.exists():
        abort(404)
    return send_file(p, as_attachment=True)


@app.post("/batches/<int:bid>/roboflow")
def batch_roboflow_target(bid):
    ws, proj = request.form.get("workspace", ""), request.form.get("project", "")
    if ws not in roboflow_keys():
        abort(400)
    db.x("UPDATE batch SET rf_workspace=?, rf_project=? WHERE id=?", (ws, proj or None, bid))
    return "", 204


@app.post("/batches/<int:bid>/labels/<int:sid>/delete")
def batch_label_delete(bid, sid):
    s = db.q1("SELECT * FROM label_set WHERE id=? AND batch_id=?", (sid, bid)) or abort(404)
    if any(sid in json.loads(r["label_set_ids"]) for r in db.q("SELECT label_set_ids FROM training_run")):
        flash(f"Set {sid} was used to train a model, so it stays.", "error")
        return redirect(url_for("batch", bid=bid))
    db.x("DELETE FROM label WHERE label_set_id=?", (sid,))
    db.x("DELETE FROM label_set WHERE id=?", (sid,))
    shutil.rmtree(DATA / "labels" / str(sid), ignore_errors=True)
    # step the batch back if it no longer has what its status claims
    if s["kind"] == "corrected" and not db.q1("SELECT 1 FROM label_set WHERE batch_id=? AND kind='corrected'", (bid,)):
        db.x("UPDATE batch SET status=CASE WHEN rf_sent_to IS NOT NULL THEN 'sent to Roboflow' ELSE 'pre-labelled' END "
             "WHERE id=? AND status='corrected'", (bid,))
    if not db.q1("SELECT 1 FROM label_set WHERE batch_id=?", (bid,)):
        db.x("UPDATE batch SET status='new' WHERE id=? AND status='pre-labelled'", (bid,))
    flash(f"Deleted label set {sid}.", "ok")
    return redirect(url_for("batch", bid=bid))


@app.post("/batches/<int:bid>/upload")
def batch_upload(bid):
    b = db.q1("SELECT * FROM batch WHERE id=?", (bid,)) or abort(404)
    if not b["rf_project"]:
        flash("Choose a Roboflow workspace and project first.", "error")
        return redirect(url_for("batch", bid=bid))
    sid = request.form.get("label_set_id") or None   # blank = images only
    if sid and not db.q1("SELECT 1 FROM label_set WHERE id=? AND batch_id=?", (sid, bid)):
        abort(400)
    sid = sid and int(sid)
    jobs.submit("roboflow", f"Send batch {bid} to Roboflow ({b['rf_project']}, {f'set {sid}' if sid else 'no labels'})",
                roboflow_io.api_upload, bid, sid)
    flash(f"Sending to Roboflow project {b['rf_project']}.", "ok")
    return redirect(url_for("batch", bid=bid))


@app.post("/batches/<int:bid>/pull")
def batch_pull(bid):
    b = db.q1("SELECT * FROM batch WHERE id=?", (bid,)) or abort(404)
    version = request.form.get("version", "").strip()
    if not (b["rf_project"] and version.isdigit()):
        flash("Choose the project and the Roboflow version to pull.", "error")
        return redirect(url_for("batch", bid=bid))
    jobs.submit("roboflow", f"Pull {b['rf_project']} version {version} into batch {bid}", roboflow_io.api_pull, bid, version)
    flash(f"Pulling {b['rf_project']} version {version} from Roboflow.", "ok")
    return redirect(url_for("batch", bid=bid))


# Roboflow lookups for the batch page's pickers; errors come back as text the page shows
@app.route("/api/roboflow/<ws>/projects", methods=["GET", "POST"])
def rf_projects(ws):
    try:
        if request.method == "POST":
            name = (request.get_json(silent=True) or {}).get("name", "").strip()
            if not name:
                return jsonify(error="Give the new project a name."), 400
            return jsonify(roboflow_io.create_project(ws, name))
        return jsonify(roboflow_io.list_projects(ws))
    except Exception as e:  # network or Roboflow refusals go back to the page as text
        return jsonify(error=str(e)), 502


@app.route("/api/roboflow/<ws>/<proj>/versions")
def rf_versions(ws, proj):
    try:
        return jsonify(roboflow_io.list_versions(ws, proj))
    except Exception as e:  # network or Roboflow refusals go back to the page as text
        return jsonify(error=str(e)), 502


@app.post("/batches/<int:bid>/import")
def batch_import(bid):
    f = request.files.get("zip")
    if not f or not f.filename.lower().endswith(".zip"):
        flash("Choose the .zip you exported from Roboflow (YOLO format).", "error")
        return redirect(url_for("batch", bid=bid))
    try:
        msg = roboflow_io.import_zip(None, bid, f.read(), f.filename)
        flash(msg, "ok")
    except Exception as e:  # zipfile/yaml/label errors all go back to the user as text
        flash(f"Import failed: {e}", "error")
    return redirect(url_for("batch", bid=bid))


# ---------------------------------------------------------------- training

@app.route("/training")
def training_list():
    runs = db.q("SELECT * FROM training_run ORDER BY id DESC")
    # Every batch with its label sets (Roboflow corrections first, then pre-labels, newest first in each).
    sets = db.q("""SELECT s.*, b.name AS batch_name, b.round, COUNT(l.frame_id) AS n FROM label_set s
                   JOIN batch b ON b.id=s.batch_id LEFT JOIN label l ON l.label_set_id=s.id
                   GROUP BY s.id ORDER BY b.id DESC, s.kind='prelabel', s.id DESC""")
    batches = {}
    for s in sets:
        batches.setdefault(s["batch_id"], {"name": s["batch_name"], "round": s["round"], "sets": []})["sets"].append(s)
    # Form values: Ultralytics defaults, or a previous run's settings for "Copy settings".
    pre = {"name": "", "copy_to": "", "extra": "", **training.DEFAULTS, **{k: s[3] for k, s in training.SPEC.items()}}
    pre["version"], pre["size"] = training.DEFAULT_BASE
    # Default pick per batch: its corrected labels if any, else nothing ticked.
    pre["sets"] = [b["sets"][0]["id"] for b in batches.values() if b["sets"][0]["kind"] == "corrected"]
    src = db.q1("SELECT * FROM training_run WHERE id=?", (request.args.get("from"),)) if request.args.get("from") else None
    if src:
        p = json.loads(src["params"])
        pre |= {k: p[k] for k in training.DEFAULTS} | p.get("advanced", {})
        pre["version"], pre["size"] = training.parse_base(Path(src["base_model"]).name)
        pre["sets"] = json.loads(src["label_set_ids"])
        pre |= {"name": f"{src['name']} (copy)", "copy_to": p.get("copy_to", ""),
                "extra": "\n".join(f"{k}={json.dumps(v) if not isinstance(v, str) else v}" for k, v in p.get("extra", {}).items())}
    return render_template("training.html", runs=runs, batches=batches, versions=training.VERSIONS, sizes=training.SIZES,
                           defaults=training.DEFAULTS, resumable={r["id"] for r in runs if training.resumable(r)},
                           pre=pre, src=src, advanced=training.ADVANCED, football=training.FOOTBALL,
                           models={m["training_run_id"]: m for m in db.q("SELECT * FROM model WHERE kind='kitman'")})


@app.post("/training/new")
def training_new():
    form = request.form
    # One label set per ticked batch, and it must belong to that batch.
    ids = []
    for b in form.getlist("batch_id"):   # not "batch": that's the batch-size field
        ls = db.q1("SELECT id FROM label_set WHERE id=? AND batch_id=?", (form.get(f"labels_{b}"), b))
        if ls:
            ids.append(ls["id"])
    if not ids:
        flash("Tick at least one batch and choose which of its labels to train on.", "error")
        return redirect(url_for("training_list"))
    copy_to = form.get("copy_to", "").strip()
    try:
        params = {k: int(form[k]) for k in training.DEFAULTS if form.get(k)}
        advanced, extra = training.settings_from_form(form)
        if copy_to and not Path(copy_to).is_dir():
            raise ValueError(f"The results folder “{copy_to}” doesn't exist. Choose it again.")
        base = training.base_weights(form.get("version", ""), form.get("size", ""))
        rid = training.create(form.get("name", "").strip() or "Untitled run", base, params, ids,
                              advanced, extra, copy_to)
    except ValueError as e:
        flash(str(e), "error")
        return redirect(url_for("training_list"))
    flash("Training started.", "ok")
    return redirect(url_for("training_run", rid=rid))


@app.route("/training/<int:rid>")
def training_run(rid):
    run = db.q1("SELECT * FROM training_run WHERE id=?", (rid,)) or abort(404)
    ids = json.loads(run["label_set_ids"])
    sets = db.q(f"""SELECT s.*, b.name AS batch_name, b.round FROM label_set s JOIN batch b ON b.id=s.batch_id
                    WHERE s.id IN ({','.join('?' * len(ids))})""", ids) if ids else []
    model = db.q1("SELECT * FROM model WHERE training_run_id=?", (rid,))
    used_by = db.q("SELECT s.*, b.name AS batch_name FROM label_set s JOIN batch b ON b.id=s.batch_id "
                   "WHERE s.model_id=?", (model["id"],)) if model else []
    p = json.loads(run["params"])
    passed = training.train_args(p)
    used, args_file = {}, Path(run["run_dir"] or "") / "train" / "args.yaml"
    if args_file.is_file():   # Ultralytics' own record of every setting the run used, defaults included
        import yaml
        used = yaml.safe_load(args_file.read_text(encoding="utf-8")) or {}
    return render_template("run.html", run=run, sets=sets, model=model, used_by=used_by,
                           passed=passed, used=used, copy_to=p.get("copy_to", ""), resumable=training.resumable(run))


@app.route("/api/training/<int:rid>")
def training_live(rid):
    run = db.q1("SELECT * FROM training_run WHERE id=?", (rid,)) or abort(404)
    return jsonify(status=run["status"], epochs=training.epochs(run), log=training.log_tail(run))


@app.post("/training/<int:rid>/stop")
def training_stop(rid):
    try:
        training.stop(rid)
    except (ValueError, OSError) as e:
        flash(f"Couldn't stop run {rid}: {e}", "error")
        return redirect(url_for("training_run", rid=rid))
    flash(f"Run {rid} stopped and removed; nothing of it is kept.", "ok")
    return redirect(url_for("training_list"))


@app.post("/training/<int:rid>/pause")
def training_pause(rid):
    try:
        training.pause(rid)
    except (ValueError, OSError) as e:
        flash(str(e), "error")
        return redirect(url_for("training_run", rid=rid))
    flash(f"Run {rid} paused. Resume carries on from its last finished epoch.", "ok")
    return redirect(url_for("training_run", rid=rid))


@app.post("/training/<int:rid>/resume")
def training_resume(rid):
    try:
        training.resume(rid)
    except (ValueError, OSError) as e:
        flash(str(e), "error")
        return redirect(url_for("training_list"))
    flash(f"Run {rid} resumed from its last checkpoint.", "ok")
    return redirect(url_for("training_run", rid=rid))


@app.post("/training/<int:rid>/test")
def training_test(rid):
    jobs.submit("evaluate", f"Evaluate run {rid} on test", training.evaluate_test, rid)
    flash("Test evaluation queued. Use it once, at the end; don't tune on it.", "ok")
    return redirect(url_for("training_run", rid=rid))


# ---------------------------------------------------------------- models, activity, settings

@app.route("/models")
def models():
    rows = db.q("""SELECT m.*, r.name AS run_name, r.metrics, r.test_metrics, r.counts, r.label_set_ids,
                   (SELECT COUNT(*) FROM label_set s WHERE s.model_id=m.id) AS n_prelabel_sets
                   FROM model m LEFT JOIN training_run r ON r.id=m.training_run_id ORDER BY m.id DESC""")
    for m in rows:
        m["source"] = "run" if m["training_run_id"] else \
            "uploaded" if Path(m["path"]).parent == DATA / "models" else "coco"
    return render_template("models.html", models=rows)


@app.post("/models/add")
def model_add():
    f = request.files.get("weights")
    name = request.form.get("name", "").strip()
    if not f or not f.filename.lower().endswith(".pt"):
        flash("Choose a YOLO weights file (.pt).", "error")
        return redirect(url_for("models"))
    (DATA / "models").mkdir(parents=True, exist_ok=True)
    n = (db.q1("SELECT MAX(id) AS n FROM model")["n"] or 0) + 1
    dst = DATA / "models" / f"model{n}_{secure_filename(f.filename)}"
    f.save(dst)
    try:   # load it once: proves it's a YOLO model and tells us how to read its classes
        from ultralytics import YOLO
        to_ours = labels.class_map(YOLO(str(dst)).names)
    except Exception as e:  # a bad file must not stay registered
        dst.unlink(missing_ok=True)
        flash(f"Couldn't use {f.filename}: {e}", "error")
        return redirect(url_for("models"))
    db.x("INSERT INTO model (name, path, kind, created) VALUES (?,?,?,?)",
         (name or Path(f.filename).stem, str(dst), "coco" if to_ours is None else "kitman", db.now()))
    flash(f"Added {name or f.filename}. It's now in the pre-label and training lists.", "ok")
    return redirect(url_for("models"))


@app.route("/activity")
def activity():
    return render_template("activity.html", jobs=db.q("SELECT * FROM job ORDER BY id DESC LIMIT 200"))


@app.route("/api/jobs/<int:jid>")
def job_api(jid):
    return jsonify(db.q1("SELECT id, status, progress, message, title FROM job WHERE id=?", (jid,)) or {})


@app.route("/api/jobs/active")
def jobs_active():
    return jsonify(db.q("SELECT id, status, progress, message, title FROM job WHERE status IN ('queued','running') "
                        "OR replace(finished,'T',' ') > datetime('now','localtime','-8 seconds') ORDER BY id"))


@app.route("/settings", methods=["GET", "POST"])
def settings():
    if request.method == "POST":
        keys = roboflow_keys()
        if request.form.get("remove"):
            keys.pop(request.form["remove"], None)
            save_roboflow_keys(keys)
            flash(f"Removed workspace {request.form['remove']}.", "ok")
        else:
            key = request.form.get("roboflow_api_key", "").strip()
            try:
                ws = roboflow_io.workspace_of(key)   # the key itself says which workspace it opens
            except Exception as e:  # bad key or no network: tell the user, save nothing
                flash(f"That key didn't work: {e}", "error")
                return redirect(url_for("settings"))
            save_roboflow_keys(keys | {ws: key})
            flash(f"Added workspace {ws}.", "ok")
        return redirect(url_for("settings"))
    return render_template("settings.html", workspaces=sorted(roboflow_keys()))


db.init()
jobs.start()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False, threaded=True)
