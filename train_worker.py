"""Runs one training run in its own process: python train_worker.py <run_id>."""

import json
import os
import sys
import traceback

import db
from training import metrics_of, train_args


def copy_results(run_dir, dest, run_id):
    """Copy best/last weights, results, plots, settings and log to the user's folder (Kitman keeps its own copy)."""
    import shutil
    from pathlib import Path
    src, out = Path(run_dir), Path(dest) / f"kitman_run{run_id}"
    (out / "weights").mkdir(parents=True, exist_ok=True)
    for w in ("best.pt", "last.pt"):
        if (src / "train" / "weights" / w).exists():
            shutil.copy2(src / "train" / "weights" / w, out / "weights" / w)
    for f in (src / "train").iterdir():
        if f.is_file():   # results.csv, args.yaml, plots
            shutil.copy2(f, out / f.name)
    for f in ("train.log", "frames.csv"):
        if (src / f).exists():
            shutil.copy2(src / f, out / f)
    return out


def main(run_id, resume=False):
    from ultralytics import YOLO

    run = db.q1("SELECT * FROM training_run WHERE id=?", (run_id,))
    p = json.loads(run["params"])
    data = os.path.join(run["run_dir"], "dataset", "data.yaml")
    db.x("UPDATE training_run SET status='running' WHERE id=?", (run_id,))
    try:
        # The returned metrics are Ultralytics' own final validation of best.pt: the last table in the log.
        if resume:   # Ultralytics restores every setting from last.pt itself
            last = os.path.join(run["run_dir"], "train", "weights", "last.pt")
            print(f"KITMAN: resuming from {last}", flush=True)
            model = YOLO(last)
            res = model.train(resume=True)
        else:
            model = YOLO(run["base_model"])
            args = train_args(p)
            print(f"KITMAN: training with {args}", flush=True)
            res = model.train(data=data, project=run["run_dir"], name="train", exist_ok=True, **args)
        best = os.path.join(run["run_dir"], "train", "weights", "best.pt")
        metrics = metrics_of(res, model.names)
        db.x("UPDATE training_run SET status='done', metrics=?, finished=? WHERE id=?",
             (db.jdump(metrics), db.now(), run_id))
        db.x("INSERT INTO model (name, path, kind, training_run_id, created) VALUES (?,?,?,?,?)",
             (f"Run {run_id}: {run['name']}", best, "kitman", run_id, db.now()))
        print(f"KITMAN: run {run_id} done, val mAP50 {metrics['overall']['map50']:.4f}", flush=True)
        if p.get("copy_to"):
            try:   # a failed copy must not fail a finished run; Kitman still has everything
                print(f"KITMAN: copied results to {copy_results(run['run_dir'], p['copy_to'], run_id)}", flush=True)
            except OSError as e:
                print(f"KITMAN: couldn't copy results to {p['copy_to']}: {e}", flush=True)
    except BaseException:
        traceback.print_exc()
        db.x("UPDATE training_run SET status='failed', finished=? WHERE id=?", (db.now(), run_id))
        raise


if __name__ == "__main__":
    main(int(sys.argv[1]), "--resume" in sys.argv)
