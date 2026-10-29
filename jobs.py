"""Background jobs: sampling, pre-labelling and Roboflow transfers run in one worker
thread, one at a time (they all want the GPU or the disk), and report progress to
the job table so the UI can show it."""

import queue
import threading
import traceback

import db

_q = queue.Queue()


class Job:
    def __init__(self, job_id):
        self.id = job_id

    def log(self, line):
        print(f"[job {self.id}] {line}", flush=True)
        db.x("UPDATE job SET log = log || ? WHERE id=?", (line + "\n", self.id))

    def progress(self, frac, message=None):
        db.x("UPDATE job SET progress=?, message=COALESCE(?, message) WHERE id=?",
             (round(float(frac), 4), message, self.id))


def submit(kind, title, fn, *args):
    job_id = db.x("INSERT INTO job (kind, title, status, created) VALUES (?,?,?,?)",
                  (kind, title, "queued", db.now()))
    _q.put((job_id, fn, args))
    return job_id


def _worker():
    while True:
        job_id, fn, args = _q.get()
        job = Job(job_id)
        db.x("UPDATE job SET status='running' WHERE id=?", (job_id,))
        try:
            msg = fn(job, *args)
            db.x("UPDATE job SET status='done', progress=1, message=?, finished=? WHERE id=?",
                 (msg or "Done", db.now(), job_id))
        except Exception as e:  # report every failure in the UI instead of losing it
            job.log(traceback.format_exc())
            db.x("UPDATE job SET status='failed', message=?, finished=? WHERE id=?",
                 (f"{type(e).__name__}: {e}", db.now(), job_id))


def start():
    threading.Thread(target=_worker, daemon=True, name="kitman-jobs").start()
