// Kitman front end: job dock, box overlay, live training chart. No framework needed.

const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

// ---------- live areas: re-render [data-live] parts of the page from the server while jobs run
let refreshing = false;
async function refreshLive() {
  const areas = document.querySelectorAll("[data-live]");
  if (!areas.length || refreshing) return;
  refreshing = true;
  try {
    const res = await fetch(location.href, { headers: { "X-Kitman-Live": "1" } });
    if (!res.ok) return;   // e.g. a page reached by POST can't be re-fetched; leave it as is
    const doc = new DOMParser().parseFromString(await res.text(), "text/html");
    areas.forEach((el) => {
      const fresh = doc.querySelector(`[data-live="${el.dataset.live}"]`);
      if (!fresh || el.contains(document.activeElement)) return;   // don't pull a control out from under the user
      const keep = {};   // values the user changed but hasn't submitted
      el.querySelectorAll("input[name], select[name], textarea[name]").forEach((i) => {
        const changed = i.type === "checkbox" ? i.checked !== i.defaultChecked
          : i.tagName === "SELECT" ? i.selectedIndex !== Math.max(0, [...i.options].findIndex((o) => o.defaultSelected))
          : i.value !== i.defaultValue;
        if (changed) keep[i.name] = i.type === "checkbox" ? i.checked : i.value;
      });
      const open = [...el.querySelectorAll("details")].map((d) => d.open);
      el.innerHTML = fresh.innerHTML;
      el.querySelectorAll("details").forEach((d, k) => { if (open[k]) d.open = true; });
      el.querySelectorAll("input[name], select[name], textarea[name]").forEach((i) => {
        if (!(i.name in keep)) return;
        if (i.type === "checkbox") i.checked = keep[i.name]; else i.value = keep[i.name];
      });
    });
  } catch { /* server restarting; the next poll tries again */ } finally { refreshing = false; }
}

// ---------- job dock: running job, a count of waiting ones; pages update in place
(function dock() {
  const el = $("#dock");
  let wasActive = false, lastRefresh = 0;
  async function poll() {
    try {
      const all = await (await fetch("/api/jobs/active")).json();
      const waiting = all.filter((j) => j.status === "queued");
      const jobs = all.filter((j) => j.status !== "queued");   // waiting jobs fold into one summary line
      el.innerHTML = jobs.map((j) => `
        <div class="job ${j.status}">
          <b>${esc(j.title)}</b>
          <span>${esc(j.message || (j.status === "running" ? "Starting" : ""))}</span>
          <div class="bar"><i style="width:${Math.round((j.status === "done" ? 1 : j.progress) * 100)}%"></i></div>
        </div>`).join("") + (waiting.length ? `
        <div class="job"><b>${waiting.length} more waiting</b><span>Next: ${esc(waiting[0].title)}. See <a href="/activity">Activity</a>.</span></div>` : "");
      // training runs aren't jobs: a live area still showing one in progress keeps refreshing too
      const active = all.some((j) => j.status === "queued" || j.status === "running")
        || !!document.querySelector("[data-live] .status.queued, [data-live] .status.running");
      // While work runs, refresh every ~3 s; once more when it all finishes.
      if ((active && Date.now() - lastRefresh > 3000) || (wasActive && !active)) {
        lastRefresh = Date.now();
        refreshLive();
      }
      wasActive = active;
      setTimeout(poll, active ? 1500 : 6000);
    } catch { setTimeout(poll, 8000); }
  }
  poll();
  // things may have changed in another tab (a run resumed there): catch up when this one is shown again
  document.addEventListener("visibilitychange", () => { if (!document.hidden) refreshLive(); });
  // ...and when Back/Forward shows the browser's saved snapshot of the page instead of loading it
  window.addEventListener("pageshow", (e) => { if (e.persisted) refreshLive(); });
})();

// ---------- several-videos picker: "All <split>" toggles, a summary of what's ticked, closes on outside click
document.addEventListener("change", (e) => {
  const picker = e.target.closest(".picker");
  if (!picker) return;
  const boxes = [...picker.querySelectorAll('input[name="match"]')];
  if (e.target.classList.contains("picker-all")) {
    boxes.filter((b) => b.dataset.split === e.target.dataset.split).forEach((b) => { b.checked = e.target.checked; });
  }
  picker.querySelectorAll(".picker-all").forEach((all) => {
    const group = boxes.filter((b) => b.dataset.split === all.dataset.split);
    all.checked = group.every((b) => b.checked);
    all.indeterminate = !all.checked && group.some((b) => b.checked);
  });
  const on = boxes.filter((b) => b.checked);
  const summary = $("summary", picker);
  summary.textContent = !on.length ? summary.dataset.empty
    : on.length === boxes.length ? `All videos (${on.length})`
    : on.length <= 4 ? on.map((b) => b.value).join(", ") : `${on.length} videos: ${on.slice(0, 3).map((b) => b.value).join(", ")}…`;
});
document.addEventListener("click", (e) => {
  document.querySelectorAll(".picker[open]").forEach((p) => { if (!p.contains(e.target)) p.open = false; });
});

// ---------- New run: football / Ultralytics defaults, and the copy-results folder
function trainingForm() {
  const form = document.querySelector("form[action$='/training/new']");
  if (!form) return;
  const set = (key, v) => {
    const el = form.elements[key];
    if (!el) return;
    if (el.type === "checkbox") el.checked = v === true || v === "true";
    else el.value = v;
  };
  $("#use-football").addEventListener("click", (e) => {
    Object.entries(JSON.parse(e.target.dataset.values)).forEach(([k, v]) => set(k, v));
  });
  $("#use-defaults").addEventListener("click", () => {
    form.querySelectorAll(".advanced [data-default]").forEach((el) => set(el.name, el.type === "checkbox" ? JSON.parse(el.dataset.default) : el.dataset.default));
  });
  const input = $("#copy-to"), note = $("#copy-to-note"), clear = $("#clear-copy-to"), pick = $("#pick-copy-to");
  pick.addEventListener("click", async () => {
    pick.disabled = true;
    note.textContent = "A folder window has opened. It may be behind this one.";
    try {
      const { folder } = await (await fetch("/videos/pick?title=" + encodeURIComponent("Choose where to copy the training results"), { method: "POST" })).json();
      if (folder) { input.value = folder; note.textContent = `Results will also be copied to ${folder}`; clear.hidden = false; }
      else note.textContent = input.value || "No folder chosen.";
    } catch { note.textContent = "Couldn't open the folder window. Check the Kitman server is running."; }
    pick.disabled = false;
  });
  clear.addEventListener("click", () => { input.value = ""; note.textContent = "Kitman keeps its own copy in kitman/data/runs."; clear.hidden = true; });
}

// ---------- add a folder: the server opens the Windows folder picker, then the folder is scanned
function pickFolder() {
  const btn = $("#pick"), note = $("#pick-note"), form = $("#scan");
  if (!btn) return;
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    note.textContent = "A folder window has opened. It may be behind this one.";
    try {
      const { folder } = await (await fetch("/videos/pick", { method: "POST" })).json();
      if (folder) { form.folder.value = folder; note.textContent = `Listing videos in ${folder}`; form.submit(); return; }
      note.textContent = "No folder chosen.";
    } catch { note.textContent = "Couldn't open the folder window. Check the Kitman server is running."; }
    btn.disabled = false;
  });
}

// ---------- video upload: show how far a ~1 GB upload has got instead of a frozen page
function uploadWithProgress() {
  const form = $("#upload"), note = $("#upload-progress");
  if (!form) return;
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const xhr = new XMLHttpRequest(), btn = $("button", form);
    btn.disabled = true;
    xhr.upload.onprogress = (ev) => {
      if (ev.lengthComputable) note.textContent = `Uploading ${Math.round((ev.loaded / ev.total) * 100)}% of ${(ev.total / 2 ** 30).toFixed(2)} GB`;
    };
    xhr.onload = () => { location.href = xhr.responseURL || location.href; };
    xhr.onerror = () => { btn.disabled = false; note.textContent = "Upload failed. Check the server is still running and try again."; };
    xhr.open("POST", form.action);
    xhr.send(new FormData(form));
  });
}

// ---------- frame viewer: draw a label set's boxes over the image
function drawBoxes(sets, classes) {
  const img = $("#img"), canvas = $("#boxes"), pick = $("#set");
  if (!pick) return;
  const colours = classes.map((c) => getComputedStyle(document.documentElement).getPropertyValue(`--c-${c}`).trim());
  function draw() {
    canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const boxes = sets[+pick.value] || [];
    const on = new Set([...document.querySelectorAll("#toggles input:checked")].map((i) => +i.value));
    ctx.lineWidth = Math.max(2, canvas.width / 700);
    for (const [c, x, y, w, h] of boxes) {
      if (!on.has(c)) continue;
      ctx.strokeStyle = colours[c] || "#fff";
      const pad = classes[c] === "ball" ? 6 : 0;   // balls are ~13 px; widen the outline so it stays visible
      ctx.strokeRect((x - w / 2) * canvas.width - pad, (y - h / 2) * canvas.height - pad, w * canvas.width + 2 * pad, h * canvas.height + 2 * pad);
    }
  }
  pick.addEventListener("change", draw);
  document.querySelectorAll("#toggles input").forEach((i) => i.addEventListener("change", draw));
  img.complete ? draw() : img.addEventListener("load", draw);
}

// ---------- training run: poll results.csv and the log while it runs
function followRun(id, status) {
  const chart = $("#chart"), note = $("#chart-note"), log = $("#log");
  async function tick() {
    const d = await (await fetch(`/api/training/${id}`)).json();
    const badge = $(".lede .status");   // same markup as the status() macro
    if (badge) { badge.className = `status ${d.status.replace(/ /g, "-")}`; badge.textContent = d.status; }
    log.textContent = d.log || "No output yet.";
    log.scrollTop = log.scrollHeight;
    plot(d.epochs);
    if (d.status === "queued" || d.status === "running") setTimeout(tick, 5000);
    else if (status === "queued" || status === "running") location.reload();   // show final metrics
  }
  function plot(rows) {
    const series = [["metrics/mAP50(B)", "var(--red)", "mAP50"], ["metrics/mAP50-95(B)", "var(--ink)", "mAP50-95"]];
    if (!rows.length) return;
    const W = 600, H = 220, L = 36, B = 24, n = rows.length;
    const max = Math.max(0.1, ...rows.flatMap((r) => series.map(([k]) => r[k] || 0)));
    const x = (i) => L + (n === 1 ? 0 : (i / (n - 1)) * (W - L - 10));
    const y = (v) => H - B - (v / max) * (H - B - 12);
    let svg = "";
    for (const t of [0, max / 2, max]) svg += `<line x1="${L}" x2="${W - 10}" y1="${y(t)}" y2="${y(t)}" stroke="var(--line-2)"/><text x="4" y="${y(t) + 4}" font-size="11" fill="var(--ink-2)">${t.toFixed(2)}</text>`;
    for (const [k, col] of series) {
      const pts = rows.map((r, i) => `${x(i).toFixed(1)},${y(r[k] || 0).toFixed(1)}`).join(" ");
      svg += `<polyline points="${pts}" fill="none" stroke="${col}" stroke-width="2"/>`;
    }
    svg += `<text x="${L}" y="${H - 6}" font-size="11" fill="var(--ink-2)">epoch 1</text><text x="${W - 10}" y="${H - 6}" font-size="11" fill="var(--ink-2)" text-anchor="end">epoch ${n}</text>`;
    chart.innerHTML = svg;
    const last = rows[n - 1];
    note.textContent = `Epoch ${n}: mAP50 ${(last["metrics/mAP50(B)"] || 0).toFixed(3)} (red), mAP50-95 ${(last["metrics/mAP50-95(B)"] || 0).toFixed(3)} (navy).`;
  }
  tick();
}

// ---------- batch page: the Roboflow workspace and project (remembered per batch) and the version to pull
function roboflowTarget() {
  const box = $("#rf-target");
  if (!box) return;
  const ws = $("select[name=workspace]", box), proj = $("select[name=project]", box);
  const note = $("#rf-note"), versions = $("#rf-versions"), send = $("#rf-send"), pull = $("#rf-pull");
  const sentTo = box.dataset.sentTo;
  const opt = (value, text) => new Option(text, value);
  const json = async (res) => {
    const d = await res.json();
    if (!res.ok) throw new Error(d.error || res.statusText);
    return d;
  };
  const save = () => {
    const body = new FormData();
    body.set("workspace", ws.value);
    body.set("project", proj.value);
    fetch(box.dataset.save, { method: "POST", body });
  };

  async function loadVersions() {
    send.disabled = pull.disabled = !proj.value;
    note.textContent = sentTo && proj.value && proj.value !== sentTo
      ? `This batch was sent to ${sentTo}. Frames already there stay in that project.` : "";
    versions.replaceChildren(opt("", proj.value ? "Loading…" : "Choose a project first"));
    if (!proj.value) return;
    try {
      const vs = await json(await fetch(`/api/roboflow/${ws.value}/${proj.value}/versions`));
      versions.replaceChildren(...(vs.length
        ? vs.map((v) => opt(v.n, `v${v.n} · ${v.name} (${v.images} images)${v.ready ? "" : ", still generating"}`))
        : [opt("", "No versions yet. Generate one in Roboflow")]));
      pull.disabled = !vs.length;
    } catch (e) {
      versions.replaceChildren(opt("", "Couldn't load versions"));
      pull.disabled = true;
      note.textContent = e.message;
    }
  }

  async function loadProjects(keep) {
    proj.replaceChildren(opt("", "Loading…"));
    send.disabled = pull.disabled = true;
    try {
      const ps = await json(await fetch(`/api/roboflow/${ws.value}/projects`));
      proj.replaceChildren(opt("", "Choose a project"), ...ps.map((p) => opt(p.id, `${p.name} (${p.images} images)`)),
        opt("__new", "+ New project…"));
      proj.value = ps.some((p) => p.id === keep) ? keep : "";
    } catch (e) {
      proj.replaceChildren(opt("", "Couldn't load projects"));
      note.textContent = e.message;
      return;
    }
    await loadVersions();
  }

  ws.addEventListener("change", async () => { await loadProjects(""); save(); });
  proj.addEventListener("change", async () => {
    if (proj.value === "__new") {
      const name = (prompt(`Name for the new project in ${ws.value}`) || "").trim();
      proj.value = "";
      if (name) {
        try {
          const p = await json(await fetch(`/api/roboflow/${ws.value}/projects`, {
            method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }) }));
          proj.add(opt(p.id, `${p.name} (0 images)`), proj.options[proj.options.length - 1]);
          proj.value = p.id;
        } catch (e) {
          note.textContent = `Couldn't create the project: ${e.message}`;
          return;
        }
      }
    }
    save();
    loadVersions();
  });
  loadProjects(box.dataset.project);
}
