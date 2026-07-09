#!/usr/bin/env python3
"""navmesh_editor — browser tool to draw a walkable area on a 3D Gaussian Splatting
scene and bake a Habitat .navmesh.

Run with a Python that has habitat_sim + flask (e.g. the habitat-gs conda env):

    python web_tools/navmesh_editor/server.py --port 8080

then open http://<host>:8080 (over an SSH tunnel if running remotely). The scene
directory and the habitat_sim interpreter can be overridden with --gs-dir /
--habitat-python (or the NAVMESH_EDITOR_GS_DIR / NAVMESH_EDITOR_PYTHON env vars).
The toolbar and on-screen hints document the in-browser controls.
"""
import argparse
import base64
import glob
import json
import logging
import os
import re
import subprocess
import sys
import threading

import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
# Repo root is two levels up: web_tools/navmesh_editor/ -> web_tools/ -> repo root.
REPO_ROOT = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir))
STATIC = os.path.join(HERE, "static")
TMP = os.path.join(HERE, "_tmp")
NAV_WORKER = os.path.join(HERE, "_nav_worker.py")
os.makedirs(TMP, exist_ok=True)

# Scene directory: a folder whose subdirectories hold GS scenes. It may point either
# at a folder whose immediate subdirs are scenes (e.g. .../gs_scenes/train) or at a
# parent like gs_scenes that groups several such folders (train/, val/, ...) — both
# layouts are discovered, and any subdir without a *.gs.ply is simply ignored.
# Defaults to the gs_scenes dataset inside this habitat-gs checkout; override with
# --gs-dir / the env var.
GS_DIR = os.environ.get(
    "NAVMESH_EDITOR_GS_DIR",
    os.path.join(REPO_ROOT, "data", "scene_datasets", "gs_scenes"),
)
# All habitat_sim work runs in _nav_worker.py as a subprocess, so this server can
# run under any Python; only the worker needs habitat_sim. By default the worker
# uses this same interpreter; override with --habitat-python / the env var.
HABITAT_PY = os.environ.get("NAVMESH_EDITOR_PYTHON", sys.executable)

SH_C0 = 0.28209479177387814
SPLAT_ALPHA_MIN = 0.1           # drop faint gaussians (floaters) from the .splat
MAX_SPLATS = 1_000_000          # cap total splats sent to the browser (keep the most opaque)

logging.getLogger("werkzeug").setLevel(logging.ERROR)   # quiet the dev-server banner + access logs
app = Flask(__name__, static_folder=None)
_lock = threading.Lock()
_scene_cache = {}     # ply_path -> meta dict (bounds/floor/suggested/splat path)
_last_bake = {}       # scene_name -> baked .navmesh tmp path
_bake_seq = {"n": 0}  # monotonic counter -> unique proxy filename per bake


# --------------------------------------------------------------------------- #
#  scene discovery / resolution
# --------------------------------------------------------------------------- #
def list_scenes():
    """Discover scenes under GS_DIR. A scene is a directory holding a *.gs.ply.
    Each immediate subdir of GS_DIR is either a scene itself (has a *.gs.ply) or a
    group whose subdirs are scenes (e.g. train/<scene>, val/<scene>); both are
    scanned, so subdirs without any *.gs.ply (assets/, avatars/, ...) drop out."""
    out, seen = [], set()
    for ply in sorted(glob.glob(os.path.join(GS_DIR, "*.gs.ply"))):
        if ply in seen:
            continue
        seen.add(ply)
        _, name, group = scene_dir_and_split(ply)
        nav = os.path.join(os.path.dirname(ply), name + ".navmesh")
        out.append({"name": name, "split": group, "has_navmesh": os.path.exists(nav)})
    for sub in sorted(glob.glob(os.path.join(GS_DIR, "*", ""))):
        plys = (sorted(glob.glob(os.path.join(sub, "*.gs.ply")))          # sub IS a scene
                or sorted(glob.glob(os.path.join(sub, "*", "*.gs.ply"))))  # sub groups scenes
        for ply in plys:
            if ply in seen:
                continue
            seen.add(ply)
            _, name, group = scene_dir_and_split(ply)
            nav = os.path.join(os.path.dirname(ply), name + ".navmesh")
            out.append({"name": name, "split": group, "has_navmesh": os.path.exists(nav)})
    return out


def resolve_ply(scene):
    if not scene:
        return None
    if scene.endswith(".gs.ply") and os.path.isfile(scene):
        return os.path.abspath(scene)
    # a scene dir given absolutely or relative to GS_DIR (e.g. "interior_x" when
    # --gs-dir points straight at a group, or "train/interior_x" when it's the parent)
    for cand in (scene, os.path.join(GS_DIR, scene)):
        if os.path.isdir(cand):
            hits = sorted(glob.glob(os.path.join(cand, "*.gs.ply")))
            if hits:
                return hits[0]
    hits = sorted(glob.glob(os.path.join(GS_DIR, f"{scene}.gs.ply")))
    if hits:
        return hits[0]
    # a bare scene name: search one group level under GS_DIR (e.g. <group>/<scene>)
    hits = sorted(glob.glob(os.path.join(GS_DIR, "*", scene, "*.gs.ply")))
    if hits:
        return hits[0]
    return None


def scene_dir_and_split(ply_path):
    d = os.path.dirname(ply_path)
    ply_name = os.path.basename(ply_path)
    ply_stem = ply_name[:-len(".gs.ply")] if ply_name.endswith(".gs.ply") else os.path.splitext(ply_name)[0]
    if os.path.abspath(d) == os.path.abspath(GS_DIR):
        name = ply_stem
        group = os.path.basename(d)
    else:
        name = os.path.basename(d)
        group = os.path.basename(os.path.dirname(d))   # parent folder name, e.g. train / val
    return d, name, group


# --------------------------------------------------------------------------- #
#  read .gs.ply ONCE -> write .splat for the browser + compute floor/suggested
# --------------------------------------------------------------------------- #
def floor_ceiling_estimate(y):
    """Locate the floor and (for indoor scenes) the ceiling from the vertical splat
    distribution. The floor and ceiling are the two dominant horizontal surfaces, so
    we split the occupied HEIGHT range at its midpoint and take the densest layer in
    each half. Splitting by height (not by a global density threshold) is what makes
    this robust: a ceiling holding many times more splats than the floor — common in
    indoor scenes — can no longer masquerade as the floor, and a sparse outlier band
    far below the floor is never the densest layer in the lower half. The ceiling is
    reported only when the densest upper layer is a genuine surface a plausible
    room-height above the floor; otherwise None (open / outdoor scene)."""
    ylo, yhi = np.percentile(y, [1, 99])
    if yhi - ylo < 1e-3:
        return float(ylo), None
    nb = max(40, int((yhi - ylo) / 0.05))
    h, edges = np.histogram(y, bins=nb, range=(ylo, yhi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    low = centers < 0.5 * (ylo + yhi)                   # lower half of the height range
    floor = float(centers[low][int(np.argmax(h[low]))])
    floor_n = int(h[low].max())
    ceiling = None
    hi_idx = np.nonzero(~low)[0]                         # upper half
    if len(hi_idx):
        j = int(hi_idx[int(np.argmax(h[hi_idx]))])      # densest upper layer
        gap = float(centers[j]) - floor
        if 1.2 <= gap <= 10.0 and h[j] >= 0.3 * floor_n:
            ceiling = float(centers[j])
    return floor, ceiling


def prepare_scene(ply_path):
    """Idempotent: ensure _tmp/<scene>.splat exists and return scene metadata."""
    if ply_path in _scene_cache and os.path.exists(_scene_cache[ply_path]["splat"]):
        return _scene_cache[ply_path]
    import time
    from plyfile import PlyData
    t0 = time.time()
    pd = PlyData.read(ply_path)
    v = pd["vertex"]
    n = len(v)
    names = v.data.dtype.names or ()
    xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float32, copy=False)  # full (bounds/floor)
    if "opacity" in names:
        a = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], np.float32)))
    else:
        a = np.ones(n, np.float32)

    # prune: drop faint AND non-finite gaussians, then cap to the most-opaque MAX_SPLATS.
    # (some scenes have NaN/Inf-position floaters that would poison the scene bounds ->
    # NaN camera -> blank render in the browser.)
    idx = np.nonzero((a > SPLAT_ALPHA_MIN) & np.isfinite(xyz).all(axis=1))[0]
    if len(idx) > MAX_SPLATS:
        idx = idx[np.argpartition(a[idx], -MAX_SPLATS)[-MAX_SPLATS:]]

    # compute attributes ONLY for the kept gaussians (far less work than all n)
    dc = np.column_stack([v["f_dc_0"][idx], v["f_dc_1"][idx], v["f_dc_2"][idx]]).astype(np.float32)
    rgb = np.clip(0.5 + SH_C0 * dc, 0.0, 1.0)
    scale = np.exp(np.column_stack([v["scale_0"][idx], v["scale_1"][idx], v["scale_2"][idx]]).astype(np.float32))
    q = np.column_stack([v[f"rot_{i}"][idx] for i in range(4)]).astype(np.float32)  # wxyz
    q /= (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)

    sd, name, split = scene_dir_and_split(ply_path)
    splat = os.path.join(TMP, f"{name}.splat")
    out = np.zeros(len(idx), dtype=[('pos', '<f4', 3), ('scale', '<f4', 3),
                                    ('rgba', 'u1', 4), ('rot', 'u1', 4)])
    out['pos'] = xyz[idx]
    out['scale'] = scale
    out['rgba'][:, :3] = (rgb * 255).astype(np.uint8)
    out['rgba'][:, 3] = (a[idx] * 255).astype(np.uint8)
    out['rot'] = np.clip(np.round(q * 128 + 128), 0, 255).astype(np.uint8)
    out.tofile(splat)
    print(f"[splat] {name}: {n:,} -> {len(idx):,} splats "
          f"({os.path.getsize(splat)/1024**2:.1f} MB) in {time.time()-t0:.1f}s", flush=True)

    fy, ceiling_y = floor_ceiling_estimate(xyz[idx, 1])   # opaque (kept) splats -> cleaner profile
    band = xyz[(xyz[:, 1] >= fy - 0.05) & (xyz[:, 1] <= fy + 0.6)]
    if len(band) < 50:
        band = xyz
    x3, x97 = np.percentile(band[:, 0], [3, 97])
    z3, z97 = np.percentile(band[:, 2], [3, 97])
    suggested = [[float(x3), float(z3)], [float(x97), float(z3)],
                 [float(x97), float(z97)], [float(x3), float(z97)]]

    meta = {"name": name, "split": split, "splat": splat,
            "n_splats": int(len(idx)), "n_total": int(n),
            "bounds": {"min": xyz[idx].min(0).tolist(), "max": xyz[idx].max(0).tolist()},
            "floor_y": fy, "ceiling_y": ceiling_y, "suggested": suggested}
    _scene_cache[ply_path] = meta
    return meta


# --------------------------------------------------------------------------- #
#  persistent navmesh worker (subprocess) + navmesh outline extraction
# --------------------------------------------------------------------------- #
# All habitat_sim work runs in _nav_worker.py. The worker imports habitat_sim ONCE
# and stays alive, so every bake after the first is ~instant (sim init + Recast are
# ~0 s; the ~6 s cost was re-importing habitat_sim each time). The worker exits on
# its own when this server dies (its stdin pipe closes).
_worker = {"proc": None}
_worker_lock = threading.Lock()


def _worker_readline_result(p):
    while True:
        line = p.stdout.readline()
        if line == "":                              # EOF -> worker died
            return {"ok": False, "error": "navmesh worker exited"}
        i = line.find("RESULT:")                    # tolerate C++ noise prefixing the marker
        if i >= 0:
            return json.loads(line[i + len("RESULT:"):])
        # any other line = habitat_sim's own stdout logging -> ignore


def _worker_start():
    p = subprocess.Popen([HABITAT_PY, NAV_WORKER, "serve"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=open(os.path.join(TMP, "_worker.err"), "a"),
                         text=True, bufsize=1)
    ready = _worker_readline_result(p)              # wait for {"ready":true}
    if not ready.get("ready"):
        raise RuntimeError(ready.get("error", "worker failed to start"))
    return p


def worker_call(req):
    """Send one request to the persistent worker (start/restart it if needed)."""
    with _worker_lock:
        p = _worker["proc"]
        if p is None or p.poll() is not None:
            try:
                p = _worker["proc"] = _worker_start()
            except Exception as e:
                return {"ok": False, "error": f"worker start failed: {e}"}
        try:
            p.stdin.write(json.dumps(req) + "\n"); p.stdin.flush()
            return _worker_readline_result(p)
        except Exception as e:
            _worker["proc"] = None                  # force restart next time
            return {"ok": False, "error": f"worker io error: {e}"}


def _loop_area(p):
    a = 0.0
    for i in range(len(p)):
        x1, z1 = p[i]; x2, z2 = p[(i + 1) % len(p)]
        a += x1 * z2 - x2 * z1
    return a / 2.0


def _rdp_open(pts, eps):
    """Iterative Douglas-Peucker on an open polyline (recursion-safe)."""
    n = len(pts)
    if n < 3:
        return list(pts)
    keep = [False] * n; keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        ax, az = pts[i0]; bx, bz = pts[i1]
        abx, abz = bx - ax, bz - az; L = (abx * abx + abz * abz) ** 0.5 + 1e-12
        dmax, idx = 0.0, -1
        for i in range(i0 + 1, i1):
            px, pz = pts[i]
            d = abs((px - ax) * abz - (pz - az) * abx) / L
            if d > dmax:
                dmax, idx = d, i
        if dmax > eps and idx != -1:
            keep[idx] = True; stack.append((i0, idx)); stack.append((idx, i1))
    return [pts[i] for i in range(n) if keep[i]]


def _rdp_loop(loop, eps):
    """Simplify a closed loop: split at the two farthest-apart vertices, RDP each arc."""
    P = np.asarray(loop)
    i1 = int(((P - P[0]) ** 2).sum(1).argmax())
    i0 = int(((P - P[i1]) ** 2).sum(1).argmax())
    if i0 > i1:
        i0, i1 = i1, i0
    c1 = loop[i0:i1 + 1]; c2 = loop[i1:] + loop[:i0 + 1]
    return _rdp_open(c1, eps)[:-1] + _rdp_open(c2, eps)[:-1]


def _point_in(poly, pt):
    x, y = pt; inside = False; n = len(poly); j = n - 1
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-20) + xi):
            inside = not inside
        j = i
    return inside


def navmesh_outline(tris, max_verts=120):
    """Boundary polygon of the navmesh in XZ as {outer, holes}: edges used by one
    triangle are boundary edges, chained into loops; the largest = outer, contained
    loops = holes. Adaptively simplified so the editable outline stays usable."""
    from collections import defaultdict
    pos = []; idmap = {}

    def vid(x, z):
        k = (round(float(x) / 1e-3), round(float(z) / 1e-3)); i = idmap.get(k)
        if i is None:
            i = len(pos); pos.append((float(x), float(z))); idmap[k] = i
        return i

    ec = defaultdict(int)
    for i in range(0, len(tris), 9):
        a = vid(tris[i], tris[i + 2]); b = vid(tris[i + 3], tris[i + 5]); c = vid(tris[i + 6], tris[i + 8])
        for u, v in ((a, b), (b, c), (c, a)):
            if u != v:
                ec[(min(u, v), max(u, v))] += 1
    adj = defaultdict(list)
    for (u, v), cnt in ec.items():
        if cnt == 1:
            adj[u].append(v); adj[v].append(u)
    seen = set(); loops = []
    for s in list(adj):
        for f in adj[s]:
            if (min(s, f), max(s, f)) in seen:
                continue
            loop = [s]; prev, cur = s, f; seen.add((min(s, f), max(s, f))); ok = True; g = 0
            while cur != s:
                loop.append(cur); nxt = None
                for nb in adj[cur]:
                    if nb != prev and (min(cur, nb), max(cur, nb)) not in seen:
                        nxt = nb; break
                if nxt is None:
                    if s in adj[cur]:
                        break
                    ok = False; break
                seen.add((min(cur, nxt), max(cur, nxt))); prev, cur = cur, nxt; g += 1
                if g > 1_000_000:
                    ok = False; break
            if ok and len(loop) >= 3:
                loops.append([pos[i] for i in loop])
    loops = [l for l in loops if abs(_loop_area(l)) > 0.3]
    if not loops:
        return None
    loops.sort(key=lambda l: abs(_loop_area(l)), reverse=True)
    outer = loops[0]
    eps = 0.04
    so = _rdp_loop(outer, eps)
    while len(so) > max_verts and eps < 3.0:
        eps *= 1.4; so = _rdp_loop(outer, eps)
    holes = []
    for l in loops[1:]:
        cx = sum(p[0] for p in l) / len(l); cz = sum(p[1] for p in l) / len(l)
        if _point_in(outer, (cx, cz)):
            sh = _rdp_loop(l, eps)
            if len(sh) >= 3:
                holes.append(sh)
    return {"outer": so, "holes": holes, "n_outer_raw": len(outer), "n_islands": len(loops)}


def navmesh_payload(navmesh_path, with_outline=True):
    res = worker_call({"cmd": "read", "path": navmesh_path})
    if not res.get("ok"):
        return None
    if with_outline:
        try:
            tris = np.frombuffer(base64.b64decode(res["tris_b64"]), np.float32)
            res["outline"] = navmesh_outline(tris)
        except Exception:
            res["outline"] = None
    return res


# --------------------------------------------------------------------------- #
#  walkable polygons -> proxy surface (OBJ) -> _nav_worker (Recast) -> .navmesh
# --------------------------------------------------------------------------- #
def _pip(points, ring):
    x, y = points[:, 0], points[:, 1]
    rx, ry = ring[:, 0], ring[:, 1]
    inside = np.zeros(len(points), bool)
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi, xj, yj = rx[i], ry[i], rx[j], ry[j]
        cond = ((yi > y) != (yj > y)) & \
               (x < (xj - xi) * (y - yi) / (yj - yi + 1e-20) + xi)
        inside ^= cond
        j = i
    return inside


def rasterize_surface(polys, res):
    V, F = [], []
    for poly in polys:
        outer = np.asarray(poly["outer"], np.float64)
        if len(outer) < 3:
            continue
        fy = float(poly["floor_y"])
        holes = [np.asarray(h, np.float64) for h in poly.get("holes", []) if len(h) >= 3]
        minx, minz = outer.min(0)
        maxx, maxz = outer.max(0)
        nx = max(1, int(np.ceil((maxx - minx) / res)))
        nz = max(1, int(np.ceil((maxz - minz) / res)))
        if nx * nz > 600_000:
            res = max(res, np.sqrt((maxx - minx) * (maxz - minz) / 600_000))
            nx = max(1, int(np.ceil((maxx - minx) / res)))
            nz = max(1, int(np.ceil((maxz - minz) / res)))
        xs = minx + (np.arange(nx) + 0.5) * res
        zs = minz + (np.arange(nz) + 0.5) * res
        gx, gz = np.meshgrid(xs, zs)
        pts = np.column_stack([gx.ravel(), gz.ravel()])
        ins = _pip(pts, outer)
        for h in holes:
            ins &= ~_pip(pts, h)
        h2 = res / 2.0
        for (cx, cz) in pts[ins]:
            k = len(V)
            V.extend([[cx - h2, fy, cz - h2], [cx + h2, fy, cz - h2],
                      [cx + h2, fy, cz + h2], [cx - h2, fy, cz + h2]])
            F.extend([[k, k + 2, k + 1], [k, k + 3, k + 2]])   # up-facing
    if not V:
        return None, None
    return np.asarray(V, np.float64), np.asarray(F, np.int64)


def write_obj(path, V, F):
    """Write a plain Wavefront OBJ (no extra deps) with up-facing winding.
    OBJ (not GLB): habitat reframes GLB stages (Z-up->Y-up), which would rotate
    this horizontal sheet to vertical -> empty navmesh. OBJ loads identity-framed."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    if float(np.cross(b - a, c - a)[:, 1].mean()) < 0:   # make the sheet face +Y
        F = F[:, ::-1]
    lines = [f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}" for v in V]
    lines += [f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}" for t in F]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def bake(scene_name, polys, params):
    # Input grid matches Recast's cell_size; rasterize_surface() caps cell count
    # for huge areas.
    cell = float(params.get("cell_size", 0.03))
    res = min(max(cell, 0.02), 0.1)
    V, F = rasterize_surface(polys, res)
    if V is None:
        return {"ok": False, "error": "walkable area is empty — draw a polygon first"}
    # Sanitize the scene name before using it in filesystem paths (defensive: it
    # comes from the request body).
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", scene_name)
    # Unique filename per bake: habitat caches loaded assets by path, so reusing the
    # same proxy path would let a stale cached mesh survive and ignore edits.
    for old in glob.glob(os.path.join(TMP, f"{safe}_proxy_*.obj")):
        try:
            os.remove(old)
        except OSError:
            pass
    _bake_seq["n"] += 1
    proxy = os.path.join(TMP, f"{safe}_proxy_{_bake_seq['n']}.obj")
    write_obj(proxy, V, F)
    nav = os.path.join(TMP, f"{safe}.navmesh")
    if os.path.exists(nav):
        os.remove(nav)
    res2 = worker_call({"cmd": "bake", "obj": proxy, "out": nav, "params": params})
    if not res2.get("ok"):
        return res2                       # surfaces worker error/log to the UI
    _last_bake[scene_name] = nav
    return res2


def save_navmesh(scene_name, ply_path):
    nav = _last_bake.get(scene_name)
    if not nav or not os.path.exists(nav):
        return {"ok": False, "error": "nothing baked yet"}
    sdir, name, _ = scene_dir_and_split(ply_path)
    dest = os.path.join(sdir, name + ".navmesh")
    import shutil
    shutil.copyfile(nav, dest)
    return {"ok": True, "path": dest}


# --------------------------------------------------------------------------- #
#  HTTP
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.route("/static/<path:p>")
def static_files(p):
    return send_from_directory(STATIC, p)


@app.route("/api/scenes")
def api_scenes():
    return jsonify({"scenes": list_scenes(), "default": app.config["DEFAULT_SCENE"]})


@app.route("/api/scene")
def api_scene():
    name = request.args.get("name", "")
    ply = resolve_ply(name)
    if not ply:
        return jsonify({"error": f"scene not found: {name}"}), 404
    meta = prepare_scene(ply)
    app.config["CUR_PLY"][name] = ply
    sd, sname, split = scene_dir_and_split(ply)
    existing_nav = os.path.join(sd, sname + ".navmesh")
    nav = navmesh_payload(existing_nav) if os.path.exists(existing_nav) else None
    return jsonify({
        "name": name, "split": split,
        "splat_url": f"/api/splat?name={name}",
        "n_splats": meta["n_splats"], "n_total": meta["n_total"],
        "bounds": meta["bounds"], "floor_y": meta["floor_y"],
        "ceiling_y": meta.get("ceiling_y"),
        "suggested": meta["suggested"], "existing_navmesh": nav,
    })


@app.route("/api/splat")
def api_splat():
    name = request.args.get("name", "")
    ply = app.config["CUR_PLY"].get(name) or resolve_ply(name)
    if not ply:
        return jsonify({"error": "scene not found"}), 404
    meta = prepare_scene(ply)
    return send_file(meta["splat"], mimetype="application/octet-stream",
                     as_attachment=False, conditional=True)


@app.route("/api/bake", methods=["POST"])
def api_bake():
    body = request.get_json(silent=True) or {}
    name = body.get("scene")
    if not name or not (app.config["CUR_PLY"].get(name) or resolve_ply(name)):
        return jsonify({"ok": False, "error": "scene not loaded"}), 400
    with _lock:
        return jsonify(bake(name, body.get("polys", []), body.get("params", {})))


@app.route("/api/save", methods=["POST"])
def api_save():
    body = request.get_json(silent=True) or {}
    name = body.get("scene")
    ply = (app.config["CUR_PLY"].get(name) or resolve_ply(name)) if name else None
    if not ply:
        return jsonify({"ok": False, "error": "scene not loaded"}), 400
    with _lock:
        return jsonify(save_navmesh(name, ply))


def main():
    global GS_DIR, HABITAT_PY
    ap = argparse.ArgumentParser(
        description="navmesh_editor — draw a walkable area on a 3DGS scene and bake a Habitat .navmesh")
    ap.add_argument("--scene", default=None, help="scene name/dir to open first")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--gs-dir", default=GS_DIR,
                    help="scene directory: a folder whose subdirs hold GS scenes — "
                         "either scene dirs directly (e.g. .../gs_scenes/train) or "
                         "grouped one level deeper (e.g. gs_scenes with train/, val/)")
    ap.add_argument("--habitat-python", default=HABITAT_PY,
                    help="Python interpreter that has habitat_sim (default: this interpreter)")
    args = ap.parse_args()
    GS_DIR = os.path.abspath(args.gs_dir)
    HABITAT_PY = args.habitat_python

    scenes = list_scenes()
    default = args.scene
    if not default and scenes:
        default = scenes[0]["name"]
    app.config["DEFAULT_SCENE"] = default
    app.config["CUR_PLY"] = {}
    print(f"navmesh_editor: {len(scenes)} scenes  default={default}")
    print(f"  gs_dir: {GS_DIR}")
    print(f"  open  http://{args.host}:{args.port}")
    print(f"  (ssh -L {args.port}:localhost:{args.port} <server>  then open in your browser)")
    # Pre-warm the navmesh worker (the one-time habitat_sim import) in the
    # background so the first save is already fast.
    threading.Thread(target=lambda: worker_call({"cmd": "ping"}), daemon=True).start()
    print("  warming navmesh worker (habitat_sim import) in background…")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
