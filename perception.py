"""Astra 6 perception: find the objects with an RGB-D camera, the way a real cell would.

Nothing here reads object poses from the simulator. MuJoCo only renders what the fixed camera
sees (colour + depth, with sensor noise added). From those pixels the pipeline
  1. labels pixels by colour against the known object appearance models,
  2. lifts the labelled pixels to 3D points with the depth image and the camera calibration,
  3. fits each object's known shape (box / cylinder / sphere) to its points to get its pose.
"""
import io
import math
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw

CAMERA = "scene_cam"
W, H = 640, 480
DEPTH_NOISE = 0.0012        # metres of depth noise at 1 m (grows with distance squared)
RGB_NOISE = 3.0             # colour noise, 0-255 scale
REJECT = {                  # coloured things that are not objects: bins and the Panda's wrist trim
    "left_bin": (0.16, 0.54, 0.54), "right_bin": (0.69, 0.48, 0.23),
    "wrist_blue": (0.04, 0.54, 0.78), "wrist_green": (0.0, 1.0, 0.0),
}


def _chroma(rgb):
    rgb = np.asarray(rgb, dtype=np.float32)
    return rgb / (rgb.sum(axis=-1, keepdims=True) + 1e-6)


# ---------- signed distance to each object model (object frame, metres) ----------
def _sdf_box(p, half):
    q = np.abs(p) - half
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)


def _sdf_cyl(p, r, hh):
    d = np.stack([np.linalg.norm(p[:, :2], axis=1) - r, np.abs(p[:, 2]) - hh], axis=1)
    return np.linalg.norm(np.maximum(d, 0), axis=1) + np.minimum(d.max(axis=1), 0)


def _sdf_ball(p, r):
    return np.linalg.norm(p, axis=1) - r


class Perception:
    def __init__(self, m, objects, fixtures=None, rng=None):
        """objects: id -> (name, shape, rgba string, ...) as in robot.OBJECTS.
        fixtures: id -> world xyz of fixed, surveyed parts of the cell (the bins), labelled in pictures."""
        self.m = m
        self.cam = m.camera(CAMERA).id
        self.renderer = mujoco.Renderer(m, H, W)
        self.f = (H / 2) / math.tan(math.radians(m.cam_fovy[self.cam]) / 2)
        self.rng = rng or np.random.default_rng()
        self.objects = objects
        self.fixtures = fixtures or {}
        self.ids = list(objects)
        protos = [_chroma([float(v) for v in o[2].split()[:3]]) for o in objects.values()]
        protos += [_chroma(c) for c in REJECT.values()]
        self.protos = np.array(protos)
        self.last = None            # latest result, see perceive()

    # ---------- sensor ----------
    def capture(self, d):
        r = self.renderer
        r.update_scene(d, camera=self.cam)
        rgb = r.render().astype(np.float32)
        r.enable_depth_rendering()
        r.update_scene(d, camera=self.cam)
        depth = r.render().copy()
        r.disable_depth_rendering()
        rgb = np.clip(rgb + self.rng.normal(0, RGB_NOISE, rgb.shape), 0, 255).astype(np.uint8)
        depth = depth + self.rng.normal(0, 1, depth.shape) * DEPTH_NOISE * depth ** 2
        return rgb, depth.astype(np.float32), d.cam_xpos[self.cam].copy(), d.cam_xmat[self.cam].reshape(3, 3).copy()

    def truth_segmentation(self, d):
        """Ground-truth object masks. Only used to grade the pipeline, never to run it."""
        r = self.renderer
        r.enable_segmentation_rendering()
        r.update_scene(d, camera=self.cam)
        seg = r.render().copy()
        r.disable_segmentation_rendering()
        return seg

    # ---------- pipeline ----------
    def perceive(self, d):
        rgb, depth, cpos, cmat = self.capture(d)
        dets = self.detect(rgb, depth, cpos, cmat)
        self.last = {"t": time.time(), "sim_t": float(d.time), "rgb": rgb, "dets": dets, "cam": (cpos, cmat)}
        return dets

    def detect(self, rgb, depth, cpos, cmat):
        # 1. colour labels: nearest appearance model in chromaticity, if the pixel is saturated enough
        f32 = rgb.astype(np.float32)
        mx, mn = f32.max(axis=2), f32.min(axis=2)
        sat = (mx - mn) / (mx + 1e-6)
        dist = np.linalg.norm(_chroma(f32)[:, :, None, :] - self.protos[None, None], axis=3)
        label = dist.argmin(axis=2)
        ok = (sat > 0.35) & (mx > 40) & (dist.min(axis=2) < 0.07)
        # 2. 3D points for every pixel (camera looks along -z, x right, y up)
        u = (np.arange(W) + 0.5 - W / 2) / self.f
        v = -(np.arange(H) + 0.5 - H / 2) / self.f
        pc = np.stack([u[None, :] * depth, v[:, None] * depth, -depth], axis=2)
        pw = pc @ cmat.T + cpos
        pad = np.pad(depth, 1, mode="edge")
        self._near_depth = np.min([pad[i:i + H, j:j + W] for i in range(3) for j in range(3)], axis=0)
        dets = {}
        for k, oid in enumerate(self.ids):
            mask = ok & (label == k)
            # drop edge pixels, where colour and depth mix with the background
            mask[1:-1, 1:-1] &= mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
            mask &= (pw[:, :, 2] > -0.01) & (pw[:, :, 2] < 0.25)
            if mask.sum() < 25:
                continue
            pts = pw[mask]
            # keep the main cluster (stray pixels elsewhere in the image)
            med = np.median(pts[:, :2], axis=0)
            near = np.linalg.norm(pts[:, :2] - med, axis=1) < 0.06
            if near.sum() < 25:
                continue
            pts = pts[near]
            rows, cols = np.nonzero(mask)
            rows, cols = rows[near], cols[near]
            p, yaw, rms, viol = self.fit(oid, pts, depth, cpos, cmat)
            dets[oid] = {"p": p, "yaw": yaw, "rms": rms, "viol": viol, "n": int(len(pts)),
                         "box": [int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())]}
        return dets

    def _model(self, oid):
        """Signed-distance function, half height and surface samples (with normals) of an object model."""
        shape = self.objects[oid][1]
        size = [float(v) for v in self.objects[oid][3].split()]
        g = np.linspace(-1, 1, 7)
        a, b = [v.ravel() for v in np.meshgrid(g, g)]
        if shape == "cube":
            h = np.array(size)
            samples, normals = [], []
            for ax in range(3):
                for sgn in (-1, 1):
                    p = np.zeros((len(a), 3)); n = np.zeros((len(a), 3))
                    o = [i for i in range(3) if i != ax]
                    p[:, ax] = sgn * h[ax]; p[:, o[0]] = a * h[o[0]] * 0.7; p[:, o[1]] = b * h[o[1]] * 0.7
                    n[:, ax] = sgn
                    samples.append(p); normals.append(n)
            return lambda q: _sdf_box(q, h), h[2], np.vstack(samples), np.vstack(normals)
        if shape == "cylinder":
            r, hh = size
            t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
            side = np.array([[r * math.cos(u), r * math.sin(u), z * hh * 0.7] for u in t for z in g])
            sn = np.array([[math.cos(u), math.sin(u), 0] for u in t for z in g])
            disk = np.array([[r * 0.7 * x, r * 0.7 * y] for x, y in zip(a, b) if x * x + y * y <= 1])
            caps = [np.c_[disk, np.full(len(disk), s * hh)] for s in (-1, 1)]
            cn = [np.tile([0, 0, s], (len(disk), 1)) for s in (-1, 1)]
            return (lambda q: _sdf_cyl(q, r, hh), hh, np.vstack([side] + caps), np.vstack([sn] + cn))
        r = size[0]
        k = np.arange(200) + 0.5                          # Fibonacci sphere
        phi, th = np.arccos(1 - 2 * k / 200), np.pi * (1 + 5 ** 0.5) * k
        n = np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]
        return lambda q: _sdf_ball(q, r), r, n * r, n[:, :] * 1.0

    def _free_space_violation(self, x, yaw, samples, normals, depth, cpos, cmat):
        """Fraction of the model's camera-facing surface that the depth image sees straight through.
        A true pose has ~0: wherever the object should be, the camera sees it or something in front of it."""
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        p = samples @ R.T + x
        n = normals @ R.T
        view = cpos - p
        facing = np.einsum("ij,ij->i", n, view) > 0.5 * np.linalg.norm(view, axis=1)   # clearly facing the camera
        p = p[facing]
        pc = (p - cpos) @ cmat                           # world -> camera frame
        z = -pc[:, 2]
        u = (pc[:, 0] / z * self.f + W / 2).astype(int)
        v = (-pc[:, 1] / z * self.f + H / 2).astype(int)
        inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not inside.any():
            return 1.0
        # nearest surface in a 3x3 window, so samples on a silhouette edge don't count as seen-through
        seen = self._near_depth[v[inside], u[inside]]
        return float(np.mean(seen > z[inside] + 0.006))

    def fit(self, oid, pts, depth, cpos, cmat):
        """Fit the object's known shape to its 3D points: robust Gauss-Newton on the signed distance,
        from several starts, keeping the best pose that does not contradict the depth image."""
        sdf, hz, samples, normals = self._model(oid)
        cube = self.objects[oid][1] == "cube"
        c = pts.mean(axis=0)
        away = c[:2] - cpos[:2]
        away /= np.linalg.norm(away) + 1e-9
        top = np.percentile(pts[:, 2], 97)

        def residual(x):
            yaw = x[3] if cube else 0.0
            cy, sy = math.cos(yaw), math.sin(yaw)
            q = pts - x[:3]
            local = np.stack([cy * q[:, 0] + sy * q[:, 1], -sy * q[:, 0] + cy * q[:, 1], q[:, 2]], axis=1)
            return sdf(local)

        best = None
        for k in (0.5, 1.0, 1.6):           # start behind the visible surface, at a few depths
            x0 = np.array([c[0] + away[0] * hz * k, c[1] + away[1] * hz * k, top - hz])
            if cube:   # coarse search for the face angle, then refine with the position
                yaw0 = min(np.radians(np.arange(0, 90, 6)), key=lambda a: np.sum(residual(np.append(x0, a)) ** 2))
                x0 = np.append(x0, yaw0)
            x = x0
            w = np.ones(len(pts))
            for _ in range(15):
                r = residual(x)
                J = np.empty((len(pts), len(x)))
                for i in range(len(x)):
                    dx = np.zeros(len(x)); dx[i] = 1e-4
                    J[:, i] = (residual(x + dx) - r) / 1e-4
                s = 1.4826 * np.median(np.abs(r)) + 1e-4        # robust scale; down-weight outliers (Huber)
                w = np.minimum(1.0, 2.0 * s / (np.abs(r) + 1e-9))
                A = J.T @ (J * w[:, None]) + 1e-6 * np.eye(len(x))
                step = np.linalg.solve(A, -J.T @ (w * r))
                n = np.linalg.norm(step[:3])
                if n > 0.008:                   # trust region: one face alone can't justify a big jump
                    step *= 0.008 / n
                x = x + step
                x[2] = max(x[2], hz - 0.002)    # objects rest on or above the table, never inside it
                if np.linalg.norm(step[:3]) < 2e-5:
                    break
            rms = float(np.sqrt(np.average(residual(x) ** 2, weights=w)))
            yaw = float(x[3]) if cube else 0.0
            viol = self._free_space_violation(x[:3], yaw, samples, normals, depth, cpos, cmat)
            score = rms + 0.02 * viol       # a pose the camera sees through costs more than any fit error
            if best is None or score < best[0]:
                best = (score, x[:3].copy(), yaw, rms, viol)
        _, p, yaw, rms, viol = best
        if cube:
            yaw = (yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4
        return p, yaw, rms, viol

    # ---------- pictures ----------
    def project(self, p, cam):
        cpos, cmat = cam
        pc = (np.asarray(p) - cpos) @ cmat
        return pc[0] / -pc[2] * self.f + W / 2, -pc[1] / -pc[2] * self.f + H / 2

    def annotated_jpeg(self, quality=80, labels=True):
        """The latest camera frame with a labelled box per detection and a tag on each fixture
        (set-of-marks for the model: it reads ids off the picture instead of reasoning about frames)."""
        if not self.last:
            return None
        img = Image.fromarray(self.last["rgb"])
        if labels:
            draw = ImageDraw.Draw(img)
            for fid, p in self.fixtures.items():
                u, v = self.project(p, self.last["cam"])
                tw = draw.textlength(fid)
                draw.rectangle([u - tw / 2 - 4, v - 8, u + tw / 2 + 4, v + 7], fill=(250, 250, 250), outline=(20, 20, 20))
                draw.text((u - tw / 2, v - 6), fid, fill=(20, 20, 20))
            for oid, det in self.last["dets"].items():
                x0, y0, x1, y1 = det["box"]
                draw.rectangle([x0 - 3, y0 - 3, x1 + 3, y1 + 3], outline=(255, 255, 255), width=2)
                tw = draw.textlength(oid)
                ty = max(0, y0 - 18)
                draw.rectangle([x0 - 3, ty, x0 + tw + 5, ty + 14], fill=(20, 20, 20))
                draw.text((x0 + 1, ty + 1), oid, fill=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=quality)
        return buf.getvalue()
