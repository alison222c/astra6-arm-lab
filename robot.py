"""MuJoCo workcell for the Astra 6 demo: Franka Panda, tabletop objects, IK controller, action executor.

The executor works from what the robot can sense, like a real cell: object poses come from the RGB-D
camera (perception.py), holding something from the finger width, touching down from a wrist force
sensor. The simulator's true poses are used only in "truth" mode, for comparison, and to grade.
"""
import asyncio
import base64
import math
import re
from pathlib import Path

import mujoco
import numpy as np

from perception import Perception

ROOT = Path(__file__).parent
PANDA_DIR = ROOT / "vendor" / "menagerie" / "franka_emika_panda"

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
HOME_Q = np.array([0, -0.3, 0, -2.2, 0, 1.9, 0.785])
TCP_OFFSET = 0.1034          # hand origin -> fingertip pad centre, along hand z
GRIP_OPEN, GRIP_CLOSED = 255.0, 0.0
WORKSPACE = np.array([[0.15, 0.85], [-0.65, 0.65], [0.015, 0.75]])

OBJECTS = {
    # id: (name, shape, colour rgba, mujoco size, start x, y, yaw)
    "red_cube": ("red cube", "cube", "0.82 0.23 0.19 1", "0.02 0.02 0.02", 0.50, 0.15, 0.3),
    "blue_cube": ("blue cube", "cube", "0.18 0.44 0.84 1", "0.02 0.02 0.02", 0.60, -0.10, -0.4),
    "green_cube": ("green cube", "cube", "0.18 0.62 0.34 1", "0.02 0.02 0.02", 0.42, -0.20, 0.7),
    "yellow_cylinder": ("yellow cylinder", "cylinder", "0.91 0.72 0.09 1", "0.02 0.03", 0.56, 0.03, 0.0),
    "purple_ball": ("purple ball", "ball", "0.54 0.31 0.82 1", "0.024", 0.66, 0.20, 0.0),
}
BINS = {"left_bin": ("left bin", 0.30, 0.45, "0.16 0.54 0.54 1"), "right_bin": ("right bin", 0.30, -0.45, "0.69 0.48 0.23 1")}
BIN_HALF, BIN_WALL_H, BIN_FLOOR = 0.08, 0.05, 0.01
CARRY_SPEED = 1.0            # speed limit while holding something
SEAT_DEPTH = 0.003           # how far to press on after touchdown before letting go
TOUCH_N = 0.2                # change in wrist load (N) that means the held object has touched down
SPOTS = {"left": (0.45, 0.30), "right": (0.45, -0.30), "center": (0.50, 0.0), "front": (0.66, 0.0), "back": (0.34, 0.0)}


class Cancelled(Exception):
    pass


class ActionError(Exception):
    pass


def _build_xml() -> str:
    """Panda from MuJoCo Menagerie plus our objects. The Menagerie keyframe is dropped because it
    only covers the arm's 9 qpos and would not match once free-floating objects are added."""
    src = (PANDA_DIR / "panda.xml").read_text()
    src = re.sub(r"<keyframe>.*?</keyframe>", "", src, flags=re.S)
    src = src.replace('<option integrator="implicitfast"/>',
                      '<option integrator="implicitfast" timestep="0.002" cone="elliptic" impratio="10"/>')
    hand = '<body name="hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">'
    src = src.replace(hand, hand + '<site name="wrist_ft"/>')           # wrist force sensor, as on the real Panda
    src = src.replace("</mujoco>", '<sensor><force name="wrist_force" site="wrist_ft"/></sensor></mujoco>')
    (PANDA_DIR / "astra_panda.xml").write_text(src)

    body = []
    for oid, (_, shape, rgba, size, x, y, yaw) in OBJECTS.items():
        half = {"cube": 0.02, "cylinder": 0.03, "ball": 0.024}[shape]
        gtype = {"cube": "box", "cylinder": "cylinder", "ball": "sphere"}[shape]
        extra = ' condim="6" friction="1.2 0.01 0.0005"' if shape == "ball" else ' condim="4" friction="1.2 0.01 0.001"'
        body.append(f'<body name="{oid}" pos="{x} {y} {half + 0.001}" euler="0 0 {yaw}"><freejoint/>'
                    f'<geom name="{oid}" type="{gtype}" size="{size}" rgba="{rgba}" mass="0.05"{extra}/></body>')
    for bid, (_, x, y, rgba) in BINS.items():
        h, t = BIN_HALF, 0.005
        walls = [f'<geom type="box" size="{h} {h} {BIN_FLOOR / 2}" pos="0 0 {BIN_FLOOR / 2}" rgba="{rgba}"/>']
        for sx, sy, w, d in [(1, 0, t, h), (-1, 0, t, h), (0, 1, h, t), (0, -1, h, t)]:
            walls.append(f'<geom type="box" size="{w} {d} {BIN_WALL_H / 2}" pos="{sx * (h - t)} {sy * (h - t)} {BIN_WALL_H / 2}" rgba="{rgba}"/>')
        body.append(f'<body name="{bid}" pos="{x} {y} 0">{"".join(walls)}</body>')

    return f"""<mujoco model="astra6 cell">
  <include file="astra_panda.xml"/>
  <visual><global offwidth="640" offheight="480"/><map znear="0.02" zfar="4"/></visual>
  <worldbody>
    <!-- fixed RGB-D camera in front of the table, looking back at the robot (the perception sensor) -->
    <camera name="scene_cam" pos="1.2 0 0.75" xyaxes="0 1 0 -0.707 0 0.707" fovy="52"/>
    <geom name="floor" type="plane" size="1.5 1.5 0.05" rgba="0.78 0.8 0.82 1"/>
    {"".join(body)}
  </worldbody>
</mujoco>"""


def _rot_target(yaw: float) -> np.ndarray:
    """Hand pointing straight down, rotated by yaw about world z."""
    c, s = math.cos(yaw), math.sin(yaw)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return rz @ np.diag([1.0, -1.0, -1.0])


def _wrap_quarter(a: float) -> float:
    """Cube faces repeat every 90 degrees: pick the equivalent yaw closest to zero."""
    return (a + math.pi / 4) % (math.pi / 2) - math.pi / 4


class Cell:
    def __init__(self):
        xml = _build_xml()
        (PANDA_DIR / "astra_scene.xml").write_text(xml)
        self.m = mujoco.MjModel.from_xml_path(str(PANDA_DIR / "astra_scene.xml"))
        self.d = mujoco.MjData(self.m)
        self.ik = mujoco.MjData(self.m)
        self.qadr = np.array([self.m.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.dadr = np.array([self.m.joint(j).dofadr[0] for j in ARM_JOINTS])
        self.qrange = np.array([self.m.joint(j).range for j in ARM_JOINTS])
        self.hand = self.m.body("hand").id
        self.finger_bodies = {self.m.body("left_finger").id, self.m.body("right_finger").id}
        self.obj_body = {oid: self.m.body(oid).id for oid in OBJECTS}
        self.speed = 1.0
        self.cancel = False
        self.teleop_v = np.zeros(3)
        self.teleop_w = 0.0
        self.vision = Perception(self.m, OBJECTS, {b: (x, y, -0.03) for b, (_, x, y, _) in BINS.items()})
        self.sense = "vision"         # "vision": act on camera estimates; "truth": act on simulator poses
        self.ft_site = self.m.site("wrist_ft").id
        self.ft_adr = self.m.sensor("wrist_force").adr[0]
        self._look_lock = asyncio.Lock()
        self.reset()

    # ---------- state ----------
    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.qadr] = HOME_Q
        for f in ("finger_joint1", "finger_joint2"):
            self.d.qpos[self.m.joint(f).qposadr[0]] = 0.04
        self.d.ctrl[:7] = HOME_Q
        self.d.ctrl[7] = GRIP_OPEN
        mujoco.mj_forward(self.m, self.d)
        self.q_des = HOME_Q.copy()
        p, R = self.tcp()
        self.goal_p, self.goal_yaw = p.copy(), self._yaw_of(R)
        self.set_p, self.set_yaw = p.copy(), self.goal_yaw
        self.track_pose = False
        self.world = {}               # object id -> what the camera last saw (see perceive)
        self.held_id = None           # what the gripper is holding, as far as the robot knows

    def shuffle(self, rng=np.random):
        self.reset()
        placed = []
        for oid in OBJECTS:
            for _ in range(300):
                r, a = rng.uniform(0.38, 0.68), rng.uniform(-0.75, 0.75)
                x, y = r * math.cos(a), r * math.sin(a)
                if any(math.hypot(px - x, py - y) < 0.09 for px, py in placed):
                    continue
                if any(math.hypot(bx - x, by - y) < 0.16 for _, bx, by, _ in BINS.values()):
                    continue
                placed.append((x, y))
                jadr = self.m.body(oid).jntadr[0]
                qa = self.m.jnt_qposadr[jadr]
                half = self.half(oid)
                yaw = rng.uniform(0, math.pi)
                self.d.qpos[qa:qa + 7] = [x, y, half + 0.001, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
                break
        mujoco.mj_forward(self.m, self.d)

    def tcp(self, data=None):
        data = data or self.d
        R = data.xmat[self.hand].reshape(3, 3)
        return data.xpos[self.hand] + R[:, 2] * TCP_OFFSET, R

    @staticmethod
    def _yaw_of(R):
        return math.atan2(R[1, 0], R[0, 0])

    def half(self, oid):
        shape = OBJECTS[oid][1]
        return {"cube": 0.02, "cylinder": 0.03, "ball": 0.024}[shape]

    # simulator ground truth: truth mode and grading only
    def obj_pos(self, oid):
        return self.d.xpos[self.obj_body[oid]].copy()

    def obj_yaw(self, oid):
        return self._yaw_of(self.d.xmat[self.obj_body[oid]].reshape(3, 3))

    # ---------- what the robot senses ----------
    async def perceive(self):
        """Take a camera frame and update the world model. Rendering stays on this thread (it owns
        the GL context); the image processing runs in a worker so the physics keeps time."""
        async with self._look_lock:
            frame = self.vision.capture(self.d)
            sim_t = float(self.d.time)
            dets = await asyncio.to_thread(self.vision.detect, *frame)
            self.vision.last = {"sim_t": sim_t, "rgb": frame[0], "dets": dets, "cam": frame[2:]}
            for oid in OBJECTS:
                if oid in dets:
                    self.world[oid] = {"p": dets[oid]["p"], "yaw": dets[oid]["yaw"], "seen": sim_t, "visible": True}
                elif oid in self.world:
                    self.world[oid]["visible"] = False
            return dets

    def est_pos(self, oid):
        if self.sense == "truth":
            return self.obj_pos(oid)
        if oid not in self.world:
            raise ActionError(f"the camera can't see the {OBJECTS[oid][0]}")
        return self.world[oid]["p"].copy()

    def est_yaw(self, oid):
        return self.obj_yaw(oid) if self.sense == "truth" else self.world[oid]["yaw"]

    def known(self):
        return list(OBJECTS) if self.sense == "truth" else [o for o in OBJECTS if o in self.world]

    def wrist_force(self):
        """Force through the wrist, world frame (N)."""
        R = self.d.site_xmat[self.ft_site].reshape(3, 3)
        return R @ self.d.sensordata[self.ft_adr:self.ft_adr + 3]

    def grip_width(self):
        return float(sum(self.d.qpos[self.m.joint(f).qposadr[0]] for f in ("finger_joint1", "finger_joint2")))

    def grasped(self):
        """What the gripper holds. Vision mode: the fingers closed but stopped apart means something is
        between them; which object is what was picked, or the nearest one the camera saw (keyboard grasps)."""
        if self.sense == "truth":
            return self._contact_grasp()
        if self.d.ctrl[7] != GRIP_CLOSED:
            self.held_id = None
            return None
        if self.grip_width() < 0.008:
            return None
        if self.held_id is None:
            tcp = self.tcp()[0]
            near = [(np.linalg.norm(self.world[o]["p"] - tcp), o) for o in self.world]
            near = [n for n in near if n[0] < 0.05]
            self.held_id = min(near)[1] if near else None
        return self.held_id

    def _contact_grasp(self):
        """Object touched by both fingers, from the physics contacts (truth mode)."""
        touch = {}
        for c in self.d.contact[: self.d.ncon]:
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            for fb, ob in ((b1, b2), (b2, b1)):
                if fb in self.finger_bodies:
                    for oid, bid in self.obj_body.items():
                        if ob == bid:
                            touch.setdefault(oid, set()).add(fb)
        both = [oid for oid, f in touch.items() if len(f) == 2]
        return both[0] if both else None

    def support_of(self, oid):
        p = self.est_pos(oid)
        bottom = p[2] - self.half(oid)
        for other in self.known():
            if other == oid or other == self.grasped():
                continue
            q = self.est_pos(other)
            if math.hypot(*(p[:2] - q[:2])) < 0.035 and abs(bottom - (q[2] + self.half(other))) < 0.012:
                return other
        for bid, (_, bx, by, _) in BINS.items():
            if abs(p[0] - bx) < BIN_HALF and abs(p[1] - by) < BIN_HALF:
                return bid
        if bottom > 0.02:
            return "held" if self.grasped() == oid else "air"
        return "table"

    def describe(self):
        held = self.grasped()
        lines = []
        for oid, (name, shape, *_rest) in OBJECTS.items():
            if oid == held:
                lines.append(f"- {oid}: {name} ({shape}), currently held in the gripper")
                continue
            if oid not in self.known():
                lines.append(f"- {oid}: {name} ({shape}), not seen by the camera")
                continue
            p = self.est_pos(oid)
            line = f"- {oid}: {name} ({shape}), at x {p[0]:.3f} y {p[1]:.3f} z {p[2]:.3f}, resting on {self.support_of(oid)}"
            if self.sense == "vision" and not self.world[oid]["visible"]:
                line += f" (hidden from the camera now; last seen {self.d.time - self.world[oid]['seen']:.0f} s ago)"
            lines.append(line)
        return "\n".join(lines), held

    # ---------- control ----------
    def _ik_step(self, iters=3):
        Rt = _rot_target(self.set_yaw)
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        for _ in range(iters):
            self.ik.qpos[:] = self.d.qpos
            self.ik.qpos[self.qadr] = self.q_des
            mujoco.mj_kinematics(self.m, self.ik)
            mujoco.mj_comPos(self.m, self.ik)
            p, R = self.tcp(self.ik)
            ep = self.set_p - p
            er = 0.5 * sum(np.cross(R[:, i], Rt[:, i]) for i in range(3))
            e = np.concatenate([ep, er])
            if np.linalg.norm(e) < 1e-4:
                break
            mujoco.mj_jac(self.m, self.ik, jacp, jacr, p, self.hand)
            J = np.vstack([jacp[:, self.dadr], jacr[:, self.dadr]])
            lam = 0.03
            JJt = J @ J.T + lam ** 2 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            Jpinv = J.T @ np.linalg.inv(JJt)
            dq += (np.eye(7) - Jpinv @ J) @ (0.05 * (HOME_Q - self.q_des))
            n = np.linalg.norm(dq)
            if n > 0.2:
                dq *= 0.2 / n
            self.q_des = np.clip(self.q_des + dq, self.qrange[:, 0] + 0.01, self.qrange[:, 1] - 0.01)

    def control(self, dt):
        """Called at the control rate: move the setpoint toward the goal (or by teleop), then IK."""
        if np.any(self.teleop_v) or self.teleop_w:
            self.goal_p = np.clip(self.goal_p + self.teleop_v * dt, WORKSPACE[:, 0], WORKSPACE[:, 1])
            self.goal_yaw = float(np.clip(self.goal_yaw + self.teleop_w * dt, -2.6, 2.6))
            self.track_pose = True
        if self.track_pose:
            # carry more gently: a held object, gripped near its top, swings loose at full speed
            speed = min(self.speed, CARRY_SPEED) if self.d.ctrl[7] == GRIP_CLOSED and self.grip_width() > 0.008 else self.speed
            vmax, wmax = 0.28 * speed * dt, 1.6 * speed * dt
            d = self.goal_p - self.set_p
            n = np.linalg.norm(d)
            self.set_p = self.goal_p.copy() if n <= vmax else self.set_p + d / n * vmax
            dy = self.goal_yaw - self.set_yaw
            self.set_yaw = self.goal_yaw if abs(dy) <= wmax else self.set_yaw + math.copysign(wmax, dy)
            self._ik_step()
        self.d.ctrl[:7] = self.q_des

    def step(self, n):
        for _ in range(n):
            # gravity compensation, as on the real Panda controller: without it the
            # position servos sag a few millimetres and grasps land off-centre
            self.d.qfrc_applied[self.dadr] = self.d.qfrc_bias[self.dadr]
            mujoco.mj_step(self.m, self.d)

    # ---------- primitive motions (awaitable) ----------
    async def _wait(self, cond, timeout):
        t = 0.0
        while t < timeout:
            if self.cancel:
                raise Cancelled()
            if cond():
                return True
            await asyncio.sleep(0.02)
            t += 0.02
        return False

    async def move(self, p, yaw, what="that spot", tol=0.012, timeout=6.0):
        p = np.array(p, dtype=float)
        if not (WORKSPACE[0, 0] <= math.hypot(p[0], p[1]) <= 0.86 and p[2] <= WORKSPACE[2, 1]):
            raise ActionError(f"{what} is out of reach")
        p = np.clip(p, WORKSPACE[:, 0], WORKSPACE[:, 1])
        self.goal_p, self.goal_yaw, self.track_pose = p, yaw, True
        def arrived():
            tcp, R = self.tcp()
            yaw_err = abs((self._yaw_of(R) - yaw + math.pi) % (2 * math.pi) - math.pi)
            return (np.allclose(self.set_p, p) and self.set_yaw == yaw
                    and np.linalg.norm(tcp - p) < tol and yaw_err < 0.03)
        ok = await self._wait(arrived, timeout / self.speed)
        return ok

    async def move_joints(self, q, timeout=6.0):
        self.track_pose = False
        start = self.q_des.copy()
        q = np.array(q)
        steps = max(10, int(np.max(np.abs(q - start)) / (0.9 * self.speed) / 0.02))
        for i in range(1, steps + 1):
            if self.cancel:
                raise Cancelled()
            s = i / steps
            s = s * s * (3 - 2 * s)
            self.q_des = start + (q - start) * s
            await asyncio.sleep(0.02)
        await self._wait(lambda: np.max(np.abs(self.d.qpos[self.qadr] - q)) < 0.02, timeout)
        p, R = self.tcp()
        self.goal_p, self.goal_yaw = p.copy(), self._yaw_of(R)
        self.set_p, self.set_yaw = p.copy(), self.goal_yaw

    async def gripper(self, open_, width=None):
        """Open (fully, or to `width` metres between the fingers) or close the gripper."""
        self.d.ctrl[7] = (GRIP_OPEN if width is None else GRIP_OPEN * min(width, 0.08) / 0.08) if open_ else GRIP_CLOSED
        last = [self.grip_width()]

        def settled():
            w = self.grip_width()
            done = abs(w - last[0]) < 1e-4
            last[0] = w
            return done
        await asyncio.sleep(0.15)
        await self._wait(settled, 1.5)

    # ---------- actions ----------
    async def run(self, a):
        t = a.get("type")
        safe_z = 0.24
        if t == "home":
            await self.move_joints(HOME_Q)
        elif t == "gripper":
            await self.gripper(a.get("state") != "close")
        elif t == "wave":
            base = HOME_Q.copy()
            up = base.copy(); up[1], up[3], up[5] = 0.2, -1.2, 1.4
            await self.move_joints(up)
            for _ in range(3):
                l = up.copy(); l[6] += 0.9; l[4] += 0.5
                r = up.copy(); r[6] -= 0.9; r[4] -= 0.5
                await self.move_joints(l, 1.0)
                await self.move_joints(r, 1.0)
            await self.move_joints(HOME_Q)
        elif t == "point":
            oid = self._obj(a.get("object"))
            await self.perceive()
            p = self.est_pos(oid)
            held = self.grasped()
            if held == oid:
                raise ActionError("that object is in the gripper")
            if not held:
                await self.gripper(False)
            await self.move([p[0], p[1], p[2] + self.half(oid) + 0.08], self.set_yaw, f"The {OBJECTS[oid][0]}")
            await asyncio.sleep(0.6 / self.speed)
            await self.move([p[0], p[1], max(safe_z, p[2] + 0.16)], self.set_yaw, f"The {OBJECTS[oid][0]}")
            if not held:
                await self.gripper(True)
        elif t == "pick":
            await self._pick(self._obj(a.get("object")), safe_z)
        elif t == "place":
            await self._place(a.get("target"), safe_z)
        else:
            raise ActionError(f'unknown action "{t}"')

    def _obj(self, oid):
        if oid not in OBJECTS:
            raise ActionError(f'no object called "{oid}"')
        return oid

    async def _pick(self, oid, safe_z):
        name = OBJECTS[oid][0]
        held = self.grasped()
        if held == oid:
            return
        if held:
            raise ActionError(f"already holding the {OBJECTS[held][0]}; place it first")
        await self.perceive()
        on_top = [o for o in self.known() if o != oid and self.support_of(o) == oid]
        if on_top:
            raise ActionError(f"the {OBJECTS[on_top[0]][0]} is on top of it")
        p = self.est_pos(oid)
        cube_yaw = self.est_yaw(oid) if OBJECTS[oid][1] == "cube" else None
        yaw = self._finger_yaw(p, oid, cube_yaw)
        what = f"The {name}"
        await self.gripper(True, self.width(oid) + 0.025)    # open just enough: less to bump into
        await self.move([p[0], p[1], max(safe_z, p[2] + 0.13)], yaw, what)
        await self.move([p[0], p[1], p[2] + 0.06], yaw, what)
        await self.perceive()          # look again from close by: it may have been nudged
        p = self.est_pos(oid)
        if cube_yaw is not None:
            yaw = self._finger_yaw(p, oid, self.est_yaw(oid))
        # grip at most 2 cm below the top: deeper, a tall object's top wedges into the finger bodies
        grasp_z = max(p[2], p[2] + self.half(oid) - 0.02)
        self.f_empty = self.wrist_force()[2]      # wrist load with an empty hand, for unloading at the place
        await self.move([p[0], p[1], grasp_z + 0.002], yaw, what, tol=0.008, timeout=3.0)
        z0 = p[2]
        self.held_id = oid
        await self.gripper(False)
        await asyncio.sleep(0.2)
        self.grasp_dz = float(z0 - self.tcp()[0][2])   # object centre relative to the TCP
        await self.move([p[0], p[1], max(safe_z, z0 + 0.15)], yaw, what)
        if self.sense == "vision":
            await self.perceive()
            seen = self.world[oid]
            lifted = self.grasped() == oid and (not seen["visible"] or seen["p"][2] > z0 + 0.05)
        else:
            lifted = self.obj_pos(oid)[2] > z0 + 0.05 and self.grasped() == oid
        if not lifted:
            await self.gripper(True)
            raise ActionError(f"the grasp slipped; the {name} was not lifted")

    async def _place(self, target, safe_z):
        oid = self.grasped()
        if not oid:
            raise ActionError("the gripper is not holding anything")
        t = str(target or "").strip().lower().replace(" ", "_")
        t = "center" if t == "middle" else t
        half = self.half(oid)
        await self.perceive()
        if t in OBJECTS:
            if t == oid:
                raise ActionError("can't place an object on itself")
            if OBJECTS[t][1] == "ball":
                raise ActionError("nothing stays on top of the purple ball")
            if any(self.support_of(o) == t for o in self.known() if o != oid):
                raise ActionError(f"the {OBJECTS[t][0]} already has something on it")
            q = self.est_pos(t)
            x, y, rest = q[0], q[1], q[2] + self.half(t)
            what = f"The {OBJECTS[t][0]}"
        elif t in BINS:
            _, bx, by, _ = BINS[t]
            # slots in a plus pattern: each leaves the fingers a direction clear of the walls
            slots = [(0, -0.036), (0, 0.036), (-0.036, 0), (0.036, 0), (0, 0)]
            others = [self.est_pos(o)[:2] for o in self.known() if o != oid and self.support_of(o) == t]
            sx, sy = max(slots, key=lambda s: min([np.linalg.norm(np.array([bx + s[0], by + s[1]]) - q) for q in others] + [1.0]))
            x, y, rest = bx + sx, by + sy, BIN_FLOOR
            what = f"The {BINS[t][0]}"
        elif t in SPOTS:
            x, y = self._free_spot(*SPOTS[t], oid)
            rest = 0.0
            what = f"The {t} spot"
        else:
            raise ActionError(f'unknown place "{target}"')
        # a held cube can turn by quarter turns, a round object any way: pick the roomiest for the fingers
        held_yaw = self.set_yaw if OBJECTS[oid][1] == "cube" else None
        yaw = self._finger_yaw(np.array([x, y, rest]), oid, held_yaw, ignore=t)
        tcp_z = rest + half - getattr(self, "grasp_dz", 0.0)   # TCP height at which the object would just touch down
        await self.move([x, y, max(safe_z, tcp_z + 0.14)], yaw, what)
        await self.move([x, y, tcp_z + 0.045], yaw, what)
        # lower slowly until the wrist feels the support take the object's weight, then let go
        await asyncio.sleep(0.2)
        speed, self.speed = self.speed, 0.15
        try:
            self.goal_p = np.array([x, y, tcp_z - 0.012])
            # once the slow descent is steady, the load the wrist carries is the baseline
            samples = []
            await self._wait(lambda: samples.append(self.wrist_force()[2]) or len(samples) >= 20, 1.5)
            base, ema = float(np.mean(samples[8:])), [float(np.mean(samples[8:]))]

            def touched():
                ema[0] += 0.35 * (self.wrist_force()[2] - ema[0])
                return base - ema[0] > TOUCH_N or np.allclose(self.set_p, self.goal_p)
            await self._wait(touched, 4.0)
            # seat it: press on a few mm so an object touching down on one edge settles flat
            self.goal_p = self.set_p - [0, 0, SEAT_DEPTH]
            await self._wait(lambda: np.allclose(self.set_p, self.goal_p), 0.5)
            await asyncio.sleep(0.05)
        finally:
            self.speed = speed
        # unload before letting go: ease up until the wrist carries what an empty hand does, so the
        # support holds the object and nothing is pressed down; a pressed stack springs back on release
        self.goal_p = self.set_p.copy()
        f_empty, top = getattr(self, "f_empty", base), self.set_p[2] + SEAT_DEPTH + 0.004

        def unloaded():
            if self.wrist_force()[2] >= f_empty - 0.1 or self.goal_p[2] >= top:
                return True
            self.goal_p = self.goal_p + [0, 0, 0.0002]
            return False
        await self._wait(unloaded, 1.5)
        self.goal_p = self.set_p.copy()
        await asyncio.sleep(0.05)
        await self.gripper(True)
        self.held_id = None
        await self.move([x, y, max(safe_z, tcp_z + 0.14)], yaw, what)
        # look where it actually ended up
        await asyncio.sleep(0.4)
        await self.perceive()
        if self.sense == "vision" and not self.world[oid]["visible"]:
            raise ActionError(f"the camera lost sight of the {OBJECTS[oid][0]} after placing it")
        landed = self.support_of(oid)
        expected = "table" if t in SPOTS else t
        if landed != expected:
            where = OBJECTS.get(landed, BINS.get(landed, (landed,)))[0]
            raise ActionError(f"the {OBJECTS[oid][0]} did not stay put; it ended up on the {where}")

    def width(self, oid):
        return 2 * {"cube": 0.02, "cylinder": 0.02, "ball": 0.024}[OBJECTS[oid][1]]

    def _finger_yaw(self, p, oid, cube_yaw=None, ignore=None):
        """Wrist angle whose fingers have the most room around p: away from bin walls and the other
        objects the camera has seen. A cube must be gripped across faces (cube_yaw); round objects
        and held cubes can turn freely. Prefers the current angle when it is good enough."""
        if cube_yaw is not None:
            a = _wrap_quarter(cube_yaw)
            cands = [a, a + math.pi / 2 if a < 0 else a - math.pi / 2]
        else:
            a = (self.set_yaw + math.pi / 2) % math.pi - math.pi / 2
            cands = [a] + [(a + k * math.pi / 4 + math.pi / 2) % math.pi - math.pi / 2 for k in (1, 2, 3)]
        reach = self.width(oid) / 2 + 0.0125 + 0.012     # half the open gap, finger thickness
        others = [self.est_pos(o)[:2] for o in self.known() if o not in (oid, ignore, self.grasped())]

        def room(yaw):
            axis = np.array([math.sin(yaw), -math.cos(yaw)])   # the fingers close along the hand's y axis
            worst = 1.0
            for f in (p[:2] + axis * reach, p[:2] - axis * reach):
                for _, bx, by, _ in BINS.values():
                    d = np.abs(f - [bx, by])
                    if d.max() < BIN_HALF + 0.02:
                        worst = min(worst, BIN_HALF - 0.01 - d.max())
                for q in others:
                    worst = min(worst, np.linalg.norm(f - q) - 0.03)
            return worst
        best = max(cands, key=room)
        return cands[0] if room(cands[0]) >= min(room(best), 0.012) else best

    def _free_spot(self, x0, y0, oid):
        cands = [(0.0, 0.0)] + [(math.cos(a * math.pi / 4) * 0.08 * k, math.sin(a * math.pi / 4) * 0.08 * k)
                                for k in (1, 2) for a in range(8)]
        for dx, dy in cands:
            x, y = x0 + dx, y0 + dy
            if all(math.hypot(*(self.est_pos(o)[:2] - [x, y])) > 0.075 for o in self.known() if o != oid):
                return x, y
        return x0, y0

    # ---------- streaming to the browser ----------
    def visual_geoms(self):
        """Geoms to draw (skip Menagerie's collision meshes, group 3)."""
        return [g for g in range(self.m.ngeom) if self.m.geom_group[g] != 3]

    def scene_json(self):
        m = self.m
        out = []
        b64 = lambda a: base64.b64encode(np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode()
        for g in self.visual_geoms():
            rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
            item = {"type": int(m.geom_type[g]), "size": m.geom_size[g].tolist(), "rgba": rgba.tolist(),
                    "body": m.body(m.geom_bodyid[g]).name}
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                va, fa, na = m.mesh_vertadr[mid], m.mesh_faceadr[mid], m.mesh_normaladr[mid]
                faces = m.mesh_face[fa:fa + m.mesh_facenum[mid]]
                fnorm = m.mesh_facenormal[fa:fa + m.mesh_facenum[mid]]
                verts = m.mesh_vert[va:va + m.mesh_vertnum[mid]]
                norms = m.mesh_normal[na:na + m.mesh_normalnum[mid]]
                item["pos"] = b64(verts[faces.reshape(-1)])
                item["nrm"] = b64(norms[fnorm.reshape(-1)])
            out.append(item)
        return {"geoms": out, "objects": {k: v[0] for k, v in OBJECTS.items()},
                "bins": {k: v[0] for k, v in BINS.items()}}

    def frame_bytes(self, geoms):
        """Per geom: position (3) + rotation matrix (9, row-major); the hand's body frame last, for the wrist camera."""
        pos = np.vstack([self.d.geom_xpos[geoms], self.d.xpos[self.hand]])
        mat = np.vstack([self.d.geom_xmat[geoms], self.d.xmat[self.hand]])
        return np.concatenate([pos, mat], axis=1).astype(np.float32).tobytes()

    def status(self):
        return {"q": [round(float(v), 3) for v in self.d.qpos[self.qadr]],
                "grip": round(self.grip_width() * 1000, 1),
                "tcp": [round(float(v), 3) for v in self.tcp()[0]],
                "held": self.grasped(), "time": round(float(self.d.time), 2), "sense": self.sense}
