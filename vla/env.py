"""Table-top pushing environment in MuJoCo.

A 2-DoF pusher (x/y slide joints) must push the named block onto the named pad.
Observations: RGB + depth from one fixed camera, pusher position, and the full
simulator state (used by the scripted expert and the state-only baseline).
"""

import mujoco
import numpy as np

BLOCK_COLORS = {"red": (0.85, 0.15, 0.15), "green": (0.15, 0.7, 0.2), "blue": (0.15, 0.3, 0.9)}
PAD_COLORS = {"yellow": (0.95, 0.8, 0.1), "purple": (0.6, 0.2, 0.8)}
BLOCK_NAMES = list(BLOCK_COLORS)
PAD_NAMES = list(PAD_COLORS)

# Never shown in training demos; used to test compositional generalization.
HELDOUT_TASKS = {("red", "purple")}

TEMPLATES = [
    "push the {b} block to the {p} pad",
    "move the {b} block onto the {p} pad",
    "put the {b} cube on the {p} target",
    "slide the {b} cube to the {p} pad",
]

BLOCK_HALF = 0.022
PUSHER_RADIUS = 0.018
PAD_RADIUS = 0.045
SUCCESS_DIST = 0.035      # block center within this distance of pad center
WORKSPACE = 0.2           # pusher limits: [-WORKSPACE, WORKSPACE] in x and y
MAX_STEP = 0.02           # meters moved per control step at |action| = 1
CONTROL_DT = 0.1          # 10 Hz control
IMG_SIZE = 84
DEPTH_NEAR, DEPTH_FAR = 0.35, 1.0


def _rgba(c):
    return f"{c[0]} {c[1]} {c[2]} 1"


def build_xml():
    blocks = "\n".join(
        f"""<body name="block_{n}" pos="0 0 {BLOCK_HALF}">
              <freejoint name="block_{n}"/>
              <geom type="box" size="{BLOCK_HALF} {BLOCK_HALF} {BLOCK_HALF}" mass="0.1"
                    rgba="{_rgba(c)}" friction="0.6 0.005 0.0001"/>
            </body>"""
        for n, c in BLOCK_COLORS.items()
    )
    pads = "\n".join(
        f"""<body name="pad_{n}" mocap="true" pos="0 0 0.0005">
              <geom type="cylinder" size="{PAD_RADIUS} 0.0005" rgba="{_rgba(c)}"
                    contype="0" conaffinity="0"/>
            </body>"""
        for n, c in PAD_COLORS.items()
    )
    lim = WORKSPACE + 0.02
    return f"""
<mujoco model="push">
  <option timestep="0.002" integrator="implicitfast"/>
  <visual><global offwidth="640" offheight="640"/><quality shadowsize="1024"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.55 0.55 0.55" rgb2="0.5 0.5 0.5"
             width="256" height="256"/>
    <material name="table" texture="grid" texrepeat="6 6"/>
  </asset>
  <worldbody>
    <light pos="0.3 -0.3 1.2" dir="-0.2 0.2 -1" diffuse="0.7 0.7 0.7"/>
    <light pos="-0.3 0.2 1.0" dir="0.2 -0.1 -1" diffuse="0.3 0.3 0.3"/>
    <geom name="table" type="plane" size="0.4 0.4 0.01" material="table" friction="0.4 0.005 0.0001"/>
    <camera name="front" pos="0 -0.45 0.5" xyaxes="1 0 0 0 0.729 0.685" fovy="45"/>
    <camera name="top" pos="0 0 0.9" xyaxes="1 0 0 0 1 0" fovy="35"/>
    <body name="pusher" pos="0 0 0.03">
      <joint name="px" type="slide" axis="1 0 0" range="-{lim} {lim}" damping="15"/>
      <joint name="py" type="slide" axis="0 1 0" range="-{lim} {lim}" damping="15"/>
      <geom type="cylinder" size="{PUSHER_RADIUS} 0.025" mass="0.5" rgba="0.95 0.95 0.95 1"/>
    </body>
    {blocks}
    {pads}
  </worldbody>
  <actuator>
    <position name="ax" joint="px" kp="300" ctrlrange="-{lim} {lim}"/>
    <position name="ay" joint="py" kp="300" ctrlrange="-{lim} {lim}"/>
  </actuator>
</mujoco>
"""


def make_instruction(block, pad, rng):
    return TEMPLATES[rng.integers(len(TEMPLATES))].format(b=block, p=pad)


class PushEnv:
    """Push the named block onto the named pad."""

    def __init__(self, render=True, img_size=IMG_SIZE):
        self.model = mujoco.MjModel.from_xml_string(build_xml())
        self.data = mujoco.MjData(self.model)
        self.n_substeps = int(round(CONTROL_DT / self.model.opt.timestep))
        self.img_size = img_size
        self.renderer = mujoco.Renderer(self.model, img_size, img_size) if render else None
        self.block_qadr = [self.model.jnt_qposadr[self.model.joint(f"block_{n}").id] for n in BLOCK_NAMES]
        self.block_vadr = [self.model.jnt_dofadr[self.model.joint(f"block_{n}").id] for n in BLOCK_NAMES]
        self.pad_mocap = [self.model.body_mocapid[self.model.body(f"pad_{n}").id] for n in PAD_NAMES]

    # ---- task setup -------------------------------------------------------------------
    def sample_task(self, rng, allow_heldout):
        while True:
            b = BLOCK_NAMES[rng.integers(len(BLOCK_NAMES))]
            p = PAD_NAMES[rng.integers(len(PAD_NAMES))]
            if allow_heldout or (b, p) not in HELDOUT_TASKS:
                return b, p

    def reset(self, seed, task=None, allow_heldout=False):
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.block, self.pad = task if task is not None else self.sample_task(rng, allow_heldout)
        self.instruction = make_instruction(self.block, self.pad, rng)

        pads = self._sample_points(rng, 2, 0.16, min_dist=0.16, avoid=[])
        blocks = self._sample_points(rng, 3, 0.14, min_dist=0.08, avoid=[(q, 0.09) for q in pads])
        # Keep the target block far enough from its pad that the task is non-trivial.
        bi, pi = BLOCK_NAMES.index(self.block), PAD_NAMES.index(self.pad)
        if np.linalg.norm(blocks[bi] - pads[pi]) < 0.1:
            return self.reset(seed + 7919, task=(self.block, self.pad), allow_heldout=True)
        pusher = self._sample_points(rng, 1, WORKSPACE - 0.02, min_dist=0,
                                     avoid=[(q, 0.07) for q in blocks])[0]

        for k, q in enumerate(pads):
            self.data.mocap_pos[self.pad_mocap[k]] = [q[0], q[1], 0.0005]
        for k, q in enumerate(blocks):
            a = self.block_qadr[k]
            yaw = rng.uniform(-np.pi, np.pi)
            self.data.qpos[a:a + 7] = [q[0], q[1], BLOCK_HALF, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.data.qpos[0:2] = pusher
        self.data.ctrl[:] = pusher
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        return self.observe()

    @staticmethod
    def _sample_points(rng, n, half, min_dist, avoid):
        for _ in range(1000):
            pts = rng.uniform(-half, half, size=(n, 2))
            ok = all(np.linalg.norm(pts[i] - pts[j]) >= min_dist for i in range(n) for j in range(i))
            ok = ok and all(np.linalg.norm(p - q) >= d for p in pts for q, d in avoid)
            if ok:
                return pts
        raise RuntimeError("could not sample a valid layout")

    # ---- stepping ---------------------------------------------------------------------
    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        target = self.pusher_xy() + action * MAX_STEP
        self.data.ctrl[:] = np.clip(target, -WORKSPACE, WORKSPACE)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        return self.observe(), self.is_success()

    # ---- state access -----------------------------------------------------------------
    def pusher_xy(self):
        return self.data.qpos[0:2].copy()

    def blocks_xy(self):
        return np.array([self.data.qpos[a:a + 2] for a in self.block_qadr])

    def pads_xy(self):
        return np.array([self.data.mocap_pos[m][:2] for m in self.pad_mocap])

    def target_block_xy(self):
        return self.blocks_xy()[BLOCK_NAMES.index(self.block)]

    def target_pad_xy(self):
        return self.pads_xy()[PAD_NAMES.index(self.pad)]

    def is_success(self):
        return bool(np.linalg.norm(self.target_block_xy() - self.target_pad_xy()) < SUCCESS_DIST)

    def state_vector(self):
        """Pusher, 3 blocks, 2 pads: 6 objects x (x, y), ordered by fixed color index."""
        return np.concatenate([self.pusher_xy()[None], self.blocks_xy(), self.pads_xy()]).astype(np.float32)

    # ---- observations -----------------------------------------------------------------
    def render(self, camera="front"):
        self.renderer.update_scene(self.data, camera)
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera)
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        depth = np.clip((depth - DEPTH_NEAR) / (DEPTH_FAR - DEPTH_NEAR), 0.0, 1.0)
        return rgb, depth.astype(np.float32)

    def observe(self):
        obs = {"proprio": self.pusher_xy().astype(np.float32), "state": self.state_vector()}
        if self.renderer is not None:
            obs["rgb"], obs["depth"] = self.render()
        return obs
