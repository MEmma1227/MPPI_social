"""
MPPI path tracking + static obstacles + PySocialForce pedestrians as dynamic obstacles.

Pedestrians are simulated with PySocialForce. Each control step the controller
  1) reads the pedestrian states from PySocialForce,
  2) predicts their positions over the MPPI horizon (constant velocity or a social-force rollout),
  3) penalizes sampled trajectories that hit (or come close to) the predicted pedestrians,
and the vehicle is written back into PySocialForce as an extra agent, so pedestrians react to it.

Put this file in python_simple_mppi/scripts/ (next to pathtracking_kbm_obav.py), then from the repo root:
    pip install pysocialforce
    python scripts/run_mppi_social.py

Only needs pathtracking_kbm_obav.py (the original repo's vehicle model); the MPPI controller
(vectorized, with dynamic-obstacle cost) and the viewer are included in this file.
"""
import copy
import logging
import shutil
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation

import pysocialforce as psf
from pathtracking_kbm_obav import Vehicle

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# ---- Visualization options ----
LIVE_VIEW = True             # show the simulation in a window while it runs
LIVE_EVERY = 2               # redraw every N steps
REALTIME = True              # play at simulation time
PLAYBACK_SPEED = 1.0
PRINT_EVERY = 10
KEEP_WINDOW_OPEN = True
SHOW_SAMPLED_TRAJS = False   # draw all MPPI samples (slow)
SHOW_PED_PREDICTION = True   # draw the predicted pedestrian paths used by MPPI
VIEW_HALF_WIDTH = 16.0       # [m] the view follows the vehicle; 0 = show the whole map
SAVE_VIDEO = False            # .mp4 if ffmpeg is installed, otherwise .gif
OUTPUT_FILE = "mppi_social_demo.mp4"
SEED = 0
# ---- Visualization options END ----

# ---- Vehicle / MPPI parameters (same as run_mppi_obav_fast_view.py) ----
DELTA_T = 0.05               # [s] simulation step (vehicle AND pedestrians)
SIM_STEPS = 683
WHEEL_BASE = 2.5
VEHICLE_WIDTH = 3.0
VEHICLE_LENGTH = 4.0
MAX_STEER_ABS = 0.523
MAX_ACCEL_ABS = 2.000

MPPI_DELTA_T = DELTA_T * 2.0  # [s] MPPI internal step -> horizon = 20 x 0.1 = 2 s
HORIZON_STEP_T = 25
NUMBER_OF_SAMPLES_K = 500     # a few more samples help with moving obstacles (vectorized, still fast)
PARAM_EXPLORATION = 0.05
PARAM_LAMBDA = 100.0
PARAM_ALPHA = 0.98
SIGMA = np.array([[0.075, 0.0], [0.0, 2.0]])
STAGE_COST_WEIGHT = np.array([50.0, 50.0, 1.0, 20.0])
TERMINAL_COST_WEIGHT = np.array([50.0, 50.0, 1.0, 20.0])
COLLISION_SAFETY_MARGIN_RATE = 1.2
# ---- Pedestrian (dynamic obstacle) parameters ----
PED_RADIUS = 0.35             # [m] same as PySocialForce agent_radius
PED_SAFETY_MARGIN = 0.5       # [m] extra clearance that counts as a collision for MPPI
PED_COLLISION_COST = 1.0e10   # hard penalty (same scale as static obstacles)
PED_SOFT_WEIGHT = 300.0       # soft "keep your distance" cost, 0 = off
PED_SOFT_SIGMA = 1.0          # [m] width of the soft cost
PED_PREDICTION = "cv"         # "cv": constant velocity, "sf": social-force rollout (slower, smarter)
ROBOT_IN_SOCIAL_FORCE = True  # True: pedestrians see the vehicle and avoid it
# ---- Parameters END ----

STATIC_OBSTACLES = np.array([
    [+8.0, +5.0, 5.0],   # x, y, radius [m]
    [+18.0, -5.0, 4.0],
])

# pedestrians: (px, py, vx, vy, gx, gy). The initial speed is also each pedestrian's preferred
# speed in PySocialForce, so it must be non-zero or that pedestrian will never move.
PEDESTRIANS = np.array([
    [13.0, -4.5, 0.0, 1.2, 13.0, 8.0],    # crosses the first straight in front of the car
    [27.0, 15.0, -1.0, 0.0, 34.0, 15.0],  # crosses the right curve (walks outward)
    [26.0, 16.0, -1.0, 0.0, 34.0, 16.0],  # its companion (same group)
    [4.0, 36.0, 0.0, -1.1, 4.0, 24.0],    # crosses the top straight
    [-10.0, 33.3, 1.0, 0.0, 22.0, 33.3],  # walks on the roadside next to the top straight
])
PED_BACK_AND_FORTH = True     # True: a pedestrian that reaches its goal turns around (keeps the scene busy)
PED_GROUPS = [[0], [1, 2], [3], [4]]


# =====================================================================================
#  Vectorized MPPI with static circles + time-varying pedestrian obstacles
# =====================================================================================
class SocialMPPI:
    def __init__(self, delta_t, wheel_base, vehicle_width, vehicle_length, max_steer_abs,
                 max_accel_abs, ref_path, horizon_step_T, number_of_samples_K, param_exploration,
                 param_lambda, param_alpha, sigma, stage_cost_weight, terminal_cost_weight,
                 obstacle_circles, collision_safety_margin_rate, ped_radius, ped_safety_margin,
                 ped_collision_cost, ped_soft_weight, ped_soft_sigma, seed=None):
        self.dim_x, self.dim_u = 4, 2
        self.T, self.K = horizon_step_T, number_of_samples_K
        self.param_exploration = param_exploration
        self.param_lambda = param_lambda
        self.param_gamma = param_lambda * (1.0 - param_alpha)
        self.Sigma = sigma
        self.Sigma_inv = np.linalg.inv(sigma)
        self.stage_w = stage_cost_weight
        self.term_w = terminal_cost_weight
        self.dt = delta_t
        self.L = wheel_base
        self.max_steer, self.max_accel = max_steer_abs, max_accel_abs
        self.ref_path = ref_path
        self.obstacle_circles = np.asarray(obstacle_circles, dtype=float).reshape(-1, 3)
        # half length / half width of the vehicle box used for collision checks
        self.hl = 0.5 * vehicle_length * collision_safety_margin_rate
        self.hw = 0.5 * vehicle_width * collision_safety_margin_rate
        self.ped_hit_dist = ped_radius + ped_safety_margin
        self.ped_collision_cost = ped_collision_cost
        self.ped_soft_weight, self.ped_soft_sigma = ped_soft_weight, ped_soft_sigma
        self.rng = np.random.default_rng(seed)

        self.u_prev = np.zeros((self.T, self.dim_u))
        self.prev_waypoints_idx = 0
        self.ped_pred = np.zeros((self.T + 1, 0, 2))  # (T+1, N, 2), index t = t*dt seconds ahead

    # ---------- pedestrians ----------
    def set_pedestrian_prediction(self, ped_pred: np.ndarray) -> None:
        """ped_pred: (T+1, N, 2) predicted pedestrian positions at t = 0, dt, ..., T*dt"""
        assert ped_pred.shape[0] == self.T + 1
        self.ped_pred = ped_pred

    # ---------- main ----------
    def calc_control_input(self, observed_x: np.ndarray):
        u = self.u_prev
        x0 = np.asarray(observed_x, dtype=float)

        self._update_nearest_waypoint(x0[0], x0[1])
        if self.prev_waypoints_idx >= self.ref_path.shape[0] - 1:
            print("[ERROR] Reached the end of the reference path.")
            raise IndexError

        # sample control sequences
        epsilon = self.rng.multivariate_normal(np.zeros(self.dim_u), self.Sigma, (self.K, self.T))
        n_exploit = int(np.ceil((1.0 - self.param_exploration) * self.K))
        v = epsilon.copy()
        v[:n_exploit] += u[None]                       # exploitation around the previous solution
        v[:, :, 0] = np.clip(v[:, :, 0], -self.max_steer, self.max_steer)
        v[:, :, 1] = np.clip(v[:, :, 1], -self.max_accel, self.max_accel)

        # roll out all K trajectories at once
        trajs = self._rollout(x0, v)                   # (K, T, 4), trajs[:, t-1] = state at time t

        # costs
        S = self._state_cost(trajs, terminal=False).sum(axis=1)
        S += self._state_cost(trajs[:, -1:], terminal=True)[:, 0]
        S += self._obstacle_cost(trajs).sum(axis=1)
        S += self._obstacle_cost(trajs[:, -1:], t_offset=self.T - 1)[:, 0]
        S += self.param_gamma * np.einsum("ti,ij,ktj->k", u, self.Sigma_inv, v)

        # MPPI update
        w = self._compute_weights(S)
        w_epsilon = np.einsum("k,kti->ti", w, epsilon)
        w_epsilon = self._moving_average_filter(w_epsilon, window_size=10)
        u = u + w_epsilon

        optimal_traj = self._rollout(x0, self._clip(u)[None])[0]
        sampled_traj_list = trajs[np.argsort(S)]

        self.u_prev[:-1] = u[1:]
        self.u_prev[-1] = u[-1]
        return self._clip(u[0:1])[0], u, optimal_traj, sampled_traj_list

    # ---------- helpers ----------
    def _clip(self, u):
        u = u.copy()
        u[..., 0] = np.clip(u[..., 0], -self.max_steer, self.max_steer)
        u[..., 1] = np.clip(u[..., 1], -self.max_accel, self.max_accel)
        return u

    def _rollout(self, x0, v):
        K, T = v.shape[0], v.shape[1]
        trajs = np.zeros((K, T, self.dim_x))
        x = np.tile(x0, (K, 1))
        for t in range(T):
            steer, accel = v[:, t, 0], v[:, t, 1]
            nx = x[:, 0] + x[:, 3] * np.cos(x[:, 2]) * self.dt
            ny = x[:, 1] + x[:, 3] * np.sin(x[:, 2]) * self.dt
            nyaw = x[:, 2] + x[:, 3] / self.L * np.tan(steer) * self.dt
            nv = x[:, 3] + accel * self.dt
            x = np.stack([nx, ny, nyaw, nv], axis=1)
            trajs[:, t] = x
        return trajs

    def _update_nearest_waypoint(self, x, y):
        idx, _ = self._nearest_waypoints(np.array([x]), np.array([y]))
        self.prev_waypoints_idx = int(idx[0])

    def _nearest_waypoints(self, xs, ys, search_len=200):
        prev = self.prev_waypoints_idx
        seg = self.ref_path[prev:prev + search_len]
        d = (xs[..., None] - seg[:, 0]) ** 2 + (ys[..., None] - seg[:, 1]) ** 2
        idx = np.argmin(d, axis=-1)
        return idx + prev, seg[idx]

    def _state_cost(self, trajs, terminal):
        x, y, yaw, vel = trajs[..., 0], trajs[..., 1], trajs[..., 2], trajs[..., 3]
        _, ref = self._nearest_waypoints(x, y)
        yaw = (yaw + 2.0 * np.pi) % (2.0 * np.pi)
        yaw_diff = np.arctan2(np.sin(yaw - ref[..., 2]), np.cos(yaw - ref[..., 2]))
        w = self.term_w if terminal else self.stage_w
        return (w[0] * (x - ref[..., 0]) ** 2 + w[1] * (y - ref[..., 1]) ** 2 +
                w[2] * yaw_diff ** 2 + w[3] * (vel - ref[..., 3]) ** 2)

    def _box_distance(self, trajs, cx, cy):
        """signed distance from points (cx, cy) to the (safety-scaled) vehicle box.
        trajs: (K, T, 4); cx, cy broadcastable to (K, T, M). Negative = inside the box."""
        dx = cx - trajs[..., 0:1]
        dy = cy - trajs[..., 1:2]
        c, s = np.cos(trajs[..., 2:3]), np.sin(trajs[..., 2:3])
        lx = np.abs(c * dx + s * dy) - self.hl
        ly = np.abs(-s * dx + c * dy) - self.hw
        outside = np.hypot(np.maximum(lx, 0.0), np.maximum(ly, 0.0))
        inside = np.minimum(np.maximum(lx, ly), 0.0)
        return outside + inside

    def _obstacle_cost(self, trajs, t_offset=0):
        """(K, T') cost of static circles + predicted pedestrians at matching time steps"""
        K, Tn = trajs.shape[0], trajs.shape[1]
        cost = np.zeros((K, Tn))

        # static circular obstacles: same 9 key-point check as the original repo
        if len(self.obstacle_circles):
            hl, hw = self.hl, self.hw
            kx = np.array([-hl, -hl, 0.0, hl, hl, hl, 0.0, -hl, -hl])
            ky = np.array([0.0, hw, hw, hw, 0.0, -hw, -hw, -hw, 0.0])
            c, s = np.cos(trajs[..., 2:3]), np.sin(trajs[..., 2:3])
            px = trajs[..., 0:1] + c * kx - s * ky                       # (K, T', 9)
            py = trajs[..., 1:2] + s * kx + c * ky
            ox, oy, r = (self.obstacle_circles[:, i] for i in range(3))
            d2 = (px[..., None] - ox) ** 2 + (py[..., None] - oy) ** 2   # (K, T', 9, M)
            cost += np.any(d2 < r ** 2, axis=(-1, -2)) * 1.0e10

        # pedestrians are small, so they could slip between the key points:
        # use the exact distance to the vehicle box instead

        # pedestrians: trajs[:, j] is the state at time t = t_offset + j + 1
        if self.ped_pred.shape[1] > 0:
            peds = self.ped_pred[t_offset + 1:t_offset + 1 + Tn]       # (T', N, 2)
            d = self._box_distance(trajs, peds[None, :, :, 0], peds[None, :, :, 1])
            cost += np.any(d < self.ped_hit_dist, axis=-1) * self.ped_collision_cost
            if self.ped_soft_weight > 0.0:
                dd = np.maximum(d, 0.0)
                cost += self.ped_soft_weight * np.exp(-dd ** 2 / (2 * self.ped_soft_sigma ** 2)).sum(-1)
        return cost

    def _compute_weights(self, S):
        rho = S.min()
        e = np.exp(-(S - rho) / self.param_lambda)
        return e / e.sum()

    @staticmethod
    def _moving_average_filter(xx, window_size):
        b = np.ones(window_size) / window_size
        n = xx.shape[0]
        out = np.zeros_like(xx)
        for d in range(xx.shape[1]):
            out[:, d] = np.convolve(xx[:, d], b, mode="same")
            n_conv = int(np.ceil(window_size / 2))
            out[0, d] *= window_size / n_conv
            for i in range(1, n_conv):
                out[i, d] *= window_size / (i + n_conv)
                out[-i, d] *= window_size / (i + n_conv - (window_size % 2))
        return out


# =====================================================================================
#  PySocialForce helpers
# =====================================================================================
def circles_to_segments(circles, n_seg=16):
    """PySocialForce only knows line obstacles (x1, x2, y1, y2): approximate each circle by a polygon"""
    segs = []
    for cx, cy, r in circles:
        a = np.linspace(0, 2 * np.pi, n_seg + 1)
        px, py = cx + r * np.cos(a), cy + r * np.sin(a)
        for i in range(n_seg):
            segs.append([px[i], px[i + 1], py[i], py[i + 1]])
    return segs


def vehicle_to_ped_row(state, goal):
    x, y, yaw, v = state
    return np.array([x, y, v * np.cos(yaw), v * np.sin(yaw), goal[0], goal[1]])


def predict_pedestrians(sim, n_peds, robot_state, horizon, dt, mode="cv"):
    """(horizon+1, n_peds, 2) predicted pedestrian positions"""
    st = sim.peds.state
    pos, vel = st[:n_peds, 0:2], st[:n_peds, 2:4]
    if mode == "cv":
        ts = np.arange(horizon + 1)[:, None, None] * dt
        return pos[None] + ts * vel[None]

    # social-force rollout on a copy of the simulator; the vehicle moves at constant velocity
    sim_c = copy.deepcopy(sim)
    sim_c.peds.step_width = dt
    pred = [pos.copy()]
    has_robot = sim_c.peds.size() > n_peds
    rx, ry, ryaw, rv = robot_state
    for t in range(1, horizon + 1):
        sim_c.step_once()
        if has_robot:
            sim_c.peds.state[n_peds, 0:4] = [rx + rv * np.cos(ryaw) * t * dt,
                                             ry + rv * np.sin(ryaw) * t * dt,
                                             rv * np.cos(ryaw), rv * np.sin(ryaw)]
        pred.append(sim_c.peds.state[:n_peds, 0:2].copy())
    return np.stack(pred)


# =====================================================================================
#  Viewer
# =====================================================================================
class SocialViewer:
    def __init__(self, ref_path, obstacles, n_peds, vehicle_width, vehicle_length,
                 show_sampled_trajs=False, show_ped_prediction=True, half_width=16.0, offscreen=False):
        if offscreen:
            from matplotlib.figure import Figure
            self.fig = Figure(figsize=(9, 9), dpi=100)
        else:
            self.fig = plt.figure(figsize=(9, 9))
        self.ax = self.fig.add_subplot(111)
        self.vw, self.vl = vehicle_width, vehicle_length
        self.half_width = half_width
        self.show_sampled = show_sampled_trajs
        self.show_pred = show_ped_prediction
        ax = self.ax
        ax.set_aspect("equal")
        ax.plot(ref_path[:, 0], ref_path[:, 1], "--", color="0.6", lw=1, label="reference path")
        for ox, oy, r in obstacles:
            ax.add_patch(patches.Circle((ox, oy), r, fc="0.75", ec="0.3"))
        self.body = patches.Polygon(np.zeros((4, 2)), closed=True, fc="tab:blue", ec="k", alpha=0.85)
        ax.add_patch(self.body)
        self.ped_patches = [patches.Circle((0, 0), PED_RADIUS, fc="tab:orange", ec="k", zorder=5)
                            for _ in range(n_peds)]
        for p in self.ped_patches:
            ax.add_patch(p)
        self.ped_trails = [ax.plot([], [], "-", color="tab:orange", lw=0.8, alpha=0.5)[0]
                           for _ in range(n_peds)]
        self.ped_preds = [ax.plot([], [], ":", color="tab:red", lw=1.2)[0] for _ in range(n_peds)]
        self.trail_xy = [[] for _ in range(n_peds)]
        self.opt_line, = ax.plot([], [], "-", color="tab:purple", lw=2, label="MPPI optimal trajectory")
        self.sample_lines = []
        self.title = ax.set_title("")
        ax.plot([], [], ":", color="tab:red", label="predicted pedestrian path")
        ax.legend(loc="upper right", fontsize=8)
        if half_width <= 0:
            pad = 6
            ax.set_xlim(ref_path[:, 0].min() - pad, ref_path[:, 0].max() + pad)
            ax.set_ylim(ref_path[:, 1].min() - pad, ref_path[:, 1].max() + pad)

    def update(self, state, steer, accel, t, step, peds, optimal_traj=None,
               sampled_trajs=None, ped_pred=None, min_clearance=np.inf, n_collisions=0):
        x, y, yaw, v = state
        c, s = np.cos(yaw), np.sin(yaw)
        hl, hw = 0.5 * self.vl, 0.5 * self.vw
        corners = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
        self.body.set_xy(corners @ np.array([[c, s], [-s, c]]) + [x, y])
        for i, p in enumerate(peds):
            self.ped_patches[i].center = (p[0], p[1])
            self.trail_xy[i].append((p[0], p[1]))
            tr = np.array(self.trail_xy[i][-200:])
            self.ped_trails[i].set_data(tr[:, 0], tr[:, 1])
            if self.show_pred and ped_pred is not None:
                self.ped_preds[i].set_data(ped_pred[:, i, 0], ped_pred[:, i, 1])
        if optimal_traj is not None:
            self.opt_line.set_data(optimal_traj[:, 0], optimal_traj[:, 1])
        if self.show_sampled and sampled_trajs is not None:
            for ln in self.sample_lines:
                ln.remove()
            self.sample_lines = [self.ax.plot(tr[:, 0], tr[:, 1], "-", color="tab:green",
                                              lw=0.4, alpha=0.3)[0] for tr in sampled_trajs[::5]]
        if self.half_width > 0:
            self.ax.set_xlim(x - self.half_width, x + self.half_width)
            self.ax.set_ylim(y - self.half_width, y + self.half_width)
        self.title.set_text(f"t = {t:5.2f} s   v = {v:4.2f} m/s   steer = {steer:+.2f}   accel = {accel:+.2f}\n"
                            f"min clearance to pedestrians = {min_clearance:4.2f} m   collisions = {n_collisions}")

    def refresh(self):
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def is_open(self):
        return plt.fignum_exists(self.fig.number)

    def save_video(self, records, filename, interval_ms):
        self.trail_xy = [[] for _ in self.trail_xy]

        def draw(i):
            self.update(**records[i])
            return []

        if shutil.which("ffmpeg") and filename.endswith(".mp4"):
            writer = animation.FFMpegWriter(fps=1000 / interval_ms)
        else:
            filename = filename.rsplit(".", 1)[0] + ".gif"
            writer = animation.PillowWriter(fps=min(20, 1000 / interval_ms))
        ani = animation.FuncAnimation(self.fig, draw, frames=len(records), blit=False)
        ani.save(filename, writer=writer)
        return filename


def vehicle_ped_clearance(state, peds, vehicle_width, vehicle_length):
    """distance from each pedestrian's edge to the real (unscaled) vehicle box"""
    x, y, yaw, _ = state
    dx, dy = peds[:, 0] - x, peds[:, 1] - y
    c, s = np.cos(yaw), np.sin(yaw)
    lx = np.abs(c * dx + s * dy) - 0.5 * vehicle_length
    ly = np.abs(-s * dx + c * dy) - 0.5 * vehicle_width
    d = np.hypot(np.maximum(lx, 0), np.maximum(ly, 0)) + np.minimum(np.maximum(lx, ly), 0)
    return d - PED_RADIUS


# =====================================================================================
#  Main loop
# =====================================================================================
def run_simulation(live_view=LIVE_VIEW, save_video=SAVE_VIDEO):
    print("[INFO] MPPI path tracking with PySocialForce pedestrians as dynamic obstacles")
    ref_path = np.genfromtxt("./data/ovalpath.csv", delimiter=",", skip_header=1)
    n_peds = len(PEDESTRIANS)

    vehicle = Vehicle(wheel_base=WHEEL_BASE, vehicle_width=VEHICLE_WIDTH, vehicle_length=VEHICLE_LENGTH,
                      max_steer_abs=MAX_STEER_ABS, max_accel_abs=MAX_ACCEL_ABS,
                      ref_path=ref_path[:, 0:2], obstacle_circles=STATIC_OBSTACLES,
                      delta_t=DELTA_T, visualize=False)
    vehicle.reset(init_state=np.array([0.0, 0.0, 0.0, 0.0]))
    plt.close("all")

    # --- PySocialForce: pedestrians (+ the vehicle as the last agent) ---
    init_state = PEDESTRIANS.copy()
    if ROBOT_IN_SOCIAL_FORCE:
        # the goal is irrelevant (the row is overwritten every step); a non-zero speed keeps PSF happy
        robot_row = vehicle_to_ped_row(vehicle.get_state(), goal=(1e3, 0.0))
        robot_row[2] = 1e-3
        init_state = np.vstack([init_state, robot_row])
    sim = psf.Simulator(init_state, groups=PED_GROUPS, obstacles=circles_to_segments(STATIC_OBSTACLES))
    sim.peds.step_width = DELTA_T   # PSF default is 0.4~1.0 s; match the vehicle simulation step
    ped_starts, ped_goals = PEDESTRIANS[:, 0:2].copy(), PEDESTRIANS[:, 4:6].copy()

    mppi = SocialMPPI(delta_t=MPPI_DELTA_T, wheel_base=WHEEL_BASE, vehicle_width=VEHICLE_WIDTH,
                      vehicle_length=VEHICLE_LENGTH, max_steer_abs=MAX_STEER_ABS,
                      max_accel_abs=MAX_ACCEL_ABS, ref_path=ref_path, horizon_step_T=HORIZON_STEP_T,
                      number_of_samples_K=NUMBER_OF_SAMPLES_K, param_exploration=PARAM_EXPLORATION,
                      param_lambda=PARAM_LAMBDA, param_alpha=PARAM_ALPHA, sigma=SIGMA,
                      stage_cost_weight=STAGE_COST_WEIGHT, terminal_cost_weight=TERMINAL_COST_WEIGHT,
                      obstacle_circles=STATIC_OBSTACLES,
                      collision_safety_margin_rate=COLLISION_SAFETY_MARGIN_RATE,
                      ped_radius=PED_RADIUS, ped_safety_margin=PED_SAFETY_MARGIN,
                      ped_collision_cost=PED_COLLISION_COST, ped_soft_weight=PED_SOFT_WEIGHT,
                      ped_soft_sigma=PED_SOFT_SIGMA, seed=SEED)

    def make_viewer(offscreen=False):
        return SocialViewer(ref_path, STATIC_OBSTACLES, n_peds, VEHICLE_WIDTH, VEHICLE_LENGTH,
                            show_sampled_trajs=SHOW_SAMPLED_TRAJS, show_ped_prediction=SHOW_PED_PREDICTION,
                            half_width=VIEW_HALF_WIDTH, offscreen=offscreen)

    viewer = None
    if live_view:
        plt.ion()
        viewer = make_viewer()
        viewer.fig.show()

    records, frame = [], None
    min_clear_all, n_collision_steps, t_ctrl = np.inf, 0, 0.0
    t_start = time.perf_counter()
    i = -1
    for i in range(SIM_STEPS):
        x = vehicle.get_state()

        # 1) pedestrian prediction -> MPPI
        ped_pred = predict_pedestrians(sim, n_peds, x, HORIZON_STEP_T, MPPI_DELTA_T, PED_PREDICTION)
        mppi.set_pedestrian_prediction(ped_pred)

        # 2) MPPI
        try:
            t0 = time.perf_counter()
            u, _, optimal_traj, sampled = mppi.calc_control_input(observed_x=x)
            t_ctrl += time.perf_counter() - t0
        except IndexError:
            print("[INFO] reached the end of the reference path.")
            break

        # 3) step vehicle and pedestrians, then write the vehicle back into PySocialForce
        vehicle.update(u=u, delta_t=DELTA_T, append_frame=False)
        sim.step_once()
        if ROBOT_IN_SOCIAL_FORCE:
            sim.peds.state[n_peds, 0:4] = vehicle_to_ped_row(vehicle.get_state(), (0, 0))[0:4]

        if PED_BACK_AND_FORTH:
            st = sim.peds.state
            arrived = np.linalg.norm(st[:n_peds, 0:2] - ped_goals, axis=1) < 0.6
            for j in np.where(arrived)[0]:
                ped_goals[j], ped_starts[j] = ped_starts[j].copy(), ped_goals[j].copy()
                st[j, 4:6] = ped_goals[j]

        # metrics
        x_new = vehicle.get_state()
        peds_now = sim.peds.state[:n_peds, 0:2].copy()
        clear = vehicle_ped_clearance(x_new, peds_now, VEHICLE_WIDTH, VEHICLE_LENGTH)
        min_clear = float(clear.min())
        min_clear_all = min(min_clear_all, min_clear)
        if min_clear < 0.0:
            n_collision_steps += 1

        if i % PRINT_EVERY == 0:
            print(f"t={i*DELTA_T:5.2f}s x={x[0]:+7.2f} y={x[1]:+7.2f} yaw={x[2]:+5.2f} v={x[3]:+5.2f} "
                  f"steer={u[0]:+5.2f} accel={u[1]:+5.2f} | min ped clearance {min_clear:5.2f} m")

        frame = dict(state=x_new, steer=u[0], accel=u[1], t=(i + 1) * DELTA_T, step=i + 1,
                     peds=peds_now, optimal_traj=optimal_traj[:, 0:2].copy(),
                     sampled_trajs=sampled[:, :, 0:2].copy() if SHOW_SAMPLED_TRAJS else None,
                     ped_pred=ped_pred.copy() if SHOW_PED_PREDICTION else None,
                     min_clearance=min_clear, n_collisions=n_collision_steps)
        if save_video:
            records.append(frame)

        if live_view:
            if not viewer.is_open():
                print("[INFO] window closed, stopping simulation")
                break
            if i % LIVE_EVERY == 0:
                viewer.update(**frame)
                viewer.refresh()
            if REALTIME:
                wait = t_start + (i + 1) * DELTA_T / PLAYBACK_SPEED - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)

    n_done = max(i + 1, 1)
    print(f"[INFO] finished {n_done*DELTA_T:.1f} s of simulation in {time.perf_counter()-t_start:.1f} s "
          f"(MPPI {t_ctrl/n_done*1000:.1f} ms/step)")
    print(f"[RESULT] minimum clearance to pedestrians: {min_clear_all:.2f} m, "
          f"steps in collision: {n_collision_steps}")

    if save_video and records:
        saver = make_viewer(offscreen=True)
        out = saver.save_video(records, OUTPUT_FILE, interval_ms=int(DELTA_T * 1000))
        print(f"[INFO] saved {out} ({len(records)} frames)")

    if live_view and KEEP_WINDOW_OPEN and viewer.is_open() and frame is not None:
        viewer.update(**frame)
        plt.ioff()
        print("[INFO] close the window to exit")
        plt.show()
    return min_clear_all, n_collision_steps


if __name__ == "__main__":
    run_simulation()
