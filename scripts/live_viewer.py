"""
Fast live viewer for the MPPI path-tracking demo.

Same layout as pathtracking_kbm_obav.Vehicle (main view centered on the vehicle, minimap,
steering / acceleration gauges), but every artist is created ONCE and then only its data is
updated each step. The original Vehicle creates ~35 new artists (incl. two pie charts) every
step and keeps all of them in the figure, which makes live drawing slow and uneven.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


def _rot(xy: np.ndarray, angle: float, offset=(0.0, 0.0)) -> np.ndarray:
    """rotate (N,2) points by angle and translate"""
    c, s = np.cos(angle), np.sin(angle)
    return xy @ np.array([[c, s], [-s, c]]) + np.asarray(offset)


def _rect(length: float, width: float) -> np.ndarray:
    l, w = 0.5 * length, 0.5 * width
    return np.array([[-l, -w], [l, -w], [l, w], [-l, w]])


class LiveViewer:
    VIEW_X, VIEW_Y = 20.0, 25.0                 # main view half size [m] (same as the original)
    MINIMAP_X, MINIMAP_Y = (-40.0, 40.0), (-10.0, 40.0)

    def __init__(self, ref_path_xy: np.ndarray, obstacle_circles: np.ndarray,
                 vehicle_width: float = 3.0, vehicle_length: float = 4.0,
                 max_steer_abs: float = 0.523, max_accel_abs: float = 2.0,
                 show_sampled_trajs: bool = False, offscreen: bool = False):
        """offscreen=True: draw into an in-memory figure (no window), used for writing the mp4"""
        self.ref = np.asarray(ref_path_xy)[:, 0:2]
        self.obs = np.asarray(obstacle_circles)
        self.vw, self.vl = vehicle_width, vehicle_length
        self.max_steer, self.max_accel = max_steer_abs, max_accel_abs
        self.show_samples = show_sampled_trajs

        self.offscreen = offscreen
        if offscreen:
            self.fig = Figure(figsize=(9, 9), dpi=100)   # fixed 900x900 px, not affected by screen scaling
            FigureCanvasAgg(self.fig)
        else:
            self.fig = plt.figure(figsize=(9, 9))
        # always draw into THIS figure (never into whatever plt considers the "current" figure)
        ax = self.main_ax = plt.subplot2grid((3, 4), (0, 0), rowspan=3, colspan=3, fig=self.fig)
        mm = self.minimap_ax = plt.subplot2grid((3, 4), (0, 3), fig=self.fig)
        sa = self.steer_ax = plt.subplot2grid((3, 4), (1, 3), fig=self.fig)
        aa = self.accel_ax = plt.subplot2grid((3, 4), (2, 3), fig=self.fig)

        # ---- main view (vehicle stays at the center, the world moves) ----
        ax.set_aspect('equal')
        ax.set_xlim(-self.VIEW_X, self.VIEW_X)
        ax.set_ylim(-self.VIEW_Y, self.VIEW_Y)
        ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        self.ref_line, = ax.plot([], [], color='black', linestyle='dashed', linewidth=1.5)
        self.samples = LineCollection([], colors='gray', linewidths=0.2, alpha=0.3, zorder=4)
        ax.add_collection(self.samples)
        self.opt_line, = ax.plot([], [], color='#990099', linewidth=1.5, zorder=5)
        self.obs_main = [ax.add_patch(patches.Circle((0, 0), r, fc='white', ec='black', lw=2.0, zorder=0))
                         for _, _, r in self.obs]
        self.body = ax.add_patch(patches.Polygon(_rect(self.vl, self.vw), closed=True, fill=False,
                                                 ec='black', lw=2.0, zorder=3))
        ww, wl = 0.4, 0.7
        self.wheel_shape = _rect(wl, ww)
        self.wheel_pos = [(-0.3 * self.vl, 0.3 * self.vw), (-0.3 * self.vl, -0.3 * self.vw),   # rear L, rear R
                          (0.3 * self.vl, 0.3 * self.vw), (0.3 * self.vl, -0.3 * self.vw)]     # front L, front R
        self.wheels = [ax.add_patch(patches.Polygon(self.wheel_shape, closed=True, color='black', zorder=3))
                       for _ in range(4)]
        ax.add_patch(patches.Circle((0, 0), radius=self.vw / 20.0, fc='white', ec='black', lw=2.0, zorder=6))
        self.info = ax.text(0.5, 0.02, "", ha='center', va='bottom', transform=ax.transAxes, fontsize=14, fontfamily='monospace')

        # ---- minimap ----
        mm.set_aspect('equal'); mm.axis('off')
        mm.set_xlim(*self.MINIMAP_X); mm.set_ylim(*self.MINIMAP_Y)
        mm.plot(self.ref[:, 0], self.ref[:, 1], color='black', linestyle='dashed')
        for ox, oy, r in self.obs:
            mm.add_patch(patches.Circle((ox, oy), r, fc='white', ec='black', lw=2.0, zorder=0))
        self.mm_car = mm.add_patch(patches.Polygon(_rect(self.vl, self.vw), closed=True, fc='white',
                                                   ec='black', lw=2.0, zorder=3))

        # ---- gauges (3/4 ring, 0 at the top, same convention as the original pie charts) ----
        self.steer_wedge, self.steer_text = self._make_gauge(sa, "Steering Angle")
        self.accel_wedge, self.accel_text = self._make_gauge(aa, "Acceleration")

        self.fig.tight_layout()

    @staticmethod
    def _make_gauge(gax, title):
        gax.set_title(title, fontsize=12)
        gax.set_aspect('equal'); gax.axis('off')
        gax.set_xlim(-1.2, 1.2); gax.set_ylim(-1.4, 1.2)
        gax.add_patch(patches.Wedge((0, 0), 1.0, -45, 225, width=0.4, color='lightgray'))
        wedge = gax.add_patch(patches.Wedge((0, 0), 1.0, 90, 90, width=0.4, color='black'))
        text = gax.text(0, -1.0, "", size=14, ha='center', va='center', fontfamily='monospace')
        return wedge, text

    @staticmethod
    def _set_gauge(wedge, frac):
        """frac in [-1, 1]; positive = wedge grows counter-clockwise (to the left of the top)"""
        a = 90.0 + 135.0 * float(np.clip(frac, -1.0, 1.0))
        wedge.set_theta1(min(90.0, a)); wedge.set_theta2(max(90.0, a))

    def update(self, state, steer, accel, optimal_traj=None, sampled_trajs=None, t=None, step=None):
        """move every artist to the new state (no artist is created here)"""
        x, y, yaw, v = state
        steer = float(np.clip(steer, -self.max_steer, self.max_steer))
        accel = float(np.clip(accel, -self.max_accel, self.max_accel))
        off = np.array([x, y])

        self.ref_line.set_data(self.ref[:, 0] - x, self.ref[:, 1] - y)
        if optimal_traj is not None and len(optimal_traj):
            self.opt_line.set_data(optimal_traj[:, 0] - x, optimal_traj[:, 1] - y)
        if self.show_samples and sampled_trajs is not None and len(sampled_trajs):
            self.samples.set_segments(np.asarray(sampled_trajs)[:, :, 0:2] - off)
        for circ, (ox, oy, _) in zip(self.obs_main, self.obs):
            circ.set_center((ox - x, oy - y))

        self.body.set_xy(_rot(_rect(self.vl, self.vw), yaw))
        for k, (wheel, pos) in enumerate(zip(self.wheels, self.wheel_pos)):
            local = _rot(self.wheel_shape, steer if k >= 2 else 0.0, pos)   # front wheels turn with steer
            wheel.set_xy(_rot(local, yaw))
        # distance from the vehicle center to the nearest obstacle edge (negative = center inside the obstacle)
        d_obs = np.min(np.hypot(self.obs[:, 0] - x, self.obs[:, 1] - y) - self.obs[:, 2]) if len(self.obs) else np.inf
        lines = []
        if t is not None and step is not None:
            lines.append(f"time = {t:>6.2f} [s]   step = {step:>4d}")
        lines.append(f"vehicle velocity = {v:>+6.1f} [m/s]")
        # lines.append(f"nearest obstacle = {d_obs:>6.2f} [m]")
        self.info.set_text("\n".join(lines))

        self.mm_car.set_xy(_rot(_rect(self.vl, self.vw), yaw, off))
        self._set_gauge(self.steer_wedge, steer / self.max_steer)        # left turn -> left side
        self.steer_text.set_text(f"{np.rad2deg(steer):+.2f} [deg]")
        self._set_gauge(self.accel_wedge, -accel / self.max_accel)       # acceleration -> right side
        self.accel_text.set_text(f"{accel:+.2f} [m/s^2]")

    def refresh(self):
        """redraw the window and process GUI events (does not steal focus like plt.pause)"""
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def is_open(self) -> bool:
        return (not self.offscreen) and plt.fignum_exists(self.fig.number)

    def save_mp4(self, records: list, filename: str, interval_ms: int):
        """
        render recorded steps to a video; records = list of dicts with update() arguments.
        Use this on an offscreen viewer: a window's pixel size depends on Windows display scaling
        and on resizing, and H.264 fails when the width/height is odd (e.g. 1123x1123).
        """
        anim = FuncAnimation(self.fig, lambda k: self.update(**records[k]), frames=len(records),
                             interval=interval_ms, blit=False)
        # safety net: pad to an even width/height in case the figure size is odd anyway
        writer = FFMpegWriter(fps=1000.0 / interval_ms,
                              extra_args=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"])
        anim.save(filename, writer=writer, dpi=self.fig.dpi)