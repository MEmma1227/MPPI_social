"""
Vectorized (NumPy) drop-in replacement for MPPIControllerForPathTracking
in python_simple_mppi/scripts/mppi_pathtracking_obav.py.

Same algorithm, same parameters, same random numbers -> same result,
but all K sampled trajectories are simulated at once instead of one by one
in Python for-loops.

Usage: put this file in python_simple_mppi/scripts/ and replace
    from mppi_pathtracking_obav import MPPIControllerForPathTracking
with
    from mppi_pathtracking_obav_fast import MPPIControllerForPathTracking
"""
import numpy as np
from typing import Tuple

from mppi_pathtracking_obav import MPPIControllerForPathTracking as _Original


class MPPIControllerForPathTracking(_Original):

    SEARCH_IDX_LEN = 200  # same forward search window as the original

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Sigma_inv = np.linalg.inv(self.Sigma)  # original recomputes this K*T times

        # vehicle outline key points (same 9 points + closing point as the original)
        vw = self.vehicle_w * self.collision_safety_margin_rate
        vl = self.vehicle_l * self.collision_safety_margin_rate
        sx = np.array([-0.5*vl, -0.5*vl, 0.0, +0.5*vl, +0.5*vl, +0.5*vl, 0.0, -0.5*vl, -0.5*vl])
        sy = np.array([0.0, +0.5*vw, +0.5*vw, +0.5*vw, 0.0, -0.5*vw, -0.5*vw, -0.5*vw, 0.0])
        self._shape_x, self._shape_y = sx, sy

    # ---------- batched helpers: every function takes K states at once ----------

    def _F_batch(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """x: (K,4), v: (K,2) -> (K,4) kinematic bicycle model"""
        px, py, yaw, vel = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        steer, accel = v[:, 0], v[:, 1]
        dt, l = self.delta_t, self.wheel_base
        return np.stack([px + vel*np.cos(yaw)*dt,
                         py + vel*np.sin(yaw)*dt,
                         yaw + vel/l*np.tan(steer)*dt,
                         vel + accel*dt], axis=1)

    def _g_batch(self, v: np.ndarray) -> np.ndarray:
        """clamp inputs, v: (..., 2)"""
        v[..., 0] = np.clip(v[..., 0], -self.max_steer_abs, self.max_steer_abs)
        v[..., 1] = np.clip(v[..., 1], -self.max_accel_abs, self.max_accel_abs)
        return v

    def _nearest_ref_batch(self, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        """nearest waypoint (searching forward from prev idx) for K points -> (K,4)"""
        i0 = self.prev_waypoints_idx
        seg = self.ref_path[i0:i0 + self.SEARCH_IDX_LEN]          # (N,4)
        d = (px[:, None] - seg[None, :, 0])**2 + (py[:, None] - seg[None, :, 1])**2  # (K,N)
        return seg[np.argmin(d, axis=1)]                           # (K,4)

    def _is_collided_batch(self, x: np.ndarray) -> np.ndarray:
        """x: (K,4) -> (K,) 1.0 if any outline point is inside any obstacle"""
        px, py, yaw = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        c, s = np.cos(yaw), np.sin(yaw)
        qx = self._shape_x*c - self._shape_y*s + px                # (K,9)
        qy = self._shape_x*s + self._shape_y*c + py
        hit = np.zeros(x.shape[0], dtype=bool)
        for ox, oy, r in self.obstacle_circles:                    # loop only over obstacles (few)
            hit |= np.any((qx - ox)**2 + (qy - oy)**2 < r**2, axis=1)
        return hit.astype(float)

    def _state_cost_batch(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """stage / terminal cost for K states -> (K,)"""
        ref = self._nearest_ref_batch(x[:, 0], x[:, 1])
        yaw = (x[:, 2] + 2.0*np.pi) % (2.0*np.pi)
        yaw_diff = np.arctan2(np.sin(yaw - ref[:, 2]), np.cos(yaw - ref[:, 2]))
        cost = (weight[0]*(x[:, 0] - ref[:, 0])**2 + weight[1]*(x[:, 1] - ref[:, 1])**2 +
                weight[2]*yaw_diff**2 + weight[3]*(x[:, 3] - ref[:, 3])**2)
        return cost + self._is_collided_batch(x) * 1.0e10

    def _rollout(self, x0: np.ndarray, v: np.ndarray) -> np.ndarray:
        """simulate all trajectories, x0: (4,), v: (K,T,2) -> (K,T,4)"""
        K, T = v.shape[0], v.shape[1]
        traj = np.zeros((K, T, self.dim_x))
        x = np.tile(x0, (K, 1))
        for t in range(T):                                         # loop only over time (T=20)
            x = self._F_batch(x, v[:, t])
            traj[:, t] = x
        return traj

    # ---------- main function: same steps as the original, no K loop ----------

    def calc_control_input(self, observed_x: np.ndarray) -> Tuple[float, np.ndarray]:
        u = self.u_prev
        x0 = observed_x

        self._get_nearest_waypoint(x0[0], x0[1], update_prev_idx=True)
        if self.prev_waypoints_idx >= self.ref_path.shape[0]-1:
            print("[ERROR] Reached the end of the reference path.")
            raise IndexError

        # 1) sample noise (identical call to the original -> identical random numbers)
        epsilon = self._calc_epsilon(self.Sigma, self.K, self.T, self.dim_u)   # (K,T,2)

        # 2) build sampled control sequences: exploitation + exploration samples
        v = u[None, :, :] + epsilon
        n_exploit = int(np.ceil((1.0 - self.param_exploration) * self.K))
        v[n_exploit:] = epsilon[n_exploit:]
        v = self._g_batch(v)

        # 3) roll out all K trajectories at once
        x0 = np.asarray(x0, dtype=float)
        traj = self._rollout(x0, v)                                            # (K,T,4)

        # 4) costs: stage costs for t=1..T, control cost term, terminal cost
        K, T = self.K, self.T
        flat = traj.reshape(K*T, self.dim_x)
        stage = self._state_cost_batch(flat, self.stage_cost_weight).reshape(K, T)
        ctrl = self.param_gamma * np.einsum('ti,ij,ktj->kt', u, self.Sigma_inv, v)
        S = stage.sum(axis=1) + ctrl.sum(axis=1)
        S += self._state_cost_batch(traj[:, -1], self.terminal_cost_weight)

        # 5) weights and update (same as the original)
        w = self._compute_weights(S)
        w_epsilon = np.einsum('k,kti->ti', w, epsilon)
        w_epsilon = self._moving_average_filter(xx=w_epsilon, window_size=10)
        u += w_epsilon

        # optimal trajectory for plotting
        optimal_traj = np.zeros((self.T, self.dim_x))
        if self.visualize_optimal_traj:
            # note: the original clips u in place here (its _g modifies its argument), keep that behavior
            optimal_traj = self._rollout(x0, self._g_batch(u)[None])[0]

        # sampled trajectories for plotting (reuse the rollouts, no re-simulation)
        sampled_traj_list = np.zeros((self.K, self.T, self.dim_x))
        if self.visualze_sampled_trajs:
            sampled_traj_list = traj

        self.u_prev[:-1] = u[1:]
        self.u_prev[-1] = u[-1]
        return u[0], u, optimal_traj, sampled_traj_list

    def _compute_weights(self, S: np.ndarray) -> np.ndarray:
        z = np.exp((-1.0/self.param_lambda) * (S - S.min()))
        return z / z.sum()
