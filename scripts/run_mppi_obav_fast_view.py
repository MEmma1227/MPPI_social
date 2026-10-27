"""
Run the path-tracking + obstacle-avoidance demo with the vectorized MPPI controller.

Put this file and mppi_pathtracking_obav_fast.py in python_simple_mppi/scripts/,
keep mppi_pathtracking_obav.py unchanged, then run from the repo root:
    python scripts/run_mppi_obav_fast.py
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from pathtracking_kbm_obav import Vehicle
from mppi_pathtracking_obav_fast import MPPIControllerForPathTracking
from live_viewer import LiveViewer

# ---- Visualization options ----
LIVE_VIEW = True             # True: show the simulation in a window while it runs
# redraw every N steps (2~5 = smoother if drawing is slow)
LIVE_EVERY = 1
# True: play at simulation time (1 s of simulation = 1 s on screen); False: as fast as possible
REALTIME = True
PLAYBACK_SPEED = 1.0         # with REALTIME: 2.0 = twice as fast, 0.5 = slow motion
PRINT_EVERY = 10             # print the state every N steps (printing every step slows Windows down)
# keep the window open after the simulation ends (close it to exit)
KEEP_WINDOW_OPEN = True
# True: draw all sampled trajectories (nice for reports, slow to render)
SHOW_SAMPLED_TRAJS = False
SAVE_MP4 = False              # False: skip writing the video (fastest)
OUTPUT_FILE = "mppi_pathtracking_obav_demo.mp4"
# ---- Visualization options END ----

# ---- Parameter options ----
DELTA_T = 0.05  # [sec] simulation time step
SIM_STEPS = 683  # [steps] number of simulation steps
WHEEL_BASE = 2.5  # [m] wheel base of the vehicle
VEHICLE_WIDTH = 3.0  # [m] width of the vehicle 
VEHICLE_LENGTH = 4.0  # [m] length of the vehicle
MAX_STEER_ABS = 0.523  # [rad] maximum steering angle 30 deg
MAX_ACCEL_ABS = 2.000  # [m/s^2] maximum acceleration

HORIZON_STEP_T = 20  # [steps] horizon length of the MPPI controller 預測幾步。20 × 0.1 = 往前看 2 秒
NUMBER_OF_SAMPLES_K = 100  # [samples] number of sampled trajectories 每次抽幾條軌跡
PARAM_EXPLORATION = 0.05  # [variance] exploration variance of the MPPI controller 5% 的樣本不以上一次的解為中心，而是以 0 為中心純粹亂抽，避免一直困在同一個解附近
PARAM_LAMBDA = 100.0  # [weight] weight of the MPPI controller 	溫度 λ，越小越只聽最好的幾條，越大越接近平均所有樣本
PARAM_ALPHA = 0.98  # [discount] discount factor of the MPPI controller 控制成本的係數
SIGMA = np.array([[0.075, 0.0], [0.0, 2.0]])  # [variance] variance of the MPPI controller 控制雜訊的共變異數(雜訊太小 → 不敢轉，最後停下來；雜訊太大 → 亂轉，最後停下來)
STAGE_COST_WEIGHT = np.array([50.0, 50.0, 1.0, 20.0])  # [weight] stage cost weight of the MPPI controller 每一步的權重：位置、位置、航向、速度誤差
TERMINAL_COST_WEIGHT = np.array([50.0, 50.0, 1.0, 20.0])  # [weight] terminal cost weight of the MPPI controller 只在第 20 步（終點）算一次的權重
COLLISION_SAFETY_MARGIN_RATE = 1.2  # [rate] collision safety margin rate of the MPPI controller 碰撞安全邊界的倍率，1.0 = 剛好碰到，1.2 = 邊界放大 20%
# ---- Parameter options END ----


def run_simulation_mppi_pathtracking() -> None:
    print("[INFO] Start simulation of pathtracking with (vectorized) MPPI controller")

    delta_t = DELTA_T  # [sec]
    sim_steps = SIM_STEPS  # [steps]

    OBSTACLE_CIRCLES = np.array([
        [+ 8.0, +5.0, 5.0],  # pos_x, pos_y, radius [m]
        [+18.0, -5.0, 4.0],
    ])

    ref_path = np.genfromtxt('./data/ovalpath.csv',delimiter=',', skip_header=1)

    vehicle = Vehicle(
        wheel_base=WHEEL_BASE,  # [m] default: 2.5
        vehicle_width=VEHICLE_WIDTH,  # [m] drawing only
        vehicle_length=VEHICLE_LENGTH,  # [m] drawing only
        max_steer_abs=MAX_STEER_ABS,  # [rad] default: 0.523
        max_accel_abs=MAX_ACCEL_ABS,  # [m/s^2] default: 2.000
        ref_path=ref_path[:, 0:2],
        obstacle_circles=OBSTACLE_CIRCLES,
        delta_t=DELTA_T,  # [s] default: 0.05
        visualize=False,  # drawing is done by LiveViewer (much faster)
    )
    vehicle.reset(init_state=np.array([0.0, 0.0, 0.0, 0.0]))

    mppi = MPPIControllerForPathTracking(
        delta_t=DELTA_T*2.0,
        wheel_base=WHEEL_BASE,  # [m] default: 2.5
        vehicle_width=VEHICLE_WIDTH,  # [m] default: 3.0
        vehicle_length=VEHICLE_LENGTH,  # [m] default: 4.0
        max_steer_abs=MAX_STEER_ABS,  # [rad] default: 0.523
        max_accel_abs=MAX_ACCEL_ABS,  # [m/s^2] default: 2.000
        ref_path=ref_path,
        horizon_step_T=HORIZON_STEP_T,  # [steps] default: 20
        number_of_samples_K=NUMBER_OF_SAMPLES_K,  # [samples] default: 100
        param_exploration=PARAM_EXPLORATION,  # [variance] default: 0.05
        param_lambda=PARAM_LAMBDA,  # [weight] default: 100.0
        param_alpha=PARAM_ALPHA,  # [discount] default: 0.98
        sigma=SIGMA, # [variance] default: [[0.075, 0.0], [0.0, 2.0]] # [[steer, 0], [0, acceleration]]
        stage_cost_weight=STAGE_COST_WEIGHT, # [weight] default: [50.0, 50.0, 1.0, 20.0]
        terminal_cost_weight=TERMINAL_COST_WEIGHT, # [weight] default: [50.0, 50.0, 1.0, 20.0]
        visualze_sampled_trajs=SHOW_SAMPLED_TRAJS,
        obstacle_circles=OBSTACLE_CIRCLES,
        collision_safety_margin_rate=COLLISION_SAFETY_MARGIN_RATE,  # [rate] default: 1.2
    )

    def make_viewer(offscreen=False):
        return LiveViewer(ref_path[:, 0:2], OBSTACLE_CIRCLES, vehicle_width=VEHICLE_WIDTH,
                          vehicle_length=VEHICLE_LENGTH, max_steer_abs=MAX_STEER_ABS,
                          max_accel_abs=MAX_ACCEL_ABS, show_sampled_trajs=SHOW_SAMPLED_TRAJS,
                          offscreen=offscreen)

    # Vehicle(visualize=True) opens its own (empty) figure; close any such figure so only the viewer is shown
    if getattr(vehicle, "fig", None) is not None:
        plt.close(vehicle.fig)
    plt.close("all")

    viewer = None
    if LIVE_VIEW:
        plt.ion()
        viewer = make_viewer()
        viewer.update(vehicle.get_state(), 0.0, 0.0, t=0.0, step=0)
        viewer.fig.show()
        viewer.refresh()

    records = []  # what each step looked like, used to render the mp4 afterwards
    t_ctrl = 0.0
    t_start = time.perf_counter()
    for i in range(sim_steps):
        current_state = vehicle.get_state()
        try:
            t0 = time.perf_counter()
            optimal_input, _, optimal_traj, sampled_traj_list = mppi.calc_control_input(
                observed_x=current_state)
            t_ctrl += time.perf_counter() - t0
        except IndexError:
            print("[ERROR] IndexError detected. Terminate simulation.")
            break

        if i % PRINT_EVERY == 0:
            print(f"Time: {i*delta_t:>2.2f}[s], x={current_state[0]:>+3.3f}[m], y={current_state[1]:>+3.3f}[m], "
                  f"yaw={current_state[2]:>+3.3f}[rad], v={current_state[3]:>+3.3f}[m/s], "
                  f"steer={optimal_input[0]:>+6.2f}[rad], accel={optimal_input[1]:>+6.2f}[m/s]")

        vehicle.update(u=optimal_input, delta_t=delta_t, append_frame=False)
        # vehicle.update(u=optimal_input, delta_t=delta_t, append_frame=True)
        

        frame = dict(state=vehicle.get_state(), steer=optimal_input[0], accel=optimal_input[1],
                     t=(i + 1) * delta_t, step=i + 1,
                     optimal_traj=optimal_traj[:, 0:2].copy(),
                     sampled_trajs=sampled_traj_list[:, :, 0:2].copy() if SHOW_SAMPLED_TRAJS else None)
        if SAVE_MP4:
            records.append(frame)

        if LIVE_VIEW:
            # closing the window stops the simulation
            if not viewer.is_open():
                print("[INFO] window closed, stopping simulation")
                break
            if i % LIVE_EVERY == 0:
                viewer.update(**frame)
                viewer.refresh()
            if REALTIME:
                # wait until the wall clock catches up with the simulation clock
                wait = t_start + (i + 1) * delta_t / PLAYBACK_SPEED - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)

    n_done = i + 1
    elapsed = time.perf_counter() - t_start
    print(f"[INFO] simulation loop finished in {elapsed:.1f} s for {n_done*delta_t:.1f} s of simulation "
          f"(MPPI {t_ctrl / n_done * 1000:.1f} ms/step)")

    if SAVE_MP4 and records:
        t0 = time.perf_counter()
        # render the video in a separate in-memory figure (fixed 900x900 px), independent of the window
        saver = make_viewer(offscreen=True)
        saver.save_mp4(records, OUTPUT_FILE, interval_ms=int(delta_t * 1000))
        print(f"[INFO] saved {OUTPUT_FILE} ({len(records)} frames) in {time.perf_counter() - t0:.1f} s")

    if LIVE_VIEW and KEEP_WINDOW_OPEN and viewer.is_open():
        viewer.update(**frame)  # show the last step
        plt.ioff()
        print("[INFO] close the window to exit")
        plt.show()


if __name__ == "__main__":
    run_simulation_mppi_pathtracking()