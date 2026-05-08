# Closed-loop simulation: gyrostat truth + sensors + MEKF + controller.
# Sensor parameters and the MEKF come from HW3 so the two homeworks stay in
# sync; the truth dynamics and disturbances come from this directory.

import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'hw3'))

from scipy.linalg import block_diag

from hw3.utils import L, expq
from hw3.sensors import bearing_sensor, star_tracker, gyro, sample_M
from hw3.mekf import (predict, predict_jac, bearing_predict, bearing_jac,
                      star_innov, star_jac)
from hw3.mekf_sim import (W_bearing, W_st, W_gyro_PSD, V_bias_PSD,
                           sigma_align_bearing, sigma_align_gyro,
                           sigma_scale_gyro, sigma_arw,
                           derive_V, joseph_update, small_dq, sun_eci, r_N)

from hw4.dynamics import rk4_step, make_state, coe2rv
from hw4.actuators import saturate_wheel_torque
from hw4.disturbances import DEFAULT_SURFACES


def run_closed_loop(controller, t_final, dt=1.0, x0_truth=None,
                    q0_est=None, beta0_est=None, P0=None,
                    use_grav=True, use_drag=True,
                    use_st=True, use_bearing=True,
                    surfaces=DEFAULT_SURFACES, seed=42):
    """controller(k, q_hat, omega_hat, rho_hat) -> (u_W, info_dict).
    The truth state is the 16-D [r, v, q, omega, rho]. Returns dict of arrays."""
    rng = np.random.default_rng(seed)
    n = int(round(t_final / dt)) + 1

    if x0_truth is None:
        a, e, inc = 6800.0, 0.001, np.radians(51.64)
        r0, v0 = coe2rv(a, e, inc, np.radians(30), np.radians(60), 0.0)
        x0_truth = make_state(r0, v0, np.array([1.0, 0, 0, 0]), np.zeros(3))
    x = x0_truth.copy()

    M_b = sample_M(sigma_align_bearing, rng=rng)
    M_g = sample_M(sigma_align_gyro, sigma_scale_gyro, rng=rng)
    sun_s = bearing_sensor(sun_eci, W_bearing, M=M_b, rng=rng)
    st = star_tracker(W_st, rng=rng)
    g = gyro(dt, W_gyro_PSD, V_bias_PSD, M=M_g, rng=rng)

    if q0_est is None:
        # 5 deg half-angle MEKF prior offset around the truth attitude
        q0_est = L(x[6:10]) @ expq(np.radians(5) * rng.standard_normal(3))
        q0_est /= np.linalg.norm(q0_est)
    if beta0_est is None:
        beta0_est = np.zeros(3)
    if P0 is None:
        P0 = block_diag(np.radians(30)**2 * np.eye(3),
                        (sigma_arw / np.sqrt(dt))**2 * np.eye(3))
    q_hat, b_hat, P = q0_est.copy(), beta0_est.copy(), P0.copy()
    V = derive_V(dt)

    x_hist = np.zeros((16, n)); x_hist[:, 0] = x
    q_hat_hist = np.zeros((4, n)); q_hat_hist[:, 0] = q_hat
    b_hat_hist = np.zeros((3, n)); b_hat_hist[:, 0] = b_hat
    omega_hat_hist = np.zeros((3, n))
    P_hist = np.zeros((6, 6, n)); P_hist[:, :, 0] = P
    u_W_hist = np.zeros((4, n - 1))
    tau_cmd_hist = np.zeros((3, n - 1))

    gyro_meas = g.getMsmt(x[10:13])
    omega_hat = gyro_meas - b_hat
    omega_hat_hist[:, 0] = omega_hat

    for k in range(n - 1):
        # Control from current estimate
        u_W, info = controller(k, q_hat, omega_hat, x_hist[13:16, k])
        u_W = saturate_wheel_torque(u_W)
        u_W_hist[:, k] = u_W
        if 'tau_cmd' in info:
            tau_cmd_hist[:, k] = info['tau_cmd']

        # Truth one step (zero-order hold on u_W)
        x = rk4_step(x, u_W, dt, use_grav=use_grav, use_drag=use_drag,
                     surfaces=surfaces)
        x_hist[:, k + 1] = x

        # Measurements
        gyro_meas = g.getMsmt(x[10:13])
        bear_meas = sun_s.getMsmt(x[6:10])
        st_meas = st.getMsmt(x[6:10])

        # MEKF: predict + sequential bearing/star updates
        q_hat, b_hat = predict(q_hat, b_hat, gyro_meas, dt)
        A = predict_jac(q_hat, b_hat, gyro_meas, dt)
        P = A @ P @ A.T + V

        if use_bearing:
            z = bear_meas - bearing_predict(q_hat, r_N)
            C = bearing_jac(q_hat, r_N)
            K = P @ C.T @ np.linalg.inv(C @ P @ C.T + W_bearing)
            dx = K @ z
            q_hat = L(q_hat) @ small_dq(dx[0:3]); q_hat /= np.linalg.norm(q_hat)
            b_hat = b_hat + dx[3:6]
            P = joseph_update(P, K, C, W_bearing)
        if use_st:
            z = star_innov(q_hat, st_meas)
            C = star_jac(q_hat, st_meas)
            K = P @ C.T @ np.linalg.inv(C @ P @ C.T + W_st)
            dx = -K @ z   # vector-part residual sign convention (Lecture 12)
            q_hat = L(q_hat) @ small_dq(dx[0:3]); q_hat /= np.linalg.norm(q_hat)
            b_hat = b_hat + dx[3:6]
            P = joseph_update(P, K, C, W_st)

        omega_hat = gyro_meas - b_hat
        q_hat_hist[:, k + 1] = q_hat
        b_hat_hist[:, k + 1] = b_hat
        omega_hat_hist[:, k + 1] = omega_hat
        P_hist[:, :, k + 1] = P

    return dict(t=np.arange(n) * dt, x=x_hist,
                q_hat=q_hat_hist, b_hat=b_hat_hist, omega_hat=omega_hat_hist,
                P=P_hist, u_W=u_W_hist, tau_cmd=tau_cmd_hist)
