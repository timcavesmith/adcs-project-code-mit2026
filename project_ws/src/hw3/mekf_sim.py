# MEKF sim HW3 Part 2

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.linalg import block_diag

from utils import hat, H, T, L, R, G, Q, expq, logq
from sensors import bearing_sensor, star_tracker, gyro
from mekf import (predict, predict_jac, bearing_predict, bearing_jac,
                  star_tracker_innov, star_tracker_jac, quat_err)

out_dir = os.path.dirname(__file__)

# dreamchaser inertia
J = np.array([[12685.21,      0,  -1358.05],
              [     0,  51407.14,       0  ],
              [-1358.05,      0,  57999.34]])

def dynamics(x):
    q = x[0:4]; omega = x[4:7]
    qdot = 0.5 * G(q) @ omega
    omegadot = np.linalg.solve(J, -hat(omega) @ J @ omega)
    return np.concatenate([qdot, omegadot])

def rk4step(x, dt):
    k1 = dynamics(x)
    k2 = dynamics(x + k1 * dt / 2)
    k3 = dynamics(x + k2 * dt / 2)
    k4 = dynamics(x + dt * k3)
    xn = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    xn[0:4] /= np.linalg.norm(xn[0:4])
    return xn


# sensor params
M_bearing = np.array([[ 9.99999998e-01, -4.84790564e-05,  4.84837572e-05],
                      [ 4.84814068e-05,  9.99999998e-01, -4.84790564e-05],
                      [-4.84814069e-05,  4.84814068e-05,  9.99999998e-01]])
covW_bearing = 2.5e-3 * np.eye(3)
b_bearing = np.zeros(3)

M_gyro_mat = np.array([[ 9.99984545e-01, -7.27262750e-05, -7.27156979e-05],
                       [ 7.27209865e-05,  9.99984545e-01, -7.27262750e-05],
                       [ 7.27232338e-05,  7.27232336e-05,  1.00001544e+00]])
covW_gyro = 1.04e-12 * np.eye(3)   # rad^2/s
covBias_gyro = 2.8e-16 * np.eye(3) # rad^2/s^2

covW_st = np.diag([2.35e-11, 2.35e-11, 1.5e-9])  # rad^2

# inertial reference vectors (sun and nadir)
r_N = np.array([[1, 0, 0],
                [0, 0, 1]], dtype=float).T  # (3,2)

M_b_inv = np.linalg.inv(M_bearing)
M_g_inv = np.linalg.inv(M_gyro_mat)


def run_mekf(rate=10, tf=60.0, q0_true=None, omega0=None,
             q0_est=None, P0=None, seed=42, use_st=True):
    np.random.seed(seed)
    dt = 1.0 / rate
    n = int(tf * rate)

    if q0_true is None:
        q0_true = np.random.randn(4)
        q0_true /= np.linalg.norm(q0_true)
    if omega0 is None:
        omega0 = np.radians(3.0) * np.random.randn(3)

    # propagate truth
    xtraj = np.zeros((7, n))
    xtraj[:, 0] = np.concatenate([q0_true, omega0])
    for k in range(n - 1):
        xtraj[:, k + 1] = rk4step(xtraj[:, k], dt)

    # sensors
    sun = bearing_sensor([1, 0, 0], b_bearing, M_bearing, covW_bearing)
    nad = bearing_sensor([0, 0, 1], b_bearing, M_bearing, covW_bearing)
    st = star_tracker(covW_st)
    g = gyro(rate, M_gyro_mat, covBias_gyro, covW_gyro)
    g.b = np.zeros(3)

    # measurements
    gyro_meas = np.zeros((3, n))
    bear_meas = np.zeros((6, n))
    st_meas = np.zeros((4, n))
    bias_true = np.zeros((3, n))

    for k in range(n):
        gyro_meas[:, k] = g.getMsmt(xtraj[4:7, k])
        bias_true[:, k] = g.b.copy()
        y1 = sun.getMsmt(xtraj[0:4, k])
        y2 = nad.getMsmt(xtraj[0:4, k])
        bear_meas[:, k] = np.concatenate([y1, y2])
        st_meas[:, k] = st.getMsmt(xtraj[0:4, k])

    # undo known calibration
    for k in range(n):
        bear_meas[0:3, k] = M_b_inv @ (bear_meas[0:3, k] - b_bearing)
        bear_meas[3:6, k] = M_b_inv @ (bear_meas[3:6, k] - b_bearing)
        gyro_meas[:, k] = M_g_inv @ gyro_meas[:, k]

    # process noise (tuned up from raw specs to avoid smug filter)
    V_att = covW_gyro * np.sqrt(rate) + 1e-6 * np.eye(3)
    V_bias = covBias_gyro / np.sqrt(rate) + 1e-6 * np.eye(3)
    V = block_diag(V_att, V_bias)

    W_bear = block_diag(covW_bearing, covW_bearing)
    W_st_mat = covW_st

    # filter init
    if q0_est is None:
        phi_init = np.radians(10) * np.random.randn(3)
        q0_est = L(q0_true) @ expq(phi_init)
        q0_est /= np.linalg.norm(q0_est)
    if P0 is None:
        P0 = 0.5 * np.eye(6)

    q_filt = np.zeros((4, n))
    beta_filt = np.zeros((3, n))
    P_hist = np.zeros((6, 6, n))
    q_filt[:, 0] = q0_est
    beta_filt[:, 0] = np.zeros(3)
    P_hist[:, :, 0] = P0

    I6 = np.eye(6)

    for k in range(n - 1):
        qk = q_filt[:, k]
        bk = beta_filt[:, k]
        Pk = P_hist[:, :, k]

        # predict
        q_pred, b_pred = predict(qk, bk, gyro_meas[:, k], dt)
        A = predict_jac(qk, bk, gyro_meas[:, k], dt)
        P_pred = A @ Pk @ A.T + V

        # update
        z_b = bear_meas[:, k + 1] - bearing_predict(q_pred, r_N)
        C_b = bearing_jac(q_pred, r_N)

        if use_st:
            z_st = star_tracker_innov(q_pred, st_meas[:, k + 1])
            C_st = star_tracker_jac()
            z = np.concatenate([z_b, z_st])
            C = np.vstack([C_b, C_st])
            W = block_diag(W_bear, W_st_mat)
        else:
            z = z_b
            C = C_b
            W = W_bear

        S = C @ P_pred @ C.T + W
        K = P_pred @ C.T @ np.linalg.inv(S)

        dx = K @ z
        phi = dx[0:3]
        dbeta = dx[3:6]

        phi_sq = phi @ phi
        if phi_sq < 1.0:
            dq = np.concatenate([[np.sqrt(1 - phi_sq)], phi])
        else:
            dq = expq(phi)

        q_upd = L(q_pred) @ dq
        q_upd /= np.linalg.norm(q_upd)
        b_upd = b_pred + dbeta

        P_upd = (I6 - K @ C) @ P_pred @ (I6 - K @ C).T + K @ W @ K.T

        q_filt[:, k + 1] = q_upd
        beta_filt[:, k + 1] = b_upd
        P_hist[:, :, k + 1] = P_upd

    # errors
    phi_err = np.zeros((3, n))
    for k in range(n):
        phi_err[:, k] = quat_err(q_filt[:, k], xtraj[0:4, k])
    angle_err_deg = 2 * np.degrees(np.linalg.norm(phi_err, axis=0))
    beta_err = beta_filt - bias_true
    t = np.linspace(0, tf, n)

    return dict(t=t, xtraj=xtraj, q_filt=q_filt, beta_filt=beta_filt,
                P_hist=P_hist, phi_err=phi_err, angle_err_deg=angle_err_deg,
                beta_err=beta_err, bias_true=bias_true)


if __name__ == "__main__":

    # demo: quat tracking + angle error
    res = run_mekf(rate=10, tf=60, seed=42, use_st=True)
    t = res['t']

    colors = ['C0', 'C1', 'C2', 'C3']
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for i in range(4):
        axes[0].plot(t, res['xtraj'][i, :], '--', color=colors[i], alpha=0.4, linewidth=1)
        axes[0].plot(t, res['q_filt'][i, :], color=colors[i], linewidth=1.2, label=f'q{i}')
    axes[0].set_ylabel('Quaternion')
    axes[0].legend(ncol=4, fontsize=7, loc='upper right')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, res['angle_err_deg'], linewidth=0.7)
    axes[1].set_ylabel('Attitude error [deg]')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    plt.suptitle('MEKF demo: quaternion tracking and attitude error')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'hw3_mekf_demo.png'), dpi=150, bbox_inches='tight')

    # sample rate comparison — plot low rates last so they're on top
    rates = [50, 10, 1]
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    for rate in rates:
        r = run_mekf(rate=rate, tf=60, seed=42, use_st=True)
        ax2.plot(r['t'], r['angle_err_deg'], linewidth=0.6, alpha=0.8, label=f'{rate} Hz')
    ax2.set_ylabel('Attitude error [deg]')
    ax2.set_xlabel('Time [s]')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('MEKF error at different sample rates')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'hw3_rate_comparison.png'), dpi=150, bbox_inches='tight')

    # consistency: skip initial transient so y-axis shows steady-state behavior
    res3 = run_mekf(rate=10, tf=60, seed=7, use_st=True)
    t3 = res3['t']
    i_start = int(5.0 * 10)
    fig3, axes3 = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
    labels_phi = [r'$\phi_1$', r'$\phi_2$', r'$\phi_3$']
    labels_b = [r'$\beta_1$', r'$\beta_2$', r'$\beta_3$']
    for i in range(3):
        sig = 3 * np.sqrt(res3['P_hist'][i, i, i_start:])
        axes3[0, i].plot(t3[i_start:], res3['phi_err'][i, i_start:], 'b', linewidth=0.5)
        axes3[0, i].plot(t3[i_start:], sig, 'r--', linewidth=0.8)
        axes3[0, i].plot(t3[i_start:], -sig, 'r--', linewidth=0.8)
        axes3[0, i].set_title(labels_phi[i])
        axes3[0, i].grid(True, alpha=0.3)

        sig_b = 3 * np.sqrt(res3['P_hist'][3 + i, 3 + i, i_start:])
        axes3[1, i].plot(t3[i_start:], res3['beta_err'][i, i_start:], 'b', linewidth=0.5)
        axes3[1, i].plot(t3[i_start:], sig_b, 'r--', linewidth=0.8)
        axes3[1, i].plot(t3[i_start:], -sig_b, 'r--', linewidth=0.8)
        axes3[1, i].set_title(labels_b[i])
        axes3[1, i].set_xlabel('Time [s]')
        axes3[1, i].grid(True, alpha=0.3)

    axes3[0, 0].set_ylabel('Attitude error [rad]')
    axes3[1, 0].set_ylabel('Bias error [rad/s]')
    plt.suptitle('MEKF consistency: errors vs 3-sigma bounds (t > 5 s)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'hw3_consistency.png'), dpi=150, bbox_inches='tight')

    # convergence — plot largest error first so smaller ones draw on top
    q0_true = expq(0.3 * np.array([1, 0.5, -0.7]))
    omega0 = np.radians(np.array([2, -1.5, 1.0]))
    init_errs_deg = [170, 90, 30, 5]

    fig4, ax4 = plt.subplots(figsize=(10, 4))
    for err_deg in init_errs_deg:
        np.random.seed(10)
        phi_init = (np.radians(err_deg) / 2) * np.array([1.0, 0, 0])
        q0_est = L(q0_true) @ expq(phi_init)
        q0_est /= np.linalg.norm(q0_est)
        r = run_mekf(rate=10, tf=15, q0_true=q0_true, omega0=omega0,
                     q0_est=q0_est, seed=10, use_st=True)
        ax4.plot(r['t'], r['angle_err_deg'], linewidth=0.7, label='%d deg' % err_deg)

    ax4.set_ylabel('Attitude error [deg]')
    ax4.set_xlabel('Time [s]')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_title('MEKF convergence from different initial errors')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'hw3_convergence.png'), dpi=150, bbox_inches='tight')

    # monte carlo
    n_mc = 20
    ss_errors = []
    for trial in range(n_mc):
        r = run_mekf(rate=10, tf=60, seed=100 + trial, use_st=True)
        ss_errors.append(np.mean(r['angle_err_deg'][-100:]))
    ss_errors = np.array(ss_errors)

    print('--- Monte Carlo ({} runs) ---'.format(n_mc))
    print('Mean SS error: %.6f deg' % np.mean(ss_errors))
    print('Std:           %.6f deg' % np.std(ss_errors))
    print('Max:           %.6f deg' % np.max(ss_errors))
    print('\nHW2 static (Wahba q-method): 3.07e-05 deg')
    print('MEKF SS mean:               %.2e deg' % np.mean(ss_errors))

    plt.show()
    print('\nPlots saved to', out_dir)
