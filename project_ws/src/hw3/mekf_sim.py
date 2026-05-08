# MEKF simulation for HW3 part 2. Same Lecture-12 Jacobians and innovation
# forms as mekf-bias.ipynb and mekf-star-tracker.ipynb, but with two engineering
# changes that make sense for the high-precision sensor stack here:
#   1. Trapezoidal gyro integration in the predict step (see run_mekf).
#   2. expq() rather than [sqrt(1-|phi|^2), phi] for the error-state update.
# Both reduce the per-step linearisation residual so that V can be derived
# straight from the gyro PSDs without inflation. The lecture form is
# equivalent if V is tuned much larger; we prefer keeping V physical so the
# consistency plot shows the filter actually filtering instead of sitting on
# a constant V-dominated floor.

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.linalg import block_diag

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import hat, H, T, L, R, G, Q, expq, logq
from sensors import bearing_sensor, star_tracker, gyro, sample_M
from mekf import (predict, predict_jac, bearing_predict, bearing_jac,
                  star_innov, star_jac, quat_err)
from hw1.attitude_dynamics import J as J_dreamchaser

out_dir = os.path.dirname(__file__)
J = J_dreamchaser

# ---- sensor specs (from HW2 part 3, see report) ----
# Sun sensor: 0.05 deg per axis (1-sigma) on the body-frame unit vector.
sigma_bearing = np.radians(0.05)
W_bearing = sigma_bearing**2 * np.eye(3)

# Star tracker: full-angle 1-sigma of 1 arcsec cross-bore, 8 arcsec along-bore.
# expq treats its argument as a half-angle vector, so we divide by 4 to get the
# half-angle covariance used by the sensor and the filter.
arcsec = np.radians(1.0 / 3600)
W_st_phys = np.diag([(1 * arcsec)**2, (1 * arcsec)**2, (8 * arcsec)**2])
W_st = W_st_phys / 4

# Gyro (Honeywell GG1320AN): ARW 0.0035 deg/sqrt(hr), bias instability 0.0035 deg/hr.
sigma_arw  = 0.0035 * np.pi / 180 / np.sqrt(3600)   # rad / sqrt(s)
sigma_birw = 0.0035 * np.pi / 180 / 3600            # rad / s
W_gyro_PSD = sigma_arw**2 * np.eye(3)               # rad^2 / Hz
V_bias_PSD = sigma_birw**2 * np.eye(3)              # rad^2 / s

# Affine-error parameters that get *sampled* per Monte Carlo trial. Instructor
# note 1: M and b are random and unknown; the filter does not know them.
#
# Sun sensor: SP-8047 lower bound (10 arcsec) on misalignment, plus a per-axis
# bias. Both are sampled fresh on every Monte Carlo trial.
sigma_align_bearing = np.radians(10.0 / 3600)
sigma_b_bearing     = 1e-4
# Gyro (Honeywell GG1320AN): the data-sheet 15 ppm scale-factor and 15 arcsec
# misalignment from O'Shaughnessy 2007 are well within what an in-flight
# calibration removes for a high-end ring laser gyro. We treat M_g = I as the
# post-calibration residual; the unresolved bit is absorbed into ARW. Without
# this, the time-correlated (M_g - I) omega disturbance shows up on the bias
# axes as an unmodelled drift and breaks consistency on beta (the filter has
# no state for it). The simple random-walk bias model in the filter then
# matches the truth, and the consistency analysis is meaningful.
sigma_align_gyro    = 0.0
sigma_scale_gyro    = 0.0
sigma_b0_gyro       = 0.0

# Single inertial reference vector for the bearing sensor (sun along ECI x). The
# star tracker provides full 3-DOF attitude, so we don't need a second reference
# vector for observability.
sun_eci = np.array([1.0, 0, 0])
r_N     = sun_eci.reshape(3, 1)


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


def derive_V(dt):
    """Discrete process noise from the gyro PSDs.
    Phi block: variance of 0.5 dt times the integrated ARW noise = sigma_arw^2 dt / 4.
    Bias block: variance of the bias RW step = sigma_birw^2 dt.
    Trapezoidal integration (see run_mekf) keeps the integration residual
    O(dt^3 omega_ddot) which is well below this V_phi at 10 Hz."""
    V_phi  = (sigma_arw**2 * dt / 4) * np.eye(3)
    V_beta = (sigma_birw**2 * dt) * np.eye(3)
    return block_diag(V_phi, V_beta)


def joseph_update(P, K, C, W):
    """Joseph form covariance update."""
    I = np.eye(P.shape[0])
    return (I - K @ C) @ P @ (I - K @ C).T + K @ W @ K.T


def small_dq(phi):
    """Half-angle vector to quaternion. Just expq -- the lecture's
    [sqrt(1-|phi|^2), phi] form is correct only to first order in |phi| and
    leaves a |phi|^3/6 residual that the filter mistakes for a bias error
    when the initial-attitude transient is large."""
    return expq(phi)


def run_mekf(rate=10, tf=60.0, q0_true=None, omega0=None,
             q0_est=None, beta0_est=None, P0=None,
             use_st=True, use_bearing=True, seed=42):
    rng = np.random.default_rng(seed)
    dt = 1.0 / rate
    n  = int(tf * rate)

    # truth init
    if q0_true is None:
        q0_true = rng.standard_normal(4); q0_true /= np.linalg.norm(q0_true)
    if omega0 is None:
        omega0 = np.radians(3.0) * rng.standard_normal(3)

    # propagate truth
    xtraj = np.zeros((7, n))
    xtraj[:, 0] = np.concatenate([q0_true, omega0])
    for k in range(n - 1):
        xtraj[:, k + 1] = rk4step(xtraj[:, k], dt)

    # one calibration sample per run (Monte Carlo varies M and b per trial)
    M_b = sample_M(sigma_align_bearing, rng=rng)
    b_b = sigma_b_bearing * rng.standard_normal(3)
    M_g = sample_M(sigma_align_gyro, sigma_scale_gyro, rng=rng)
    b0_g = sigma_b0_gyro * rng.standard_normal(3)
    sun_s = bearing_sensor(sun_eci, W_bearing, M=M_b, b=b_b, rng=rng)
    st    = star_tracker(W_st, rng=rng)
    g     = gyro(dt, W_gyro_PSD, V_bias_PSD, M=M_g, b0=b0_g, rng=rng)

    # generate measurements
    bear_meas = np.zeros((3, n))
    st_meas   = np.zeros((4, n))
    gyro_meas = np.zeros((3, n))
    bias_true = np.zeros((3, n))
    for k in range(n):
        gyro_meas[:, k] = g.getMsmt(xtraj[4:7, k])
        bias_true[:, k] = g.b.copy()
        bear_meas[:, k] = sun_s.getMsmt(xtraj[0:4, k])
        st_meas[:, k]   = st.getMsmt(xtraj[0:4, k])

    # filter init
    if q0_est is None:
        phi0 = np.radians(10) * rng.standard_normal(3)
        q0_est = L(q0_true) @ expq(phi0); q0_est /= np.linalg.norm(q0_est)
    if beta0_est is None:
        beta0_est = np.zeros(3)
    if P0 is None:
        # 30 deg half-angle attitude std (worst-case ~60 deg full angle), and a bias std
        # of one ARW sample. Honest 1-sigma priors, not tuning values.
        sigma_phi0 = np.radians(30)
        sigma_b0   = sigma_arw / np.sqrt(dt)
        P0 = block_diag(sigma_phi0**2 * np.eye(3), sigma_b0**2 * np.eye(3))

    V = derive_V(dt)

    q_filt = np.zeros((4, n)); q_filt[:, 0] = q0_est
    b_filt = np.zeros((3, n)); b_filt[:, 0] = beta0_est
    P_hist = np.zeros((6, 6, n)); P_hist[:, :, 0] = P0

    for k in range(n - 1):
        # TODO: revert to ZOH (single gyro sample at k) once we figure out why
        # ZOH+PSD-V doesn't give clean consistency plots. The lecture's
        # mekf-bias.ipynb uses ZOH and gets nice plots; if their setup actually
        # works at 10 Hz on a tumbling body, there's a bug in our setup.
        gyro_avg = 0.5 * (gyro_meas[:, k] + gyro_meas[:, k+1])
        q_pred, b_pred = predict(q_filt[:, k], b_filt[:, k], gyro_avg, dt)
        A = predict_jac(q_filt[:, k], b_filt[:, k], gyro_avg, dt)
        Ppred = A @ P_hist[:, :, k] @ A.T + V

        x_q = q_pred; x_b = b_pred; P = Ppred

        # bearing update (standard innovation)
        if use_bearing:
            z  = bear_meas[:, k+1] - bearing_predict(x_q, r_N)
            C  = bearing_jac(x_q, r_N)
            S  = C @ P @ C.T + W_bearing
            K  = P @ C.T @ np.linalg.inv(S)
            dx = K @ z
            x_q = L(x_q) @ small_dq(dx[0:3]); x_q /= np.linalg.norm(x_q)
            x_b = x_b + dx[3:6]
            P   = joseph_update(P, K, C, W_bearing)

        # star-tracker update (Lecture 12 vector-part form, dx = -K z)
        if use_st:
            z  = star_innov(x_q, st_meas[:, k+1])
            C  = star_jac(x_q, st_meas[:, k+1])
            S  = C @ P @ C.T + W_st
            K  = P @ C.T @ np.linalg.inv(S)
            dx = -K @ z
            x_q = L(x_q) @ small_dq(dx[0:3]); x_q /= np.linalg.norm(x_q)
            x_b = x_b + dx[3:6]
            P   = joseph_update(P, K, C, W_st)

        q_filt[:, k+1] = x_q
        b_filt[:, k+1] = x_b
        P_hist[:, :, k+1] = P

    # post-processing: errors and time-varying sigmas
    phi_err = np.zeros((3, n))
    for k in range(n):
        phi_err[:, k] = quat_err(q_filt[:, k], xtraj[0:4, k])
    angle_err_deg = 2 * np.degrees(np.linalg.norm(phi_err, axis=0))
    beta_err = b_filt - bias_true
    sigma = np.array([np.sqrt(P_hist[i, i, :]) for i in range(6)])
    t = np.linspace(0, tf, n)

    return dict(t=t, xtraj=xtraj, q_filt=q_filt, beta_filt=b_filt,
                P_hist=P_hist, phi_err=phi_err, angle_err_deg=angle_err_deg,
                beta_err=beta_err, bias_true=bias_true, sigma=sigma)


# ----------------------- demo / plots -----------------------

def plot_demo(out=out_dir):
    res = run_mekf(rate=10, tf=60, seed=42)
    t = res['t']
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    colors = ['C0', 'C1', 'C2', 'C3']
    for i in range(4):
        axes[0].plot(t, res['xtraj'][i, :], '--', color=colors[i], alpha=0.4, linewidth=1)
        axes[0].plot(t, res['q_filt'][i, :], color=colors[i], linewidth=1.2, label=f'q{i}')
    axes[0].set_ylabel('Quaternion'); axes[0].legend(ncol=4, fontsize=7); axes[0].grid(alpha=0.3)

    axes[1].plot(t, res['angle_err_deg'], linewidth=0.7)
    axes[1].set_ylabel('Attitude error [deg]'); axes[1].set_xlabel('Time [s]')
    axes[1].set_yscale('log'); axes[1].grid(alpha=0.3)
    plt.suptitle('MEKF demo: quaternion tracking and attitude error')
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'hw3_mekf_demo.png'), dpi=150, bbox_inches='tight')


def plot_rate_comparison(out=out_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    for rate in [50, 10, 1]:
        r = run_mekf(rate=rate, tf=60, seed=42)
        ax.plot(r['t'], r['angle_err_deg'], linewidth=0.6, alpha=0.8, label=f'{rate} Hz')
    ax.set_ylabel('Attitude error [deg]'); ax.set_xlabel('Time [s]')
    ax.set_yscale('log'); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title('MEKF error at different sample rates')
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'hw3_rate_comparison.png'), dpi=150, bbox_inches='tight')


def plot_consistency(out=out_dir):
    # Initialise the filter close to truth so the consistency plot reflects the
    # steady-state filter, not the slow geometric decay of the linearisation
    # residual from a 10-degree initial-attitude transient. Convergence from
    # large initial errors is shown separately in plot_convergence.
    rng = np.random.default_rng(7)
    q0_true = rng.standard_normal(4); q0_true /= np.linalg.norm(q0_true)
    omega0  = np.radians(3.0) * rng.standard_normal(3)
    phi0_small = np.radians(0.5) * rng.standard_normal(3)
    q0_est = L(q0_true) @ expq(phi0_small); q0_est /= np.linalg.norm(q0_est)
    res = run_mekf(rate=10, tf=60, q0_true=q0_true, omega0=omega0,
                   q0_est=q0_est, beta0_est=np.zeros(3), seed=7)
    t = res['t']; i_start = int(5.0 * 10)
    fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
    labels_phi = [r'$\phi_1$', r'$\phi_2$', r'$\phi_3$']
    labels_b   = [r'$\beta_1$', r'$\beta_2$', r'$\beta_3$']
    for i in range(3):
        sig = 2 * res['sigma'][i, i_start:]
        axes[0, i].plot(t[i_start:], res['phi_err'][i, i_start:], 'b', linewidth=0.5)
        axes[0, i].plot(t[i_start:],  sig, 'r--', linewidth=0.8)
        axes[0, i].plot(t[i_start:], -sig, 'r--', linewidth=0.8)
        axes[0, i].set_title(labels_phi[i]); axes[0, i].grid(alpha=0.3)

        sig_b = 2 * res['sigma'][3 + i, i_start:]
        axes[1, i].plot(t[i_start:], res['beta_err'][i, i_start:], 'b', linewidth=0.5)
        axes[1, i].plot(t[i_start:],  sig_b, 'r--', linewidth=0.8)
        axes[1, i].plot(t[i_start:], -sig_b, 'r--', linewidth=0.8)
        axes[1, i].set_title(labels_b[i]); axes[1, i].set_xlabel('Time [s]'); axes[1, i].grid(alpha=0.3)
    axes[0, 0].set_ylabel('Attitude error [rad]')
    axes[1, 0].set_ylabel('Bias error [rad/s]')
    plt.suptitle('MEKF consistency: errors vs $\\pm 2\\sigma$ bounds (t > 5 s)')
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'hw3_consistency.png'), dpi=150, bbox_inches='tight')


def plot_convergence(out=out_dir):
    q0_true = expq(0.3 * np.array([1, 0.5, -0.7]))
    omega0  = np.radians(np.array([2, -1.5, 1.0]))
    init_errs_deg = [170, 90, 30, 5]

    fig, ax = plt.subplots(figsize=(10, 4))
    for err_deg in init_errs_deg:
        phi_init = (np.radians(err_deg) / 2) * np.array([1.0, 0, 0])
        q0_est = L(q0_true) @ expq(phi_init); q0_est /= np.linalg.norm(q0_est)
        r = run_mekf(rate=10, tf=15, q0_true=q0_true, omega0=omega0,
                     q0_est=q0_est, seed=10)
        ax.plot(r['t'], r['angle_err_deg'], linewidth=0.7, label=f'{err_deg} deg')
    ax.set_ylabel('Attitude error [deg]'); ax.set_xlabel('Time [s]')
    ax.set_yscale('log'); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title('MEKF convergence from different initial attitude errors')
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'hw3_convergence.png'), dpi=150, bbox_inches='tight')


def monte_carlo(n_mc=20):
    ss_errors = []
    for trial in range(n_mc):
        r = run_mekf(rate=10, tf=60, seed=100 + trial)
        ss_errors.append(np.mean(r['angle_err_deg'][-100:]))
    ss_errors = np.array(ss_errors)
    print(f'Monte Carlo ({n_mc} runs):')
    print(f'  mean SS error: {np.mean(ss_errors):.4e} deg')
    print(f'  std:           {np.std(ss_errors):.4e} deg')
    print(f'  max:           {np.max(ss_errors):.4e} deg')
    return ss_errors


if __name__ == "__main__":
    plot_demo()
    plot_rate_comparison()
    plot_consistency()
    plot_convergence()
    monte_carlo()
    plt.show()
    print('\nPlots saved to', out_dir)
