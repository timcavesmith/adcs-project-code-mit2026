"""HW3 setup but with PURE ZOH propagation (no trapezoidal). PSD-derived V.
Plot the consistency to see if ZOH + PSD-V can give nice plots."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

from utils import L, R, G, Q, expq, logq, T, hat, H
from sensors import bearing_sensor, star_tracker, gyro, sample_M
from mekf import predict, predict_jac, bearing_predict, bearing_jac, star_innov, star_jac, quat_err
import mekf_sim as ms

dt = 0.1
n = 600
rng = np.random.default_rng(7)
q0_true = rng.standard_normal(4); q0_true /= np.linalg.norm(q0_true)
omega0  = np.radians(3.0) * rng.standard_normal(3)
phi0_small = np.radians(0.5) * rng.standard_normal(3)
q0_est = L(q0_true) @ expq(phi0_small); q0_est /= np.linalg.norm(q0_est)

# Truth
xtraj = np.zeros((7, n)); xtraj[:, 0] = np.concatenate([q0_true, omega0])
for k in range(n-1):
    xtraj[:, k+1] = ms.rk4step(xtraj[:, k], dt)

# Sensors (same as run_mekf)
M_b = sample_M(ms.sigma_align_bearing, rng=rng)
b_b = ms.sigma_b_bearing * rng.standard_normal(3)
sun_s = bearing_sensor(ms.sun_eci, ms.W_bearing, M=M_b, b=b_b, rng=rng)
st = star_tracker(ms.W_st, rng=rng)
g = gyro(dt, ms.W_gyro_PSD, ms.V_bias_PSD, M=np.eye(3), b0=np.zeros(3), rng=rng)

bear_meas = np.zeros((3, n))
st_meas   = np.zeros((4, n))
gyro_meas = np.zeros((3, n))
bias_true = np.zeros((3, n))
for k in range(n):
    gyro_meas[:, k] = g.getMsmt(xtraj[4:7, k])
    bias_true[:, k] = g.b.copy()
    bear_meas[:, k] = sun_s.getMsmt(xtraj[0:4, k])
    st_meas[:, k]   = st.getMsmt(xtraj[0:4, k])

# Filter init
sigma_phi0 = np.radians(30); sigma_b0 = ms.sigma_arw / np.sqrt(dt)
P0 = block_diag(sigma_phi0**2 * np.eye(3), sigma_b0**2 * np.eye(3))
V = ms.derive_V(dt)

q_filt = np.zeros((4, n)); q_filt[:, 0] = q0_est
b_filt = np.zeros((3, n)); b_filt[:, 0] = np.zeros(3)
P_hist = np.zeros((6, 6, n)); P_hist[:, :, 0] = P0

for k in range(n - 1):
    # PURE ZOH (single gyro sample at k) -- matches lecture
    q_pred, b_pred = predict(q_filt[:, k], b_filt[:, k], gyro_meas[:, k], dt)
    A = predict_jac(q_filt[:, k], b_filt[:, k], gyro_meas[:, k], dt)
    Ppred = A @ P_hist[:, :, k] @ A.T + V

    x_q = q_pred; x_b = b_pred; P = Ppred

    # bearing
    z = bear_meas[:, k+1] - bearing_predict(x_q, ms.r_N)
    C = bearing_jac(x_q, ms.r_N)
    S = C @ P @ C.T + ms.W_bearing
    K = P @ C.T @ np.linalg.inv(S)
    dx = K @ z
    x_q = L(x_q) @ ms.small_dq(dx[0:3]); x_q /= np.linalg.norm(x_q)
    x_b = x_b + dx[3:6]
    P = ms.joseph_update(P, K, C, ms.W_bearing)

    # star tracker
    z = star_innov(x_q, st_meas[:, k+1])
    C = star_jac(x_q, st_meas[:, k+1])
    S = C @ P @ C.T + ms.W_st
    K = P @ C.T @ np.linalg.inv(S)
    dx = -K @ z
    x_q = L(x_q) @ ms.small_dq(dx[0:3]); x_q /= np.linalg.norm(x_q)
    x_b = x_b + dx[3:6]
    P = ms.joseph_update(P, K, C, ms.W_st)

    q_filt[:, k+1] = x_q
    b_filt[:, k+1] = x_b
    P_hist[:, :, k+1] = P

# Compute errors
phi_err = np.zeros((3, n))
for k in range(n):
    phi_err[:, k] = quat_err(q_filt[:, k], xtraj[0:4, k])
beta_err = b_filt - bias_true
sig = np.array([np.sqrt(P_hist[i, i, :]) for i in range(6)])
t = np.linspace(0, n*dt, n)

# Plot
i_start = 50
fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
for i in range(3):
    s_phi = 2 * sig[i, i_start:]
    axes[0, i].plot(t[i_start:], phi_err[i, i_start:], 'b', linewidth=0.5)
    axes[0, i].plot(t[i_start:], s_phi, 'r--')
    axes[0, i].plot(t[i_start:], -s_phi, 'r--')
    axes[0, i].set_title(f'$\\phi_{i+1}$')
    axes[0, i].grid(alpha=0.3)
    s_b = 2 * sig[3+i, i_start:]
    axes[1, i].plot(t[i_start:], beta_err[i, i_start:], 'b', linewidth=0.5)
    axes[1, i].plot(t[i_start:], s_b, 'r--')
    axes[1, i].plot(t[i_start:], -s_b, 'r--')
    axes[1, i].set_title(f'$\\beta_{i+1}$')
    axes[1, i].grid(alpha=0.3)
plt.suptitle('HW3 setup with pure ZOH + PSD-derived V (no trapezoidal)')
plt.tight_layout()
plt.savefig('hw3_zoh_psdV.png', dpi=110)
print('saved hw3_zoh_psdV.png')

# Stats
inside = []
for axis in range(6):
    err = phi_err[axis, i_start:] if axis < 3 else beta_err[axis-3, i_start:]
    s = sig[axis, i_start:]
    inside.append(np.mean(np.abs(err) < 2 * s))
print('Frac inside +/-2sig:')
for axis, name in enumerate(['phi_x', 'phi_y', 'phi_z', 'beta_x', 'beta_y', 'beta_z']):
    print(f'  {name}: {inside[axis]:.2f}')
