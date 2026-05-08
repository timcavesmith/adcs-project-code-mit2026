"""Direct python port of Lecture 12 mekf-bias.ipynb. Same algorithm, same
parameters. Goal: see what the lecture's filter actually produces for a
consistency plot, so we can compare against our HW3 setup."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Quaternion utilities (translated from Julia)
def hat(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

H = np.vstack([np.zeros((1, 3)), np.eye(3)])
T = np.block([[1, np.zeros((1, 3))],
              [np.zeros((3, 1)), -np.eye(3)]])

def L(q):
    s = q[0]; v = q[1:4]
    return np.block([[s, -v.reshape(1, 3)],
                     [v.reshape(3, 1), s * np.eye(3) + hat(v)]])

def R(q):
    s = q[0]; v = q[1:4]
    return np.block([[s, -v.reshape(1, 3)],
                     [v.reshape(3, 1), s * np.eye(3) - hat(v)]])

def G(q):
    return L(q) @ H

def Q(q):
    return H.T @ (R(q).T @ L(q)) @ H

def expq(phi):
    th = np.linalg.norm(phi)
    return np.concatenate([[np.cos(th)], phi * np.sinc(th / np.pi)])

def logq(q):
    th = np.arccos(np.clip(q[0], -1.0, 1.0))
    v = q[1:4]
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        return np.zeros(3)
    return th * v / nv

# --- Setup matches mekf-bias.ipynb exactly ---
J = np.diag([1.0, 1.25, 1.5])
h = 0.1                  # time step (10 Hz)
n = 600                  # 60 sec

def dynamics(x):
    q = x[0:4]
    q = q / np.linalg.norm(q)
    omega = x[4:7]
    qdot = 0.5 * G(q) @ omega
    omegadot = -np.linalg.solve(J, hat(omega) @ J @ omega)
    return np.concatenate([qdot, omegadot])

def rkstep(x):
    f1 = dynamics(x)
    f2 = dynamics(x + 0.5 * h * f1)
    f3 = dynamics(x + 0.5 * h * f2)
    f4 = dynamics(x + h * f3)
    xn = x + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
    xn[0:4] /= np.linalg.norm(xn[0:4])
    return xn

# Same RNG seed for reproducibility
rng = np.random.default_rng(42)

# Random initial attitude, angular velocity, and HUGE bias (β0 = randn(3))
q0 = rng.standard_normal(4); q0 /= np.linalg.norm(q0)
omega0 = 0.1 * rng.standard_normal(3)
beta0 = rng.standard_normal(3)         # ~ N(0, 1) bias!
x0 = np.concatenate([q0, omega0])

# Truth trajectory
xtraj = np.zeros((7, n))
xtraj[:, 0] = x0
for k in range(n - 1):
    xtraj[:, k + 1] = rkstep(xtraj[:, k])

# Lecture's noise covariances (TUNED, both 0.01)
m_obs = 2
W = 0.01 * np.eye(3 * m_obs)            # measurement noise
V = 0.01 * np.eye(6)                    # process noise

# Gyro: ω + β0 + N(0, V[4:6,4:6])
gyro = np.zeros((3, n))
for k in range(n):
    gyro[:, k] = xtraj[4:7, k] + beta0 + np.sqrt(V[3:6, 3:6]) @ rng.standard_normal(3)

# Two random inertial reference vectors
r_N = np.zeros((3, m_obs))
for k in range(m_obs):
    r = rng.standard_normal(3)
    r_N[:, k] = r / np.linalg.norm(r)

# Vector measurements: y = Q^T r with body-frame rotation noise
ytraj = np.zeros((3 * m_obs, n))
for k in range(n):
    qk = xtraj[0:4, k]
    Qk = Q(qk)
    yk = np.zeros((3, m_obs))
    w = (np.sqrt(W) @ rng.standard_normal(3 * m_obs)).reshape(3, m_obs, order='F')
    for el in range(m_obs):
        Qw = expm(hat(w[:, el]))
        yk[:, el] = Qw.T @ Qk.T @ r_N[:, el]
    ytraj[:, k] = yk.flatten(order='F')

def state_prediction(x, u, h):
    q = x[0:4]; beta = x[4:7]
    omega = u - beta
    dq = expq(0.5 * h * omega)
    return np.concatenate([L(q) @ dq, beta])

def state_prediction_deriv(x, u, h):
    q = x[0:4]; beta = x[4:7]
    omega = u - beta
    dq = expq(0.5 * h * omega)
    qn = L(q) @ dq
    A11 = G(qn).T @ R(dq) @ G(q)
    A12 = -0.5 * h * G(qn).T @ G(q)
    return np.block([[A11, A12], [np.zeros((3, 3)), np.eye(3)]])

def measurement_prediction(x, r_N):
    q = x[0:4]
    Qk = Q(q)
    return (Qk.T @ r_N).flatten(order='F')

def measurement_prediction_deriv(x, r_N):
    q = x[0:4]
    m = r_N.shape[1]
    C = np.zeros((3 * m, 6))
    for k in range(m):
        rk = H @ r_N[:, k]
        C[3 * k:3 * k + 3, 0:3] = H.T @ (L(q).T @ L(rk) + R(q) @ R(rk) @ T) @ G(q)
    return C

# Filter init: q0 + 0.1*randn perturbation, β0_est = 0, P0 = 0.5 I
xfilt = np.zeros((7, n))
xfilt[0:4, 0] = q0 + 0.1 * rng.standard_normal(4)
xfilt[0:4, 0] /= np.linalg.norm(xfilt[0:4, 0])
P = np.zeros((6, 6, n))
P[:, :, 0] = 0.5 * np.eye(6)

for k in range(n - 1):
    # ZOH prediction
    xpred = state_prediction(xfilt[:, k], gyro[:, k], h)
    A = state_prediction_deriv(xfilt[:, k], gyro[:, k], h)
    Ppred = A @ P[:, :, k] @ A.T + V

    z = ytraj[:, k + 1] - measurement_prediction(xpred, r_N)
    C = measurement_prediction_deriv(xpred, r_N)
    S = C @ Ppred @ C.T + W
    K = Ppred @ C.T @ np.linalg.inv(S)

    dx = K @ z
    phi = dx[0:3]
    s2 = phi @ phi
    if s2 < 1.0:
        dq = np.concatenate([[np.sqrt(1 - s2)], phi])
    else:
        dq = expq(phi)
    xfilt[0:4, k + 1] = L(xpred[0:4]) @ dq
    xfilt[0:4, k + 1] /= np.linalg.norm(xfilt[0:4, k + 1])
    xfilt[4:7, k + 1] = xpred[4:7] + dx[3:6]

    P[:, :, k + 1] = (np.eye(6) - K @ C) @ Ppred @ (np.eye(6) - K @ C).T + K @ W @ K.T

# Consistency: phi error and bias error vs ±2σ
phi_err = np.zeros((3, n))
b_err = np.zeros((3, n))
sig = np.zeros((6, n))
for k in range(n):
    q_e = L(np.array([xtraj[0, k], -xtraj[1, k], -xtraj[2, k], -xtraj[3, k]])) @ xfilt[0:4, k]
    if q_e[0] < 0:
        q_e = -q_e
    phi_err[:, k] = logq(q_e)
    b_err[:, k] = xfilt[4:7, k] - beta0
    sig[:, k] = np.sqrt(np.diag(P[:, :, k]))

# Plots
t = h * np.arange(n)
fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
for i in range(3):
    axes[0, i].plot(t, phi_err[i, :], 'b', linewidth=0.5)
    axes[0, i].plot(t, 2 * sig[i, :], 'r--', linewidth=0.8)
    axes[0, i].plot(t, -2 * sig[i, :], 'r--', linewidth=0.8)
    axes[0, i].set_title(f'$\\phi_{i+1}$')
    axes[0, i].grid(alpha=0.3)
    axes[1, i].plot(t, b_err[i, :], 'b', linewidth=0.5)
    axes[1, i].plot(t, 2 * sig[3 + i, :], 'r--', linewidth=0.8)
    axes[1, i].plot(t, -2 * sig[3 + i, :], 'r--', linewidth=0.8)
    axes[1, i].set_title(f'$\\beta_{i+1}$')
    axes[1, i].set_xlabel('Time [s]')
    axes[1, i].grid(alpha=0.3)
plt.suptitle('Lecture 12 mekf-bias.ipynb -- ZOH, V=0.01*I, W=0.01*I, m=2 bearings')
plt.tight_layout()
plt.savefig('lecture_mekf_bias_consistency.png', dpi=110)
print('saved lecture_mekf_bias_consistency.png')

# Stats
i = 50  # 5 sec in
inside = []
for axis in range(6):
    err = phi_err[axis, i:] if axis < 3 else b_err[axis - 3, i:]
    s = sig[axis, i:]
    inside.append(np.mean(np.abs(err) < 2 * s))
print('Frac inside +/-2sig:')
for axis, name in enumerate(['phi_x', 'phi_y', 'phi_z', 'beta_x', 'beta_y', 'beta_z']):
    print(f'  {name}: {inside[axis]:.2f}')
