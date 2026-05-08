"""Direct python port of Lecture 12 mekf-star-tracker.ipynb."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import L, R, G, Q, expq, logq, T, hat, H

J = np.diag([1.0, 1.25, 1.5])
h = 0.1; n = 600

def dynamics(x):
    q = x[0:4]/np.linalg.norm(x[0:4]); omega = x[4:7]
    return np.concatenate([0.5 * G(q) @ omega,
                           -np.linalg.solve(J, hat(omega) @ J @ omega)])

def rkstep(x):
    f1 = dynamics(x); f2 = dynamics(x + 0.5*h*f1); f3 = dynamics(x + 0.5*h*f2); f4 = dynamics(x + h*f3)
    xn = x + (h/6)*(f1 + 2*f2 + 2*f3 + f4); xn[0:4] /= np.linalg.norm(xn[0:4]); return xn

rng = np.random.default_rng(42)
q0 = rng.standard_normal(4); q0 /= np.linalg.norm(q0)
omega0 = 0.1 * rng.standard_normal(3)
beta0 = 0.3 * rng.standard_normal(3)

xtraj = np.zeros((7, n)); xtraj[:, 0] = np.concatenate([q0, omega0])
for k in range(n-1):
    xtraj[:, k+1] = rkstep(xtraj[:, k])

m_st = 2
W = 0.001 * np.eye(3 * m_st)
V = 0.01 * np.eye(6)

gyro = np.zeros((3, n))
for k in range(n):
    gyro[:, k] = xtraj[4:7, k] + beta0 + np.sqrt(V[3:6, 3:6]) @ rng.standard_normal(3)

# Star-tracker measurements: q_meas = L(q_true) expq(w)
ytraj = np.zeros((4 * m_st, n))
for k in range(n):
    qk = xtraj[0:4, k]
    yk = np.zeros((4, m_st))
    w = (np.sqrt(W) @ rng.standard_normal(3 * m_st)).reshape(3, m_st, order='F')
    for el in range(m_st):
        yk[:, el] = L(qk) @ expq(w[:, el])
    ytraj[:, k] = yk.flatten(order='F')

def state_prediction(x, u, h):
    q = x[0:4]; beta = x[4:7]; omega = u - beta
    return np.concatenate([L(q) @ expq(0.5 * h * omega), beta])

def state_prediction_deriv(x, u, h):
    q = x[0:4]; beta = x[4:7]; omega = u - beta
    dq = expq(0.5 * h * omega); qn = L(q) @ dq
    A11 = G(qn).T @ R(dq) @ G(q); A12 = -0.5 * h * G(qn).T @ G(q)
    return np.block([[A11, A12], [np.zeros((3, 3)), np.eye(3)]])

def innovation(x, y):
    q = x[0:4]; yk = y.reshape(4, m_st, order='F')
    zk = np.zeros((3, m_st))
    for k in range(m_st):
        zk[:, k] = G(q).T @ yk[:, k]
    return zk.flatten(order='F')

def innovation_deriv(x, y):
    q = x[0:4]; yk = y.reshape(4, m_st, order='F')
    C = np.zeros((3 * m_st, 6))
    for k in range(m_st):
        C[3*k:3*k+3, 0:3] = H.T @ R(yk[:, k]) @ T @ G(q)
    return C

xfilt = np.zeros((7, n))
xfilt[0:4, 0] = q0 + 0.01 * rng.standard_normal(4)
xfilt[0:4, 0] /= np.linalg.norm(xfilt[0:4, 0])
P = np.zeros((6, 6, n)); P[:, :, 0] = 0.5 * np.eye(6)

for k in range(n - 1):
    xpred = state_prediction(xfilt[:, k], gyro[:, k], h)
    A = state_prediction_deriv(xfilt[:, k], gyro[:, k], h)
    Ppred = A @ P[:, :, k] @ A.T + V
    z = innovation(xpred, ytraj[:, k+1])
    C = innovation_deriv(xpred, ytraj[:, k+1])
    S = C @ Ppred @ C.T + W
    K = Ppred @ C.T @ np.linalg.inv(S)
    dx = -K @ z  # NEGATIVE for star tracker
    phi = dx[0:3]
    s2 = phi @ phi
    dq = np.concatenate([[np.sqrt(1-s2)], phi]) if s2 < 1.0 else expq(phi)
    xfilt[0:4, k+1] = L(xpred[0:4]) @ dq
    xfilt[0:4, k+1] /= np.linalg.norm(xfilt[0:4, k+1])
    xfilt[4:7, k+1] = xpred[4:7] + dx[3:6]
    P[:, :, k+1] = (np.eye(6) - K @ C) @ Ppred @ (np.eye(6) - K @ C).T + K @ W @ K.T

# Consistency
phi_err = np.zeros((3, n)); b_err = np.zeros((3, n)); sig = np.zeros((6, n))
for k in range(n):
    qe = L(np.array([xtraj[0,k], -xtraj[1,k], -xtraj[2,k], -xtraj[3,k]])) @ xfilt[0:4, k]
    if qe[0] < 0: qe = -qe
    phi_err[:, k] = logq(qe)
    b_err[:, k] = xfilt[4:7, k] - beta0
    sig[:, k] = np.sqrt(np.diag(P[:, :, k]))

t = h * np.arange(n)
fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
for i in range(3):
    axes[0, i].plot(t, phi_err[i, :], 'b', linewidth=0.5)
    axes[0, i].plot(t, 2*sig[i, :], 'r--'); axes[0, i].plot(t, -2*sig[i, :], 'r--')
    axes[0, i].set_title(f'$\\phi_{i+1}$'); axes[0, i].grid(alpha=0.3)
    axes[1, i].plot(t, b_err[i, :], 'b', linewidth=0.5)
    axes[1, i].plot(t, 2*sig[3+i, :], 'r--'); axes[1, i].plot(t, -2*sig[3+i, :], 'r--')
    axes[1, i].set_title(f'$\\beta_{i+1}$'); axes[1, i].grid(alpha=0.3)
plt.suptitle('Lecture 12 mekf-star-tracker.ipynb -- ZOH, V=0.01*I, W=0.001*I, m=2 star trackers')
plt.tight_layout()
plt.savefig('lecture_mekf_star_consistency.png', dpi=110)
print('saved')

i = 50; inside = []
for axis in range(6):
    err = phi_err[axis, i:] if axis < 3 else b_err[axis-3, i:]
    s = sig[axis, i:]; inside.append(np.mean(np.abs(err) < 2*s))
print('Frac inside +/-2sig:')
for axis, name in enumerate(['phi_x','phi_y','phi_z','beta_x','beta_y','beta_z']):
    print(f'  {name}: {inside[axis]:.2f}')
