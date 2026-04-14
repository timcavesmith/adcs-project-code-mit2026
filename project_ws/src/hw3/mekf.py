# mekf.py - MEKF for attitude + gyro bias (Lectures 11-12)

import numpy as np
from utils import hat, H, T, L, R, G, Q, expq, logq


def predict(q, beta, gyro, dt):
    omega = gyro - beta
    dq = expq(0.5 * dt * omega)
    qn = L(q) @ dq
    qn /= np.linalg.norm(qn)
    return qn, beta.copy()


def predict_jac(q, beta, gyro, dt):
    omega = gyro - beta
    dq = expq(0.5 * dt * omega)
    qn = L(q) @ dq
    A11 = G(qn).T @ R(dq) @ G(q)
    A = np.block([[A11,              -dt * np.eye(3)],
                  [np.zeros((3, 3)),  np.eye(3)     ]])
    return A


def bearing_predict(q, r_N):
    # r_N is (3,m)
    return (Q(q).T @ r_N).ravel(order='F')


def bearing_jac(q, r_N):
    m = r_N.shape[1]
    C = np.zeros((3 * m, 6))
    for k in range(m):
        rk = H @ r_N[:, k]
        C[3*k:3*k+3, 0:3] = H.T @ (L(q).T @ L(rk) + R(q) @ R(rk) @ T) @ G(q)
    return C


def star_tracker_innov(q_pred, q_meas):
    if q_pred @ q_meas < 0:
        q_meas = -q_meas
    q_err = L(T @ q_pred) @ q_meas
    if q_err[0] < 0:
        q_err = -q_err
    return logq(q_err)


def star_tracker_jac():
    return np.hstack([np.eye(3), np.zeros((3, 3))])


def quat_err(q_est, q_true):
    q_e = L(T @ q_est) @ q_true
    if q_e[0] < 0:
        q_e = -q_e
    return logq(q_e)
