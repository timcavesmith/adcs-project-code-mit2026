# MEKF following Lectures 11 and 12.
# State x = [q, beta]. Error state [phi, delta_beta] with phi the half-angle vector.

import numpy as np
from utils import hat, H, T, L, R, G, Q, expq, logq


def predict(q, beta, gyro, dt):
    """One-step prediction. Bias dynamics are identity (the random walk lives in V)."""
    omega = gyro - beta
    dq = expq(0.5 * dt * omega)
    qn = L(q) @ dq
    qn /= np.linalg.norm(qn)
    return qn, beta.copy()


def predict_jac(q, beta, gyro, dt):
    """Error-state Jacobian from Lecture 12.
        d phi_{k+1} / d phi_k  =  G(qn)^T R(dq) G(q)
        d phi_{k+1} / d beta_k = -0.5 dt G(qn)^T G(q)
    Bias block is identity."""
    omega = gyro - beta
    dq = expq(0.5 * dt * omega)
    qn = L(q) @ dq
    A11 = G(qn).T @ R(dq) @ G(q)
    A12 = -0.5 * dt * G(qn).T @ G(q)
    return np.block([[A11,              A12       ],
                     [np.zeros((3, 3)),  np.eye(3)]])


def bearing_predict(q, r_N):
    """Stacked body-frame prediction y_pred = Q(q)^T r_N, flattened column-major."""
    return (Q(q).T @ r_N).ravel(order='F')


def bearing_jac(q, r_N):
    """Stacked bearing Jacobian (Lecture 11). Bias columns are zero."""
    m = r_N.shape[1]
    C = np.zeros((3 * m, 6))
    for k in range(m):
        rk = H @ r_N[:, k]
        C[3*k:3*k+3, 0:3] = H.T @ (L(q).T @ L(rk) + R(q) @ R(rk) @ T) @ G(q)
    return C


def star_innov(q_pred, q_meas):
    """Vector-part innovation z = G(q_pred)^T q_meas (Lecture 12)."""
    return G(q_pred).T @ q_meas


def star_jac(q_pred, q_meas):
    """Jacobian of the vector-part star-tracker innovation. Bias columns are zero."""
    C = np.zeros((3, 6))
    C[:, 0:3] = H.T @ R(q_meas) @ T @ G(q_pred)
    return C


def quat_err(q_est, q_true):
    """3-parameter axis-angle error vector from logq(q_est^* X q_true)."""
    q_e = L(T @ q_est) @ q_true
    if q_e[0] < 0:
        q_e = -q_e
    return logq(q_e)
