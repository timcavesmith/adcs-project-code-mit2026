import numpy as np
from utils import hat, L, Q, expq


def sample_M(sigma_align_rad=0.0, sigma_scale=0.0, rng=None):
    """Sample an affine-error matrix M = (I + diag(s)) exp(hat(alpha)).
    alpha is small misalignment (rad) and s is per-axis scale error.
    Pass sigma=0 to disable a term."""
    if rng is None:
        rng = np.random.default_rng()
    alpha = rng.normal(0, sigma_align_rad, 3)
    s = rng.normal(0, sigma_scale, 3)
    th = np.linalg.norm(alpha)
    if th < 1e-12:
        M_align = np.eye(3)
    else:
        u = alpha / th
        K = hat(u)
        M_align = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
    return (np.eye(3) + np.diag(s)) @ M_align


class bearing_sensor:
    """y = M Q(q_true)^T r_inertial + b + w, w ~ N(0, W).
    Q(q) is body-to-inertial, so Q(q)^T r drops an inertial reference vector into
    the body frame. M and b model fixed scale/misalignment and bias errors."""
    def __init__(self, r_inertial, W, M=None, b=None, rng=None):
        self.r_inertial = np.asarray(r_inertial, dtype=float).reshape(3)
        self.W = np.asarray(W, dtype=float)
        self.M = np.eye(3) if M is None else np.asarray(M, dtype=float)
        self.b = np.zeros(3) if b is None else np.asarray(b, dtype=float).reshape(3)
        self.rng = rng if rng is not None else np.random.default_rng()

    def getMsmt(self, q_true):
        y_ideal = Q(q_true).T @ self.r_inertial
        w = self.rng.multivariate_normal(np.zeros(3), self.W)
        return self.M @ y_ideal + self.b + w


class star_tracker:
    """Lecture 12 form: q_meas = L(q_true) expq(w), w ~ N(0, W).
    expq takes a half-angle vector, so W is the half-angle covariance
    (data sheet full-angle variance / 4)."""
    def __init__(self, W, rng=None):
        self.W = np.asarray(W, dtype=float)
        self.rng = rng if rng is not None else np.random.default_rng()

    def getMsmt(self, q_true):
        w = self.rng.multivariate_normal(np.zeros(3), self.W)
        return L(q_true) @ expq(w)


class gyro:
    """y = M omega_true + beta(t) + w, with random-walk bias beta_{k+1} = beta_k + v.
    W_PSD has units rad^2/Hz (per-sample variance W_PSD / dt).
    V_bias_PSD has units rad^2/s (per-step variance V_bias_PSD * dt)."""
    def __init__(self, dt, W_PSD, V_bias_PSD, M=None, b0=None, rng=None):
        self.dt = dt
        self.M = np.eye(3) if M is None else np.asarray(M, dtype=float)
        self.W_disc = np.asarray(W_PSD, dtype=float) / dt
        self.V_disc = np.asarray(V_bias_PSD, dtype=float) * dt
        self.b = np.zeros(3) if b0 is None else np.asarray(b0, dtype=float).reshape(3).copy()
        self.rng = rng if rng is not None else np.random.default_rng()

    def getMsmt(self, omega_true):
        w = self.rng.multivariate_normal(np.zeros(3), self.W_disc)
        self.b = self.b + self.rng.multivariate_normal(np.zeros(3), self.V_disc)
        return self.M @ omega_true + self.b + w
