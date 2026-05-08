# Coupled orbit + gyrostat truth model for HW4 parts 3 and 4.
#   x = [r(3), v(3), q(4), omega(3), rho(3)]   in R^16
# Orbit: two-body Kepler (no perturbations -- the orbit only matters for the
# disturbance arguments). Attitude: gyrostat from Lecture 16.
#   q_dot     = 1/2 G(q) omega
#   omega_dot = -J^-1 ( omega x (J omega + rho) + rho_dot - tau_ext )
#   rho_dot   = B_W u_W

import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hw1.attitude_dynamics import G, J as J_dreamchaser
from hw3.utils import hat
from hw4.disturbances import (gravity_gradient_torque, drag_torque,
                               DEFAULT_SURFACES, MU_EARTH)
from hw4.actuators import B_W


def dynamics_full(x, u_W, J=J_dreamchaser, surfaces=DEFAULT_SURFACES,
                  use_grav=True, use_drag=True):
    """Right-hand side. u_W is the wheel torque command (4,)."""
    r, v, q, omega, rho = x[0:3], x[3:6], x[6:10], x[10:13], x[13:16]

    rdot = v
    vdot = -MU_EARTH / np.linalg.norm(r)**3 * r

    tau_ext = np.zeros(3)
    if use_grav:
        tau_ext += gravity_gradient_torque(r, q, J)
    if use_drag:
        tau_ext += drag_torque(r, v, q, surfaces)

    rho_dot = B_W @ u_W
    qdot = 0.5 * G(q) @ omega
    omega_dot = np.linalg.solve(
        J, -hat(omega) @ (J @ omega + rho) - rho_dot + tau_ext)

    return np.concatenate([rdot, vdot, qdot, omega_dot, rho_dot])


def rk4_step(x, u_W, dt, **kwargs):
    """RK4 with zero-order hold on u_W and a quaternion renormalisation."""
    k1 = dynamics_full(x, u_W, **kwargs)
    k2 = dynamics_full(x + 0.5 * dt * k1, u_W, **kwargs)
    k3 = dynamics_full(x + 0.5 * dt * k2, u_W, **kwargs)
    k4 = dynamics_full(x + dt * k3, u_W, **kwargs)
    xn = x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    xn[6:10] /= np.linalg.norm(xn[6:10])
    return xn


def make_state(r, v, q, omega, rho=None):
    if rho is None:
        rho = np.zeros(3)
    return np.concatenate([r, v, q, omega, rho])


def coe2rv(a, e, i, RAAN, omega, nu, mu=MU_EARTH):
    """Classical orbital elements to ECI (r in km, v in km/s). Inlined here so
    that the truth model does not pull matplotlib/plotly through hw1."""
    p = a * (1 - e**2)
    r_pqw = p / (1 + e * np.cos(nu)) * np.array([np.cos(nu), np.sin(nu), 0.0])
    v_pqw = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])
    cO, sO = np.cos(RAAN), np.sin(RAAN)
    ci, si = np.cos(i), np.sin(i)
    cw, sw = np.cos(omega), np.sin(omega)
    R = np.array([
        [cO*cw - sO*sw*ci,  -cO*sw - sO*cw*ci,   sO*si],
        [sO*cw + cO*sw*ci,  -sO*sw + cO*cw*ci,  -cO*si],
        [sw*si,              cw*si,                ci ],
    ])
    return R @ r_pqw, R @ v_pqw


if __name__ == "__main__":
    a, e, inc = 6800.0, 0.001, np.radians(51.64)
    r0, v0 = coe2rv(a, e, inc, np.radians(30), np.radians(60), 0.0)
    x = make_state(r0, v0, np.array([1.0, 0, 0, 0]),
                   np.array([0.001, -0.002, 0.001]))
    for _ in range(60):
        x = rk4_step(x, np.zeros(4), 1.0)
    print("After 60 s, omega:", x[10:13], " |q|:", np.linalg.norm(x[6:10]))
