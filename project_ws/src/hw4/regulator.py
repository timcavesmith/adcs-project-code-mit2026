# Quaternion PD regulator with inverse-dynamics feedforward, gains by pole
# placement on the Lecture 16 linearization.

import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hw3.utils import L, T, logq
from hw4.actuators import wheel_command, saturate_wheel_torque


def attitude_error(q_d, q_hat):
    """Full-angle phi = 2 H^T log(L(q_d)^T q_hat) with shortest-path sign flip.
    The factor of 2 converts hw3.utils.logq (half-angle tangent) into the
    full-angle vector that satisfies phi_dot ~= omega in the linearisation."""
    q_e = L(T @ q_d) @ q_hat
    if q_e[0] < 0:
        q_e = -q_e
    return 2.0 * logq(q_e)


def pd_gains(J, omega_n, zeta=1.0):
    """Pole placement on the linearised double integrator: K_p = J wn^2,
    K_d = 2 zeta wn J."""
    return (omega_n**2) * J, (2.0 * zeta * omega_n) * J


def regulator_torque(q_d, q_hat, omega_hat, rho_hat, J, K_p, K_d):
    """Body-frame torque command (Nm) including the inverse-dynamics term."""
    phi = attitude_error(q_d, q_hat)
    return -K_p @ phi - K_d @ omega_hat + np.cross(omega_hat, J @ omega_hat + rho_hat)


def regulator_command(q_d, q_hat, omega_hat, rho_hat, J, K_p, K_d, saturate=True):
    """Full controller -> wheel torque (4,) and the body torque it implements."""
    tau = regulator_torque(q_d, q_hat, omega_hat, rho_hat, J, K_p, K_d)
    u_W = wheel_command(tau)
    if saturate:
        u_W = saturate_wheel_torque(u_W)
    return u_W, tau


if __name__ == "__main__":
    from hw1.attitude_dynamics import J as J_dc
    K_p, K_d = pd_gains(J_dc, omega_n=5e-3, zeta=1.0)
    print("K_p diag:", np.diag(K_p))
    print("K_d diag:", np.diag(K_d))
