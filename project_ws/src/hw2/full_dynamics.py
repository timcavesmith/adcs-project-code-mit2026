# HW2 Part 2: orbit + attitude (gyrostat), quat and solar pointing error vs time

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from orbital_sim import mu, coe2rv
from attitude_dynamics import G, quat_to_rotmat
from safe_mode import J, h_r, omega_des, pert_omega, n_sun_body, pointing_error_deg


def dynamics_full(x):
    """State x = [r(3), v(3), q(4), omega(3)]. Orbit + gyrostat."""
    r = x[0:3]
    v = x[3:6]
    q = x[6:10]
    omega = x[10:13]

    r_norm = np.linalg.norm(r)
    rdot = v
    vdot = -mu / r_norm**3 * r

    qdot = 0.5 * G(q) @ omega
    h_total = J @ omega + h_r
    omegadot = np.linalg.solve(J, -np.cross(omega, h_total))

    return np.concatenate([rdot, vdot, qdot, omegadot])


def rk4step_full(x, h):
    k1 = dynamics_full(x)
    k2 = dynamics_full(x + k1 * h / 2)
    k3 = dynamics_full(x + k2 * h / 2)
    k4 = dynamics_full(x + h * k3)
    xn = x + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    xn[6:10] /= np.linalg.norm(xn[6:10])
    return xn


# same orbit as HW1
a, e, i = 8000.0, 0.1, np.radians(45)
RAAN, omega_coe, nu0 = np.radians(30), np.radians(60), 0.0
r0, v0 = coe2rv(a, e, i, RAAN, omega_coe, nu0)

q0 = np.array([1.0, 0, 0, 0])
omega0 = omega_des + pert_omega

x0 = np.concatenate([r0, v0, q0, omega0])

# run for several nutation periods (nutation period ~ tens of sec for 10 RPM)
h_step = 0.05
tf = 300.0
n_steps = int(tf / h_step) + 1

xhist = np.zeros((13, n_steps))
xhist[:, 0] = x0
for k in range(n_steps - 1):
    xhist[:, k + 1] = rk4step_full(xhist[:, k], h_step)

t = np.linspace(0, tf, n_steps)
# q is rows 0:4 in what we pass (pointing_error_deg uses xhist[0:4,k])
err_deg = pointing_error_deg(xhist[6:10, :])

out_dir = os.path.join(os.path.dirname(__file__), '..')

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(t, xhist[6, :], label='q0')
axes[0].plot(t, xhist[7, :], label='q1')
axes[0].plot(t, xhist[8, :], label='q2')
axes[0].plot(t, xhist[9, :], label='q3')
axes[0].set_ylabel('Quaternion')
axes[0].legend(loc='upper right', ncol=4, fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, err_deg)
axes[1].set_ylabel('Solar panel pointing error [deg]')
axes[1].set_xlabel('Time [s]')
axes[1].grid(True, alpha=0.3)
plt.suptitle('HW2.2 Full dynamics: quat and pointing error (sun along x ECI)')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'hw2_full_dynamics.png'), dpi=150, bbox_inches='tight')
plt.show()

if __name__ == "__main__":
    print('Full dynamics: orbit + gyrostat, tf =', tf, 's')
    print('Pointing error at t=0:', err_deg[0], 'deg')
    print('Pointing error at t=tf:', err_deg[-1], 'deg')
