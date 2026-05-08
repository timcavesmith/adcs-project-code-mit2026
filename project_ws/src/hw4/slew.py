# Versine eigen-axis slew planner with inverse-dynamics feedforward, plus the
# closed-loop tracking command.

import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hw3.utils import L, T, expq, logq
from hw4.actuators import B_W_pinv, wheel_command, saturate_wheel_torque
from hw4.regulator import attitude_error


def shortest_axis_angle(q_0, q_d):
    """Full-angle vector phi_final = 2 H^T log(q_0^* q_d) with shortest-path
    sign flip; same convention as the regulator's attitude_error."""
    q_e = L(T @ q_0) @ q_d
    if q_e[0] < 0:
        q_e = -q_e
    return 2.0 * logq(q_e)


def versine_profile(t, theta_final, T_man):
    """Lecture 16 page 5 versine and its first two derivatives."""
    alpha = np.pi / T_man
    return (0.5 * theta_final * (1 - np.cos(alpha * t)),
            0.5 * theta_final * alpha * np.sin(alpha * t),
            0.5 * theta_final * alpha**2 * np.cos(alpha * t))


def reference_trajectory(q_0, q_d, T_man, dt, J, rho0=None):
    """Build the eigen-axis reference. r_hat is fixed in body frame for the
    duration of the slew. rho_ref is integrated forward Euler from the
    inverse-dynamics ODE rho_dot = -J omega_dot - omega x (J omega + rho)."""
    phi_final = shortest_axis_angle(q_0, q_d)
    theta_final = np.linalg.norm(phi_final)
    r_hat = phi_final / theta_final if theta_final > 1e-9 else np.array([1.0, 0, 0])

    n = int(round(T_man / dt)) + 1
    t = np.linspace(0, T_man, n)
    if rho0 is None:
        rho0 = np.zeros(3)

    q_ref = np.zeros((4, n))
    omega_ref = np.zeros((3, n))
    omega_dot_ref = np.zeros((3, n))
    rho_ref = np.zeros((3, n)); rho_ref[:, 0] = rho0
    rho_dot_ref = np.zeros((3, n))

    for k in range(n):
        theta, theta_dot, theta_ddot = versine_profile(t[k], theta_final, T_man)
        q_ref[:, k] = L(q_0) @ expq(0.5 * theta * r_hat)
        omega_ref[:, k] = theta_dot * r_hat
        omega_dot_ref[:, k] = theta_ddot * r_hat
        if k > 0:
            rho_ref[:, k] = rho_ref[:, k - 1] + dt * rho_dot_ref[:, k - 1]
        rho_dot_ref[:, k] = (-J @ omega_dot_ref[:, k]
                             - np.cross(omega_ref[:, k],
                                        J @ omega_ref[:, k] + rho_ref[:, k]))
    return dict(t=t, q_ref=q_ref, omega_ref=omega_ref,
                omega_dot_ref=omega_dot_ref, rho_ref=rho_ref,
                rho_dot_ref=rho_dot_ref, theta_final=theta_final, r_hat=r_hat)


def slew_command(traj, k, q_hat, omega_hat, K_p, K_d, saturate=True):
    """Closed-loop wheel command at sample k (Lecture 17 page 5):
        u_W = u_W_ff - B_W^+ (K_p delta_phi + K_d delta_omega).
    Gains K_p, K_d come from the regulator (the linearisation about the
    reference is the same double integrator)."""
    q_ref = traj['q_ref'][:, k]
    omega_ref = traj['omega_ref'][:, k]

    u_W_ff = B_W_pinv @ traj['rho_dot_ref'][:, k]

    delta_phi = attitude_error(q_ref, q_hat)
    delta_omega = omega_hat - omega_ref
    tau_fb = -K_p @ delta_phi - K_d @ delta_omega
    u_W = u_W_ff + wheel_command(tau_fb)

    if saturate:
        u_W = saturate_wheel_torque(u_W)
    return u_W, tau_fb, u_W_ff


if __name__ == "__main__":
    from hw1.attitude_dynamics import J as J_dc
    q_0 = np.array([1.0, 0, 0, 0])
    axis = np.array([1.0, 0.5, -0.3]); axis /= np.linalg.norm(axis)
    q_d = expq(0.5 * np.pi * axis)
    traj = reference_trajectory(q_0, q_d, T_man=1800.0, dt=1.0, J=J_dc)
    u_ff = B_W_pinv @ traj['rho_dot_ref']
    print(f"theta_final = {np.degrees(traj['theta_final']):.2f} deg")
    print(f"peak |omega_ref| = "
          f"{np.degrees(np.max(np.linalg.norm(traj['omega_ref'], axis=0))):.4f} deg/s")
    print(f"peak |u_W,ff|    = {np.max(np.abs(u_ff)):.4f} Nm")
    print(f"q_ref(T) - q_d   = {traj['q_ref'][:, -1] - q_d}")
