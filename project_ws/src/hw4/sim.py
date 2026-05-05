"""Orbit plus attitude propagation for the HW4 rigid-body simulation.

The module integrates a 13-state vector:
    x = [r(3), v(3), q(4), omega(3)]

The attitude dynamics include gravity-gradient torque using the helper in
``environmental_torques.py``.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from environmental_torques import getGravityGradientTorque, getDragTorque
from utils import *
from surfaces import SURFACE_CENTROIDS, SURFACE_NORMALS, SURFACE_AREAS

# Earth's gravitational parameter [km^3/s^2]
mu = 398600.44

# Dreamchaser inertia [kg km^2 in this project's units]
J = np.array(
    [[12685.21, 0.0, -1358.05],
     [0.0, 51407.14, 0.0],
     [-1358.05, 0.0, 57999.34]]
)

DEFAULT_ORBIT = {
    "a": 6800.0,
    "e": 0.001,
    "i": np.radians(51.64),
    "raan": np.radians(30.0),
    "argp": np.radians(60.0),
    "nu": 0.0,
}


def initial_state(q0=None, omega0=None, orbit=None):
    """Build the initial 13-state vector for the coupled orbit/attitude sim."""
    orbit = DEFAULT_ORBIT if orbit is None else orbit
    r0, v0 = coe2rv(
        orbit["a"],
        orbit["e"],
        orbit["i"],
        orbit["raan"],
        orbit["argp"],
        orbit["nu"],
        mu=mu,
    )

    q0 = quat_normalize([1.0, 0.0, 0.0, 0.0] if q0 is None else q0)
    if omega0 is None:
        omega0 = np.array([0.05, 0.01, 0.01])
    omega0 = np.asarray(omega0, dtype=float).reshape(3,)

    return np.concatenate([r0, v0, q0, omega0])


def compute_external_torques(x, use_gravity_gradient_torque=True, use_drag_torque=True):
    """Compute gravity-gradient and drag torques for the current state."""
    r = x[0:3]
    v = x[3:6]
    q = x[6:10]

    if use_gravity_gradient_torque:
        r_body = Q(q).T @ r
        gravity_gradient_torque = getGravityGradientTorque(mu, J, r_body)
    else:
        gravity_gradient_torque = np.zeros(3)

    if use_drag_torque:
        drag_torque = getDragTorque(
            Cd=2.2,
            v_eci=v,
            r_eci=r,
            q=q,
            surface_positions=SURFACE_CENTROIDS,
            surface_normals=SURFACE_NORMALS,
            surface_areas=SURFACE_AREAS,
        )
    else:
        drag_torque = np.zeros(3)

    return gravity_gradient_torque, drag_torque

def dynamics(x, use_gravity_gradient_torque=True, use_drag_torque=True):
    """Compute the time derivative of x = [r(3), v(3), q(4), omega(3)]."""
    r = x[0:3]
    v = x[3:6]
    q = x[6:10]
    omega = x[10:13]

    r_norm = np.linalg.norm(r)
    if r_norm == 0.0:
        raise ValueError("Position norm must be non-zero.")

    rdot = v
    vdot = -mu * r / r_norm**3

    qdot = 0.5 * G(q) @ omega
    gravity_gradient_torque, drag_torque = compute_external_torques(
        x,
        use_gravity_gradient_torque=use_gravity_gradient_torque,
        use_drag_torque=use_drag_torque,
    )
    external_torque = gravity_gradient_torque + drag_torque
    omegadot = np.linalg.solve(J, external_torque - np.cross(omega, J @ omega))

    return np.concatenate([rdot, vdot, qdot, omegadot])


def rk4step_full(x, dt, use_gravity_gradient_torque=True, use_drag_torque=True, return_torques=False):
    """Advance the coupled state one fixed RK4 step."""
    k1 = dynamics(x, use_gravity_gradient_torque=use_gravity_gradient_torque, use_drag_torque=use_drag_torque)
    k2 = dynamics(x + 0.5 * dt * k1, use_gravity_gradient_torque=use_gravity_gradient_torque, use_drag_torque=use_drag_torque)
    k3 = dynamics(x + 0.5 * dt * k2, use_gravity_gradient_torque=use_gravity_gradient_torque, use_drag_torque=use_drag_torque)
    k4 = dynamics(x + dt * k3, use_gravity_gradient_torque=use_gravity_gradient_torque, use_drag_torque=use_drag_torque)

    xn = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    xn[6:10] = quat_normalize(xn[6:10])

    if not return_torques:
        return xn

    gravity_gradient_torque, drag_torque = compute_external_torques(
        x,
        use_gravity_gradient_torque=use_gravity_gradient_torque,
        use_drag_torque=use_drag_torque,
    )
    return xn, gravity_gradient_torque, drag_torque


def run_sim(
    rate=10,
    tf=60.0,
    q0_true=None,
    omega0=None,
    seed=42,
    use_gravity_gradient_torque=True,
    use_drag_torque=True,
    **_ignored,
):
    """Run a coupled orbit/attitude simulation.

    Args:
        rate: integration rate in Hz.
        tf: final simulation time in seconds.
        q0_true: optional initial quaternion, kept for backward compatibility.
        omega0: optional initial body rate in rad/s.
        seed: retained for compatibility; the current simulation is deterministic.
        use_gravity_gradient_torque: include gravity-gradient torque when True.
        use_drag_torque: include aerodynamic drag torque when True.
        **_ignored: absorbs obsolete keyword arguments from older scripts.

    Returns:
        Dictionary containing time history, state history, and the initial state.
    """
    _ = seed
    dt = 1.0 / float(rate)
    n_steps = int(np.floor(tf / dt)) + 1
    time = np.linspace(0.0, dt * (n_steps - 1), n_steps)

    x0 = initial_state(q0=q0_true, omega0=omega0)
    xtraj = np.zeros((13, n_steps))
    gravity_gradient_torque_traj = np.zeros((3, n_steps))
    drag_torque_traj = np.zeros((3, n_steps))
    xtraj[:, 0] = x0

    gravity_gradient_torque_traj[:, 0], drag_torque_traj[:, 0] = compute_external_torques(
        x0,
        use_gravity_gradient_torque=use_gravity_gradient_torque,
        use_drag_torque=use_drag_torque,
    )

    for k in range(n_steps - 1):
        xtraj[:, k + 1], gravity_gradient_torque_traj[:, k + 1], drag_torque_traj[:, k + 1] = rk4step_full(
            xtraj[:, k],
            dt,
            use_gravity_gradient_torque=use_gravity_gradient_torque,
            use_drag_torque=use_drag_torque,
            return_torques=True,
        )

    return {
        "time": time,
        "xtraj": xtraj,
        "gravity_gradient_torque_traj": gravity_gradient_torque_traj,
        "drag_torque_traj": drag_torque_traj,
        "x0": x0,
        "dt": dt,
        "use_gravity_gradient_torque": use_gravity_gradient_torque,
        "use_drag_torque": use_drag_torque,
    }


def compare_torque_cases(rate=10, num_orbits=3, q0_true=None, omega0=None, seed=42):
    """Run the simulation with and without environmental torque using the same initial state."""

    # calculate tf given number of orbits
    a = DEFAULT_ORBIT["a"]
    T_orbit = 2.0 * np.pi * np.sqrt(a**3 / mu)
    tf = num_orbits * T_orbit

    result_with_torque = run_sim(
        rate=rate,
        tf=tf,
        q0_true=q0_true,
        omega0=omega0,
        seed=seed,
        use_gravity_gradient_torque=True,
        use_drag_torque=True,
    )
    result_without_torque = run_sim(
        rate=rate,
        tf=tf,
        q0_true=q0_true,
        omega0=omega0,
        seed=seed,
        use_gravity_gradient_torque=False,
        use_drag_torque=False,
    )

    x_with = result_with_torque["xtraj"]
    x_without = result_without_torque["xtraj"]
    delta = x_with - x_without

    # quaternion helpers (w, x, y, z) ordering in this module is [q0, q1, q2, q3]
    def quat_conj(q):
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def quat_mul(q1, q2):
        s1 = q1[0]; v1 = q1[1:4]
        s2 = q2[0]; v2 = q2[1:4]
        s = s1 * s2 - np.dot(v1, v2)
        v = s1 * v2 + s2 * v1 + np.cross(v1, v2)
        return np.concatenate([[s], v])

    # compute axis-angle (3-parameter) representation of q_with * conj(q_without)
    n_steps = x_with.shape[1]
    delta_axis_angle = np.zeros((3, n_steps))
    for k in range(n_steps):
        q_w = x_with[6:10, k]
        q_wo = x_without[6:10, k]
        qd = quat_mul(q_w, quat_conj(q_wo))
        delta_axis_angle[:, k] = logq(qd)

    gravity_gradient_with = result_with_torque["gravity_gradient_torque_traj"]
    drag_with = result_with_torque["drag_torque_traj"]
    gravity_gradient_without = result_without_torque["gravity_gradient_torque_traj"]
    drag_without = result_without_torque["drag_torque_traj"]

    comparison = {
        "with_torque": result_with_torque,
        "without_torque": result_without_torque,
        "delta": delta,
        "delta_axis_angle": delta_axis_angle,
        "gravity_gradient_torque": {
            "with": gravity_gradient_with,
            "without": gravity_gradient_without,
        },
        "drag_torque": {
            "with": drag_with,
            "without": drag_without,
        },
        "final_quaternion_error_norm": np.linalg.norm(delta[6:10, -1]),
        "final_body_rate_error_norm": np.linalg.norm(delta[10:13, -1]),
    }
    return comparison


def plot_simulation(result):
    """Plot the quaternion and body-rate histories from a simulation result."""
    time = result["time"]
    xtraj = result["xtraj"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for idx, label in enumerate(["q0", "q1", "q2", "q3"]):
        axes[0].plot(time, xtraj[6 + idx, :], label=label)
    axes[0].set_ylabel("Quaternion")
    axes[0].legend(loc="upper right", ncol=4, fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for idx, label in enumerate(["wx", "wy", "wz"]):
        axes[1].plot(time, xtraj[10 + idx, :] * 180.0 / np.pi, label=label)
    axes[1].set_ylabel("Body rate [rad/s]")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend(loc="upper right", ncol=3, fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, axes


def plot_full_comparison(comparison):
    """Produce the three requested plots:
    1) position components x,y,z
    2) quaternion histories with and without torque
    3) axis-angle 3-parameter components of the quaternion difference
    4) body-rate histories with and without torque
    5) torque magnitudes for gravity-gradient and drag
    """
    result_with = comparison["with_torque"]
    result_without = comparison["without_torque"]
    time = result_with["time"]
    x_with = result_with["xtraj"]
    x_without = result_without["xtraj"]

    delta_axis = comparison.get("delta_axis_angle")
    gravity_gradient_torque = comparison.get("gravity_gradient_torque")
    drag_torque = comparison.get("drag_torque")

    fig, axes = plt.subplots(6, 1, figsize=(11, 16), sharex=True)

    # Positions (r_x, r_y, r_z)
    axes[0].plot(time, x_with[0, :], label="$r_x$")
    axes[0].plot(time, x_with[1, :], label="$r_y$")
    axes[0].plot(time, x_with[2, :], label="$r_z$")
    axes[0].set_ylabel("Position [km]")
    axes[0].legend(loc="upper left", ncol=3, fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Quaternions: with vs without torque
    # Ensure the with/without lines share the same color
    prop_cycle = plt.rcParams.get("axes.prop_cycle")
    colors = prop_cycle.by_key().get("color", []) if prop_cycle is not None else []
    ncolors = len(colors)
    for idx, label in enumerate(["$q_0$", "$q_1$", "$q_2$", "$q_3$"]):
        color = colors[idx % ncolors] if ncolors > 0 else None
        axes[1].plot(time, x_with[6 + idx, :], label=f"{label} with torques", linestyle="-", color=color)
        axes[1].plot(time, x_without[6 + idx, :], label=f"{label} without torques", linestyle="--", color=color)
    axes[1].set_ylabel("Quaternion")
    axes[1].legend(loc="upper left", ncol=4, fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Axis-angle 3-parameter components (delta)
    if delta_axis is not None:
        axes[2].plot(time, np.linalg.norm(delta_axis, axis=0) * 180.0 / np.pi, label="$|\\theta|$")
        axes[2].plot(time, delta_axis[0, :] * 180.0 / np.pi, label="$\\theta_x$")
        axes[2].plot(time, delta_axis[1, :] * 180.0 / np.pi, label="$\\theta_y$")
        axes[2].plot(time, delta_axis[2, :] * 180.0 / np.pi, label="$\\theta_z$")
        axes[2].set_ylabel("Axis-angle attitude diff [deg]")
        axes[2].legend(loc="upper left", ncol=3, fontsize=8)
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, "No axis-angle data available", ha="center", va="center")
        axes[2].grid(False)

    # Body rates: with vs without torque
    for idx, label in enumerate(["$\\omega_x$", "$\\omega_y$", "$\\omega_z$"]):
        color = colors[idx % ncolors] if ncolors > 0 else None
        axes[3].plot(time, x_with[10 + idx, :] * 180.0 / np.pi, label=f"{label} with torques", linestyle="-", color=color)
        axes[3].plot(time, x_without[10 + idx, :] * 180.0 / np.pi, label=f"{label} without torques", linestyle="--", color=color)
    axes[3].set_ylabel("Body rate [deg/s]")
    axes[3].legend(loc="upper left", ncol=3, fontsize=8)
    axes[3].grid(True, alpha=0.3)
    # Also plot magnitude of body rate for each case (deg/s)
    omega_mag_with = np.linalg.norm(result_with["xtraj"][10:13, :], axis=0) * 180.0 / np.pi
    omega_mag_without = np.linalg.norm(result_without["xtraj"][10:13, :], axis=0) * 180.0 / np.pi
    axes[3].plot(time, omega_mag_with, label="|ω| with", linestyle="-", color="k", linewidth=1.2)
    axes[3].plot(time, omega_mag_without, label="|ω| without", linestyle="--", color="0.5", linewidth=1.0)
    axes[3].legend(loc="upper left", ncol=3, fontsize=8)

    # Gravity-gradient torque magnitude
    if gravity_gradient_torque is not None:
        gg_with = np.linalg.norm(gravity_gradient_torque["with"], axis=0)
        gg_without = np.linalg.norm(gravity_gradient_torque["without"], axis=0)
        # plot magnitude for both cases
        axes[4].plot(time, gg_with, label=r"$|\tau_{gg}|$ with", linestyle="-")
        axes[4].plot(time, gg_without, label=r"$|\tau_{gg}|$ without", linestyle="--")
        # plot per-component traces for the 'with' case only
        for ci in range(3):
            c = colors[ci % ncolors] if ncolors > 0 else None
            if ci == 0:
                label = r"$\tau_{gg x}$"
            elif ci == 1:
                label = r"$\tau_{gg y}$"
            else:                label = r"$\tau_{gg z}$"
            axes[4].plot(time, gravity_gradient_torque["with"][ci, :], label=label, linestyle=':', color=c, alpha=0.9)
        axes[4].set_ylabel(r"$\tau_{gg}$ [Nm]")
        axes[4].legend(loc="upper left", ncol=2, fontsize=8)
        axes[4].grid(True, alpha=0.3)
    else:
        axes[4].text(0.5, 0.5, "No gravity-gradient history", ha="center", va="center")
        axes[4].grid(False)

    # Aerodynamic drag torque magnitude
    if drag_torque is not None:
        drag_with = np.linalg.norm(drag_torque["with"], axis=0)
        drag_without = np.linalg.norm(drag_torque["without"], axis=0)
        # plot magnitude for both cases
        axes[5].plot(time, drag_with, label=r"$|\tau_{drag}|$ with", linestyle="-")
        axes[5].plot(time, drag_without, label=r"$|\tau_{drag}|$ without", linestyle="--")
        # plot per-component traces for the 'with' case only
        for ci in range(3):
            c = colors[ci % ncolors] if ncolors > 0 else None
            if ci == 0:
                label = r"$\tau_{drag x}$"
            elif ci == 1:
                label = r"$\tau_{drag y}$"
            else:                label = r"$\tau_{drag z}$"
            axes[5].plot(time, drag_torque["with"][ci, :], label=label, linestyle=':', color=c, alpha=0.9)
        axes[5].set_ylabel(r"$\tau_{drag}$ [Nm]")
        axes[5].legend(loc="upper left", ncol=2, fontsize=8)
        axes[5].grid(True, alpha=0.3)
    else:
        axes[5].text(0.5, 0.5, "No drag history", ha="center", va="center")
        axes[5].grid(False)

    axes[5].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig("hw4_torque_plot.pdf", format="pdf", dpi=300)
    return fig, axes


if __name__ == "__main__":
    comparison = compare_torque_cases(rate=1, num_orbits=3)
    plot_full_comparison(comparison)
    plt.show()
    print("Final quaternion with torque:", comparison["with_torque"]["xtraj"][6:10, -1])
    print("Final quaternion without torque:", comparison["without_torque"]["xtraj"][6:10, -1])
    print("Final body rates with torque:", comparison["with_torque"]["xtraj"][10:13, -1])
    print("Final body rates without torque:", comparison["without_torque"]["xtraj"][10:13, -1])
    print("Final quaternion error norm:", comparison["final_quaternion_error_norm"])
    print("Final body-rate error norm:", comparison["final_body_rate_error_norm"])
    print("Max gravity-gradient torque magnitude:", np.max(np.linalg.norm(comparison["gravity_gradient_torque"]["with"], axis=0)))
    print("Max drag torque magnitude:", np.max(np.linalg.norm(comparison["drag_torque"]["with"], axis=0)))
    gg_mean_comp = np.mean(comparison["gravity_gradient_torque"]["with"], axis=1)
    drag_mean_comp = np.mean(comparison["drag_torque"]["with"], axis=1)
    print("Average gravity-gradient torque magnitude (norm of mean components):", np.linalg.norm(gg_mean_comp))
    print("Average drag torque magnitude (norm of mean components):", np.linalg.norm(drag_mean_comp))