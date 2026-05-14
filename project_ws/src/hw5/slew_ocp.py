# OCP-based slew reference. Solve the NLP from a versine warm-start, then
# resample the result onto the closed-loop dt grid so the HW4 PD tracker can
# fly it. The OCP outputs the per-wheel torque u directly, so we feed that to
# the controller as feed-forward (no B_W^+ projection that would drop the
# null-space part).

import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hw4.actuators import B_W, B_W_pinv, saturate_wheel_torque
from hw4.slew import reference_trajectory
from hw4.regulator import attitude_error
from hw5.ocp import solve_ocp


def _versine_warm_start(q0, q_f, J, T_man, N):
    """Sample the HW4 versine reference at N+1 nodes and assemble (q, omega,
    w, u) initial guesses for the collocation grid."""
    dt = T_man / max(N * 4, 100)
    traj = reference_trajectory(q0, q_f, T_man=T_man, dt=dt, J=J)
    t_nodes = np.linspace(0.0, T_man, N + 1)

    def interp(arr2d):
        return np.array([np.interp(t_nodes, traj['t'], arr2d[i, :])
                         for i in range(arr2d.shape[0])])

    q_g = interp(traj['q_ref']); q_g /= np.linalg.norm(q_g, axis=0, keepdims=True)
    om_g = interp(traj['omega_ref'])
    rho_g = interp(traj['rho_ref'])
    rho_dot_g = interp(traj['rho_dot_ref'])
    # rho = B_W w  =>  w = B_W^+ rho (min-norm guess; null space starts at 0)
    w_g = B_W_pinv @ rho_g
    u_g = B_W_pinv @ rho_dot_g
    return q_g, om_g, w_g, u_g


def build_reference(q0, q_f, J,
                    u_max, w_max, omega_max,
                    dt_sample=1.0, N=40, w_t=1.0,
                    R_diag=None, Q_diag=None,
                    T_man_warm=2500.0, t_f_min=300.0, t_f_max=3000.0,
                    verbose=False):
    """Solve the OCP, resample onto a uniform dt_sample grid for the closed
    loop, and pack into the same dict shape that hw4.slew.slew_command uses
    (plus 'u_W_ff' carrying the OCP per-wheel torques)."""
    q_g, om_g, w_g, u_g = _versine_warm_start(q0, q_f, J, T_man_warm, N)

    sol = solve_ocp(q0, q_f, J, B_W,
                    u_max=u_max, w_max=w_max, omega_max=omega_max,
                    N=N, w_t=w_t, R_diag=R_diag, Q_diag=Q_diag,
                    t_f_init=T_man_warm, t_f_min=t_f_min, t_f_max=t_f_max,
                    q_init=q_g, omega_init=om_g, w_init=w_g, u_init=u_g,
                    verbose=verbose)

    t_grid = np.arange(0.0, sol['t_f'], dt_sample)
    t_grid = np.append(t_grid, sol['t_f'])

    def interp(arr2d):
        return np.array([np.interp(t_grid, sol['t'], arr2d[i, :])
                         for i in range(arr2d.shape[0])])

    q_ref = interp(sol['q']); q_ref /= np.linalg.norm(q_ref, axis=0, keepdims=True)
    omega_ref = interp(sol['omega'])
    w_ref = interp(sol['w'])
    u_ref = interp(sol['u'])
    rho_ref = B_W @ w_ref
    rho_dot_ref = B_W @ u_ref
    omega_dot_ref = np.zeros_like(omega_ref)
    omega_dot_ref[:, :-1] = np.diff(omega_ref, axis=1) / dt_sample
    omega_dot_ref[:, -1] = omega_dot_ref[:, -2]

    return dict(t=t_grid, q_ref=q_ref, omega_ref=omega_ref,
                omega_dot_ref=omega_dot_ref, rho_ref=rho_ref,
                rho_dot_ref=rho_dot_ref, w_ref=w_ref, u_W_ff=u_ref,
                sol=sol)


def slew_command_ocp(traj, k, q_hat, omega_hat, K_p, K_d, saturate=False):
    """Same form as hw4.slew.slew_command but uses the OCP per-wheel u as the
    feed-forward, keeping its null-space (no-net-torque) component intact."""
    q_ref = traj['q_ref'][:, k]
    omega_ref = traj['omega_ref'][:, k]

    u_W_ff = traj['u_W_ff'][:, k]
    delta_phi = attitude_error(q_ref, q_hat)
    delta_omega = omega_hat - omega_ref
    tau_fb = -K_p @ delta_phi - K_d @ delta_omega
    u_W = u_W_ff - B_W_pinv @ tau_fb

    if saturate:
        u_W = saturate_wheel_torque(u_W)
    return u_W, tau_fb, u_W_ff


if __name__ == "__main__":
    import time
    from hw3.utils import expq
    from hw1.attitude_dynamics import J as J_dc
    from hw4.actuators import WHEEL_TORQUE_MAX, WHEEL_MOMENTUM_MAX

    axis = np.array([1.0, 0.5, -0.3]); axis /= np.linalg.norm(axis)
    q0 = np.array([1.0, 0, 0, 0])
    qf = expq(0.5 * np.pi * axis)
    t0 = time.time()
    traj = build_reference(q0, qf, J_dc,
                           u_max=WHEEL_TORQUE_MAX,
                           w_max=WHEEL_MOMENTUM_MAX,
                           omega_max=np.radians(0.3),
                           N=40, w_t=1.0,
                           R_diag=20.0 * np.ones(4),
                           Q_diag=1e-5 * np.ones(4),
                           verbose=False)
    print(f"solve took {time.time() - t0:.2f} s")
    sol = traj['sol']
    print(f"t_f* = {sol['t_f']:.2f} s  (warm: 2500)")
    print(f"cost = {sol['cost']:.3f}")
    print(f"peak |u|   = {np.max(np.abs(sol['u'])):.4f}  (limit {WHEEL_TORQUE_MAX})")
    print(f"peak |w|   = {np.max(np.abs(sol['w'])):.3f}   (limit {WHEEL_MOMENTUM_MAX})")
    print(f"peak |om|  = "
          f"{np.degrees(np.max(np.linalg.norm(sol['omega'], axis=0))):.4f} deg/s")
    print(f"q_term err = "
          f"{np.degrees(np.linalg.norm(attitude_error(qf, sol['q'][:, -1]))):.4e} deg")
