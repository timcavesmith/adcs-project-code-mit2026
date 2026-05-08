# HW4 part 4: eigen-axis slew with versine angle profile and inverse-dynamics
# feed-forward (Lecture 16), tracked with a constant-gain PD (Lecture 17).
#   1. 180 deg slew demo: nominal vs closed-loop trajectories.
#   2. Comparison vs the part-3 regulator from a 170 deg initial error.

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hw3.utils import L, expq
from hw1.attitude_dynamics import J as J_dc
from hw4.sim_driver import run_closed_loop
from hw4.dynamics import make_state, coe2rv
from hw4.regulator import pd_gains, regulator_command, attitude_error
from hw4.slew import reference_trajectory, slew_command
from hw4.actuators import WHEEL_TORQUE_MAX, B_W_pinv

OUT_DIR = os.path.dirname(__file__)

# Same gains as the regulator: linearization about the nominal trajectory is
# the same double integrator, with negligible Coriolis-rate corrections at
# our peak body rate 
OMEGA_N = 5e-3
ZETA = 1.0
K_P, K_D = pd_gains(J_dc, OMEGA_N, ZETA)


def slew_controller(traj, q_d):
    """Track traj while the plan is active, hand off to the part-3 regulator
    at the endpoint (the versine has a theta_ddot step that would inject a
    spurious feed-forward kick if we kept riding the trajectory)."""
    n_traj = traj['t'].size

    def ctrl(k, q_hat, omega_hat, rho_hat):
        if k < n_traj - 1:
            u_W, tau_fb, _ = slew_command(traj, k, q_hat, omega_hat,
                                          K_P, K_D, saturate=False)
            return u_W, dict(tau_cmd=tau_fb)
        u_W, tau = regulator_command(q_d, q_hat, omega_hat, rho_hat,
                                     J_dc, K_P, K_D, saturate=False)
        return u_W, dict(tau_cmd=tau)
    return ctrl


def regulator_only_controller(q_d):
    def ctrl(k, q_hat, omega_hat, rho_hat):
        u_W, tau = regulator_command(q_d, q_hat, omega_hat, rho_hat,
                                     J_dc, K_P, K_D, saturate=False)
        return u_W, dict(tau_cmd=tau)
    return ctrl


def _orbit_state(q_truth):
    a, e, inc = 6800.0, 0.001, np.radians(51.64)
    r0, v0 = coe2rv(a, e, inc, np.radians(30), np.radians(60), 0.0)
    return make_state(r0, v0, q_truth, np.zeros(3))


def _err_deg(q_d, q):
    return np.degrees(np.linalg.norm(attitude_error(q_d, q)))


# ---- Test 1: 180 deg slew ----

def slew_180_demo(T_man=1800.0, dt=1.0, axis=None, seed=11):
    if axis is None:
        axis = np.array([1.0, 0.5, -0.3]); axis /= np.linalg.norm(axis)
    q_0 = np.array([1.0, 0, 0, 0])
    q_d = expq(0.5 * np.pi * axis)

    traj = reference_trajectory(q_0, q_d, T_man=T_man, dt=dt, J=J_dc)
    u_ff = B_W_pinv @ traj['rho_dot_ref']
    print(f"180 deg slew about {axis}, T_man = {T_man:.0f} s")
    print(f"  peak nominal |omega_ref|: "
          f"{np.degrees(np.max(np.linalg.norm(traj['omega_ref'], axis=0))):.3f} deg/s")
    print(f"  peak nominal |u_W,ff|:    {np.max(np.abs(u_ff)):.3f} Nm "
          f"(limit {WHEEL_TORQUE_MAX})")

    # MEKF pre-converged; the regulator was holding pointing before the slew.
    res = run_closed_loop(slew_controller(traj, q_d), t_final=T_man + 600.0,
                          dt=dt, x0_truth=_orbit_state(q_0),
                          q0_est=q_0.copy(), seed=seed)

    n = res['x'].shape[1]
    delta_phi = np.zeros((3, n))
    err_to_qd = np.zeros(n)
    for k in range(n):
        kk = min(int(round(res['t'][k] / dt)), traj['t'].size - 1)
        delta_phi[:, k] = attitude_error(traj['q_ref'][:, kk], res['x'][6:10, k])
        err_to_qd[k] = _err_deg(q_d, res['x'][6:10, k])
    return dict(traj=traj, res=res, delta_phi=delta_phi, err_to_qd=err_to_qd,
                T_man=T_man)


def plot_slew_demo(demo, out=OUT_DIR):
    res, traj = demo['res'], demo['traj']
    t, t_traj, T_man = res['t'], traj['t'], demo['T_man']

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    colors = ['C0', 'C1', 'C2']

    # Body rates (nominal dashed, truth solid)
    omega_truth = res['x'][10:13, :]
    for i in range(3):
        axes[0].plot(t_traj, np.degrees(traj['omega_ref'][i, :]), '--',
                     color=colors[i], alpha=0.6, linewidth=0.9)
        axes[0].plot(t, np.degrees(omega_truth[i, :]), color=colors[i],
                     linewidth=1.0, label=fr'$\omega_{"xyz"[i]}$ truth')
    axes[0].set_ylabel('Body rate [deg/s]'); axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=3, title='solid: truth, dashed: nominal',
                   title_fontsize=8)

    # Per-wheel torques
    for i in range(4):
        axes[1].plot(t[:-1], res['u_W'][i, :], linewidth=0.9, label=f'wheel {i+1}')
    for sign in (+1, -1):
        axes[1].axhline(sign * WHEEL_TORQUE_MAX, color='k', linestyle=':',
                        linewidth=0.7)
    axes[1].set_ylabel(r'Wheel torque $u_W$ [Nm]')
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8, ncol=4)

    # Tracking error in axis-angle vector form (Lecture 17 page 5)
    for i in range(3):
        axes[2].plot(t, np.degrees(demo['delta_phi'][i, :]), color=colors[i],
                     linewidth=0.9, label=fr'$\delta\phi_{"xyz"[i]}$')
    axes[2].axvline(T_man, color='gray', linestyle=':', linewidth=0.7,
                    label='plan ends')
    axes[2].set_ylabel(r'Tracking error $\delta\phi$ [deg]')
    axes[2].set_xlabel('Time [s]')
    axes[2].grid(alpha=0.3); axes[2].legend(fontsize=8, ncol=4)

    fig.suptitle(f'HW4 Part 4: $180^\\circ$ eigen-axis slew, $T_\\mathrm{{man}} = {T_man:.0f}$ s')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'hw4_pt4_slew_180.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


# ---- Test 2: regulator vs slew from 170 deg ----

def regulator_vs_slew(angle_deg=170.0, T_man=1800.0, dt=1.0, seed=23):
    rng = np.random.default_rng(seed)
    axis = rng.standard_normal(3); axis /= np.linalg.norm(axis)
    q_d = np.array([1.0, 0, 0, 0])
    q_truth = L(q_d) @ expq(0.5 * np.radians(angle_deg) * axis)
    q_truth /= np.linalg.norm(q_truth)
    print(f"170 deg comparison along {axis}")
    x0 = _orbit_state(q_truth)

    res_reg = run_closed_loop(regulator_only_controller(q_d),
                              t_final=T_man + 600.0, dt=dt, x0_truth=x0,
                              q0_est=q_truth.copy(), seed=seed)
    traj = reference_trajectory(q_truth, q_d, T_man=T_man, dt=dt, J=J_dc)
    res_slew = run_closed_loop(slew_controller(traj, q_d),
                               t_final=T_man + 600.0, dt=dt, x0_truth=x0,
                               q0_est=q_truth.copy(), seed=seed)

    n = res_reg['x'].shape[1]
    err_reg = np.array([_err_deg(q_d, res_reg['x'][6:10, k]) for k in range(n)])
    err_slew = np.array([_err_deg(q_d, res_slew['x'][6:10, k]) for k in range(n)])
    print(f"  regulator: peak |u_W| = {np.max(np.abs(res_reg['u_W'])):.4f}, "
          f"final = {err_reg[-1]:.2f} deg")
    print(f"  slew     : peak |u_W| = {np.max(np.abs(res_slew['u_W'])):.4f}, "
          f"final = {err_slew[-1]:.2f} deg")
    return dict(res_reg=res_reg, err_reg=err_reg, res_slew=res_slew,
                err_slew=err_slew, T_man=T_man)


def plot_comparison(comp, out=OUT_DIR):
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    t = comp['res_reg']['t']
    axes[0].plot(t, comp['err_reg'], 'C3', linewidth=1.0, label='regulator')
    axes[0].plot(t, comp['err_slew'], 'C0', linewidth=1.0, label='eigen-axis slew')
    axes[0].axvline(comp['T_man'], color='gray', linestyle=':', linewidth=0.7)
    axes[0].set_ylabel('|attitude error| [deg]')
    axes[0].set_yscale('log'); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=9)

    axes[1].plot(t[:-1], np.max(np.abs(comp['res_reg']['u_W']), axis=0),
                 'C3', linewidth=0.9, label='regulator')
    axes[1].plot(t[:-1], np.max(np.abs(comp['res_slew']['u_W']), axis=0),
                 'C0', linewidth=0.9, label='eigen-axis slew')
    axes[1].axhline(WHEEL_TORQUE_MAX, color='k', linestyle='--', linewidth=0.7,
                    label=f'wheel limit {WHEEL_TORQUE_MAX:.2f} Nm')
    axes[1].set_ylabel(r'$\max_i |u_{W,i}|$ [Nm]')
    axes[1].set_xlabel('Time [s]'); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=9)
    fig.suptitle('HW4 Part 4: regulator vs. planned eigen-axis slew from $170^\\circ$')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'hw4_pt4_comparison.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    print("\n--- Test 1: 180 deg eigen-axis slew ---")
    demo = slew_180_demo()
    plot_slew_demo(demo)
    print(f"  peak tracking error: "
          f"{np.degrees(np.max(np.linalg.norm(demo['delta_phi'], axis=0))):.2f} deg")
    print(f"  final err to q_d:    {demo['err_to_qd'][-1]:.2f} deg")

    print("\n--- Test 2: regulator vs eigen-axis slew from 170 deg ---")
    plot_comparison(regulator_vs_slew())
    print("\nPlots saved in", OUT_DIR)
