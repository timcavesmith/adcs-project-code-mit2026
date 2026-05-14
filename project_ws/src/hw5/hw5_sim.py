# HW5: numerical-optimization slew guidance, tracked with the HW4 PD.
#   1. 180 deg slew: OCP plan, then closed-loop with truth+MEKF.
#   2. Comparison vs the HW4 versine on the same boundary conditions.

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hw3.utils import expq
from hw1.attitude_dynamics import J as J_dc
from hw4.sim_driver import run_closed_loop
from hw4.dynamics import make_state, coe2rv
from hw4.regulator import pd_gains, regulator_command, attitude_error
from hw4.slew import reference_trajectory, slew_command
from hw4.actuators import WHEEL_TORQUE_MAX, WHEEL_MOMENTUM_MAX, B_W
from hw5.slew_ocp import build_reference, slew_command_ocp

OUT_DIR = os.path.dirname(__file__)

# Reuse the HW4 regulator gains. Both controllers track a smooth reference and
# the same double-integrator linearization applies.
OMEGA_N = 5e-3
ZETA = 1.0
K_P, K_D = pd_gains(J_dc, OMEGA_N, ZETA)

# Body-rate ceiling for the OCP path constraint. Bounds the search but ends
# up slack at the 180 deg optimum (peak omega ~0.18 deg/s).
OMEGA_MAX = np.radians(0.3)

# OCP cost weights. w_t penalizes maneuver time, R penalizes wheel torque
# magnitude, Q penalizes per-wheel momentum so the planner avoids riding the
# 60 Nms saturation any longer than necessary.
W_T = 1.0
R_DIAG = 20.0 * np.ones(4)
Q_DIAG = 1e-5 * np.ones(4)
N_NODES = 40


def _orbit_state(q_truth):
    a, e, inc = 6800.0, 0.001, np.radians(51.64)
    r0, v0 = coe2rv(a, e, inc, np.radians(30), np.radians(60), 0.0)
    return make_state(r0, v0, q_truth, np.zeros(3))


def _err_deg(q_d, q):
    return np.degrees(np.linalg.norm(attitude_error(q_d, q)))


def ocp_controller(traj, q_d):
    """Track the OCP plan while it is active, then hand off to the HW4
    regulator -- same trick as the versine to avoid riding any terminal kink
    in the open-loop control."""
    n_traj = traj['t'].size

    def ctrl(k, q_hat, omega_hat, rho_hat):
        if k < n_traj - 1:
            u_W, tau_fb, _ = slew_command_ocp(traj, k, q_hat, omega_hat,
                                              K_P, K_D, saturate=False)
            return u_W, dict(tau_cmd=tau_fb)
        u_W, tau = regulator_command(q_d, q_hat, omega_hat, rho_hat,
                                     J_dc, K_P, K_D, saturate=False)
        return u_W, dict(tau_cmd=tau)
    return ctrl


def versine_controller(traj, q_d):
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


# ---- Test 1: 180 deg OCP slew ----

def slew_180_demo_ocp(axis=None, dt=1.0, seed=11):
    if axis is None:
        axis = np.array([1.0, 0.5, -0.3]); axis /= np.linalg.norm(axis)
    q_0 = np.array([1.0, 0, 0, 0])
    q_d = expq(0.5 * np.pi * axis)

    traj = build_reference(q_0, q_d, J=J_dc, dt_sample=dt,
                           u_max=WHEEL_TORQUE_MAX,
                           w_max=WHEEL_MOMENTUM_MAX,
                           omega_max=OMEGA_MAX,
                           N=N_NODES, w_t=W_T,
                           R_diag=R_DIAG, Q_diag=Q_DIAG,
                           verbose=False)
    sol = traj['sol']
    print(f"180 deg OCP slew about {axis}")
    print(f"  t_f* = {sol['t_f']:.1f} s   cost = {sol['cost']:.2f}")
    print(f"  peak nominal |u_W| = {np.max(np.abs(sol['u'])):.3f} Nm "
          f"(limit {WHEEL_TORQUE_MAX})")
    print(f"  peak nominal |w|   = {np.max(np.abs(sol['w'])):.2f} Nms "
          f"(limit {WHEEL_MOMENTUM_MAX})")
    print(f"  peak nominal |omega| = "
          f"{np.degrees(np.max(np.linalg.norm(sol['omega'], axis=0))):.3f} deg/s")

    res = run_closed_loop(ocp_controller(traj, q_d),
                          t_final=sol['t_f'] + 600.0,
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
                T_man=sol['t_f'], sol=sol, q_d=q_d, axis=axis)


def plot_slew_demo(demo, out=OUT_DIR):
    res, traj = demo['res'], demo['traj']
    t, t_traj, T_man = res['t'], traj['t'], demo['T_man']

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    colors = ['C0', 'C1', 'C2']

    omega_truth = res['x'][10:13, :]
    for i in range(3):
        axes[0].plot(t_traj, np.degrees(traj['omega_ref'][i, :]), '--',
                     color=colors[i], alpha=0.6, linewidth=0.9)
        axes[0].plot(t, np.degrees(omega_truth[i, :]), color=colors[i],
                     linewidth=1.0, label=fr'$\omega_{"xyz"[i]}$ truth')
    axes[0].set_ylabel('Body rate [deg/s]')
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=3, title='solid: truth, dashed: plan',
                   title_fontsize=8)

    for i in range(4):
        axes[1].plot(t[:-1], res['u_W'][i, :], linewidth=0.9, label=f'wheel {i+1}')
    for sign in (+1, -1):
        axes[1].axhline(sign * WHEEL_TORQUE_MAX, color='k', linestyle=':',
                        linewidth=0.7)
    axes[1].set_ylabel(r'Wheel torque $u_W$ [Nm]')
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8, ncol=4)

    for i in range(3):
        axes[2].plot(t, np.degrees(demo['delta_phi'][i, :]), color=colors[i],
                     linewidth=0.9, label=fr'$\delta\phi_{"xyz"[i]}$')
    axes[2].axvline(T_man, color='gray', linestyle=':', linewidth=0.7,
                    label='plan ends')
    axes[2].set_ylabel(r'Tracking error $\delta\phi$ [deg]')
    axes[2].set_xlabel('Time [s]')
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8, ncol=4)

    fig.suptitle(fr'$180^\circ$ OCP slew, $t_f^* = {T_man:.0f}$ s')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'hw5_slew_180.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


# ---- Test 2: OCP vs versine on the same slew ----

def comparison_vs_versine(axis=None, T_man_versine=1800.0, dt=1.0, seed=11):
    if axis is None:
        axis = np.array([1.0, 0.5, -0.3]); axis /= np.linalg.norm(axis)
    q_0 = np.array([1.0, 0, 0, 0])
    q_d = expq(0.5 * np.pi * axis)

    versine = reference_trajectory(q_0, q_d, T_man=T_man_versine, dt=dt, J=J_dc)
    ocp = build_reference(q_0, q_d, J=J_dc, dt_sample=dt,
                          u_max=WHEEL_TORQUE_MAX,
                          w_max=WHEEL_MOMENTUM_MAX,
                          omega_max=OMEGA_MAX,
                          N=N_NODES, w_t=W_T,
                          R_diag=R_DIAG, Q_diag=Q_DIAG,
                          verbose=False)
    t_final = max(T_man_versine, ocp['sol']['t_f']) + 600.0
    res_versine = run_closed_loop(versine_controller(versine, q_d),
                                  t_final=t_final, dt=dt,
                                  x0_truth=_orbit_state(q_0),
                                  q0_est=q_0.copy(), seed=seed)
    res_ocp = run_closed_loop(ocp_controller(ocp, q_d),
                              t_final=t_final, dt=dt,
                              x0_truth=_orbit_state(q_0),
                              q0_est=q_0.copy(), seed=seed)

    return dict(versine=versine, res_versine=res_versine,
                ocp=ocp, res_ocp=res_ocp,
                T_man_versine=T_man_versine, T_man_ocp=ocp['sol']['t_f'],
                q_d=q_d, sol=ocp['sol'], dt=dt)


def plot_comparison(comp, out=OUT_DIR):
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    t_v = comp['res_versine']['t']
    t_o = comp['res_ocp']['t']

    # Attitude error to q_d
    err_v = np.array([_err_deg(comp['q_d'], comp['res_versine']['x'][6:10, k])
                      for k in range(t_v.size)])
    err_o = np.array([_err_deg(comp['q_d'], comp['res_ocp']['x'][6:10, k])
                      for k in range(t_o.size)])
    axes[0].plot(t_v, err_v, 'C0', linewidth=1.0,
                 label=fr'versine, $T_\mathrm{{man}} = {comp["T_man_versine"]:.0f}$ s')
    axes[0].plot(t_o, err_o, 'C3', linewidth=1.0,
                 label=fr'OCP, $t_f^* = {comp["T_man_ocp"]:.0f}$ s')
    axes[0].axvline(comp['T_man_versine'], color='C0', linestyle=':', linewidth=0.7)
    axes[0].axvline(comp['T_man_ocp'], color='C3', linestyle=':', linewidth=0.7)
    axes[0].set_ylabel('|attitude error| [deg]')
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=9)

    # Max per-wheel torque magnitude over time
    axes[1].plot(t_v[:-1], np.max(np.abs(comp['res_versine']['u_W']), axis=0),
                 'C0', linewidth=0.9, label='versine')
    axes[1].plot(t_o[:-1], np.max(np.abs(comp['res_ocp']['u_W']), axis=0),
                 'C3', linewidth=0.9, label='OCP')
    axes[1].axhline(WHEEL_TORQUE_MAX, color='k', linestyle='--', linewidth=0.7,
                    label=f'wheel limit {WHEEL_TORQUE_MAX:.2f} Nm')
    axes[1].set_ylabel(r'$\max_i |u_{W,i}|$ [Nm]')
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=9)

    # Per-wheel momentum: planner reference (dashed) vs flown trajectory
    # (solid). The truth state only carries body rho, but each wheel's
    # momentum is the integral of its own torque command, w_i = cumsum(u_W,i).
    B_pinv = np.linalg.pinv(B_W)
    w_ver_plan = B_pinv @ comp['versine']['rho_ref']
    w_ocp_plan = comp['ocp']['w_ref']
    w_ver_flown = np.cumsum(comp['res_versine']['u_W'], axis=1) * comp['dt']
    w_ocp_flown = np.cumsum(comp['res_ocp']['u_W'], axis=1) * comp['dt']
    axes[2].plot(comp['versine']['t'], np.max(np.abs(w_ver_plan), axis=0),
                 'C0--', linewidth=0.9, label='versine plan')
    axes[2].plot(comp['res_versine']['t'][:-1], np.max(np.abs(w_ver_flown), axis=0),
                 'C0', linewidth=0.9, label='versine flown')
    axes[2].plot(comp['ocp']['t'], np.max(np.abs(w_ocp_plan), axis=0),
                 'C3--', linewidth=0.9, label='OCP plan')
    axes[2].plot(comp['res_ocp']['t'][:-1], np.max(np.abs(w_ocp_flown), axis=0),
                 'C3', linewidth=0.9, label='OCP flown')
    axes[2].axhline(WHEEL_MOMENTUM_MAX, color='k', linestyle=':', linewidth=0.7,
                    label=f'wheel limit {WHEEL_MOMENTUM_MAX:.0f} Nms')
    axes[2].set_ylabel(r'$\max_i |w_i|$ [Nms]')
    axes[2].set_xlabel('Time [s]')
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8, ncol=2)

    fig.suptitle(r'Versine eigen-axis vs OCP on the same $180^\circ$ slew')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'hw5_comparison.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    print("\n--- HW5 Test 1: 180 deg OCP slew ---")
    demo = slew_180_demo_ocp()
    plot_slew_demo(demo)
    print(f"  peak tracking error: "
          f"{np.degrees(np.max(np.linalg.norm(demo['delta_phi'], axis=0))):.2f} deg")
    print(f"  final err to q_d:    {demo['err_to_qd'][-1]:.2f} deg")

    print("\n--- HW5 Test 2: versine vs OCP on the same 180 deg slew ---")
    comp = comparison_vs_versine()
    plot_comparison(comp)
    print(f"  versine slew time: {comp['T_man_versine']:.0f} s")
    print(f"  OCP slew time:     {comp['T_man_ocp']:.0f} s "
          f"(speed-up {100*(1-comp['T_man_ocp']/comp['T_man_versine']):.1f}%)")
    B_pinv = np.linalg.pinv(B_W)
    w_ver_plan = np.max(np.abs(B_pinv @ comp['versine']['rho_ref']))
    w_ocp_plan = np.max(np.abs(comp['ocp']['w_ref']))
    w_ver_flown = np.max(np.abs(np.cumsum(comp['res_versine']['u_W'], axis=1) * comp['dt']))
    w_ocp_flown = np.max(np.abs(np.cumsum(comp['res_ocp']['u_W'], axis=1) * comp['dt']))
    print(f"  peak |w_i|  versine: plan {w_ver_plan:.1f}, flown {w_ver_flown:.1f} Nms "
          f"(limit {WHEEL_MOMENTUM_MAX})")
    print(f"  peak |w_i|  OCP:     plan {w_ocp_plan:.1f}, flown {w_ocp_flown:.1f} Nms")

    print("\nPlots saved in", OUT_DIR)
