# HW4 part 3: attitude regulator. 
#   1. Random initial conditions out to +/- 90 deg.
#   2. Multi-orbit run with disturbances + sensor noise + MEKF in the loop,
#      reporting RMS pointing error.

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
from hw4.actuators import WHEEL_TORQUE_MAX, body_torque_capability
from hw4.disturbances import MU_EARTH

OUT_DIR = os.path.dirname(__file__)


# Closed-loop natural frequency. Orbital rate ~1.13e-3 rad/s; we want the
# controller bandwidth ~5x faster so it can reject orbital-rate gravity-
# gradient disturbances. zeta = 1 (critically damped, no overshoot).
OMEGA_N = 5e-3
ZETA = 1.0
K_P, K_D = pd_gains(J_dc, OMEGA_N, ZETA)


def regulator_controller(q_d):
    def ctrl(k, q_hat, omega_hat, rho_hat):
        u_W, tau = regulator_command(q_d, q_hat, omega_hat, rho_hat,
                                     J_dc, K_P, K_D, saturate=False)
        return u_W, dict(tau_cmd=tau)
    return ctrl


def _orbit_state(q_truth, omega_truth=None):
    a, e, inc = 6800.0, 0.001, np.radians(51.64)
    r0, v0 = coe2rv(a, e, inc, np.radians(30), np.radians(60), 0.0)
    return make_state(r0, v0, q_truth,
                      np.zeros(3) if omega_truth is None else omega_truth)


def _err_deg(q_d, q):
    return np.degrees(np.linalg.norm(attitude_error(q_d, q)))


# ---- Test 1: random ICs out to +/- 90 deg ----

def random_ic_test(n_trials=5, t_final=2500.0, dt=1.0, seed=1):
    rng = np.random.default_rng(seed)
    q_d = np.array([1.0, 0, 0, 0])
    results = []
    for trial in range(n_trials):
        axis = rng.standard_normal(3); axis /= np.linalg.norm(axis)
        angle = rng.uniform(np.radians(45), np.radians(90))
        q_truth = L(q_d) @ expq(0.5 * angle * axis); q_truth /= np.linalg.norm(q_truth)
        omega_truth = np.radians(0.05) * rng.standard_normal(3)
        res = run_closed_loop(regulator_controller(q_d), t_final=t_final, dt=dt,
                              x0_truth=_orbit_state(q_truth, omega_truth),
                              seed=10 + trial)
        n = res['x'].shape[1]
        err = np.array([_err_deg(q_d, res['x'][6:10, k]) for k in range(n)])
        results.append(dict(t=res['t'], err=err, u_W=res['u_W'],
                            tau_cmd=res['tau_cmd'],
                            angle0_deg=np.degrees(angle)))
    return results


def plot_random_ic(results, out=OUT_DIR):
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True)
    cap = body_torque_capability([1, 0, 0])
    for r in results:
        axes[0].plot(r['t'], r['err'], linewidth=0.9,
                     label=f"$|\\phi_0| = {r['angle0_deg']:.0f}^\\circ$")
        axes[1].plot(r['t'][:-1], np.linalg.norm(r['tau_cmd'], axis=0), linewidth=0.9)
        axes[2].plot(r['t'][:-1], np.max(np.abs(r['u_W']), axis=0), linewidth=0.9)
    axes[0].set_ylabel('Attitude error [deg]'); axes[0].set_yscale('log')
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)
    axes[1].axhline(cap, color='k', linestyle='--', linewidth=0.7,
                    label=f'transverse cap {cap:.2f} Nm')
    axes[1].set_ylabel(r'$|\tau_\mathrm{cmd}|$ [Nm]'); axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8, loc='upper right')
    axes[2].axhline(WHEEL_TORQUE_MAX, color='k', linestyle='--', linewidth=0.7,
                    label=f'wheel limit {WHEEL_TORQUE_MAX:.2f} Nm')
    axes[2].set_ylabel(r'$\max_i |u_{W,i}|$ [Nm]')
    axes[2].set_xlabel('Time [s]'); axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8, loc='upper right')
    fig.suptitle('HW4 Part 3: regulator from random initial errors $45$--$90^\\circ$')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'hw4_pt3_random_ics.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


# ---- Test 2: multi-orbit run with disturbances + MEKF ----

def orbit_long_test(n_orbits=2, dt=1.0, q0_offset_deg=10.0, seed=7):
    a = 6800.0
    T_orbit = 2 * np.pi * np.sqrt(a**3 / MU_EARTH)
    print(f"Orbital period {T_orbit:.0f} s, {n_orbits} orbits = {n_orbits*T_orbit:.0f} s")

    q_d = np.array([1.0, 0, 0, 0])
    rng = np.random.default_rng(seed)
    axis = rng.standard_normal(3); axis /= np.linalg.norm(axis)
    q_truth = L(q_d) @ expq(0.5 * np.radians(q0_offset_deg) * axis)
    q_truth /= np.linalg.norm(q_truth)
    res = run_closed_loop(regulator_controller(q_d), t_final=n_orbits * T_orbit,
                          dt=dt, x0_truth=_orbit_state(q_truth), seed=seed)

    n = res['x'].shape[1]
    err = np.array([_err_deg(q_d, res['x'][6:10, k]) for k in range(n)])
    rms = np.sqrt(np.mean(err[res['t'] > T_orbit]**2))
    print(f"  RMS pointing error after first orbit: {rms:.3f} deg")
    print(f"  peak per-wheel |u_W|: {np.max(np.abs(res['u_W'])):.4f} Nm")
    print(f"  |rho| at end: {np.linalg.norm(res['x'][13:16, -1]):.1f} Nms")
    return res, err, rms, T_orbit


def plot_orbit_long(res, err, T_orbit, out=OUT_DIR):
    t = res['t']
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True)
    axes[0].plot(t, err, linewidth=0.7)
    axes[0].set_ylabel('Attitude error [deg]'); axes[0].set_yscale('log')
    axes[0].grid(alpha=0.3)

    rho = res['x'][13:16, :]
    cluster_cap = 4 * 60.0
    for i, lab in enumerate([r'$\rho_1$', r'$\rho_2$', r'$\rho_3$']):
        axes[1].plot(t, rho[i, :], linewidth=0.8, label=lab)
    axes[1].plot(t, np.linalg.norm(rho, axis=0), 'k', linewidth=1.2, label=r'$|\rho|$')
    axes[1].axhline(cluster_cap, color='r', linestyle='--', linewidth=0.7,
                    label=f'cluster cap {cluster_cap:.0f} Nms')
    axes[1].set_ylabel('Body wheel momentum [Nms]')
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8, ncol=2)

    axes[2].plot(t[:-1], np.max(np.abs(res['u_W']), axis=0), linewidth=0.7)
    axes[2].axhline(WHEEL_TORQUE_MAX, color='k', linestyle='--', linewidth=0.7,
                    label=f'wheel limit {WHEEL_TORQUE_MAX:.2f} Nm')
    axes[2].set_ylabel(r'$\max_i |u_{W,i}|$ [Nm]')
    axes[2].set_xlabel('Time [s]'); axes[2].grid(alpha=0.3); axes[2].legend(fontsize=8)
    for ax in axes:
        for k in range(int(t[-1] / T_orbit) + 1):
            ax.axvline(k * T_orbit, color='gray', linestyle=':', linewidth=0.5)
    fig.suptitle('HW4 Part 3: regulator over 2 orbits with disturbances + MEKF')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'hw4_pt3_orbit_long.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    print(f"Gain design: wn = {OMEGA_N:.0e} rad/s, zeta = {ZETA}")
    for ax, i in zip("xyz", range(3)):
        print(f"  axis {ax}: K_p = {K_P[i, i]:.2e}, K_d = {K_D[i, i]:.2e}")

    print("\n--- Test 1: random ICs out to +/- 90 deg ---")
    results = random_ic_test()
    for r in results:
        below = r['err'] < 5.0
        if below.any():
            print(f"  phi0 = {r['angle0_deg']:5.1f} deg: settles <5 deg at "
                  f"t = {r['t'][np.argmax(below)]:.0f} s, "
                  f"final = {r['err'][-1]:.2f} deg")
        else:
            print(f"  phi0 = {r['angle0_deg']:5.1f} deg: did not settle")
    plot_random_ic(results)

    print("\n--- Test 2: 2 orbits + disturbances + MEKF ---")
    res, err, rms, T_orbit = orbit_long_test()
    plot_orbit_long(res, err, T_orbit)
    print("\nPlots saved in", OUT_DIR)
