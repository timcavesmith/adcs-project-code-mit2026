# HW2 Part 1: safe mode — perturbed inertia, 10 RPM spin, superspin rotor, gyrostat sim

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from attitude_dynamics import J as J_nominal, hat, G, quat_to_rotmat


def exp_hat(v):
    """Rotation matrix from axis-angle v (Rodrigues). exp(hat(v))."""
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    u = v / th
    return np.eye(3) + np.sin(th) * hat(u) + (1 - np.cos(th)) * (hat(u) @ hat(u))


def perturb_inertia(J, d_std=0.03, v_std=np.radians(2), rng=None):
    """
    J = V D V^T, then D_tilde = D(I + diag(d)), V_tilde = V @ exp(hat(v)), J_tilde = V_tilde D_tilde V_tilde^T.
    d, v from Gaussian.
    """
    if rng is None:
        rng = np.random.default_rng()
    eigvals, V = np.linalg.eigh(J)
    d = rng.normal(0, d_std, 3)
    D_tilde = np.diag(eigvals * (1 + d))
    v = rng.normal(0, v_std, 3)
    V_tilde = V @ exp_hat(v)
    return V_tilde @ D_tilde @ V_tilde.T, eigvals, V, V_tilde, D_tilde


# solar panel normal in body frame (not principal so spin axis is off-principal)
n_sun_body = np.array([1.0, 0.15, 0.08])
n_sun_body /= np.linalg.norm(n_sun_body)

rpm = 10.0
omega_mag = rpm * 2 * np.pi / 60  # rad/s
omega_des = omega_mag * n_sun_body


def rotor_momentum_for_superspin(J, omega, inertia_ratio_min=1.2):
    """
    h_total = J*omega + h_r parallel to omega => h_r = lambda*omega - J*omega.
    Choose lambda so effective spin inertia lambda >= inertia_ratio_min * J_min.
    """
    J_omega = J @ omega
    # effective spin inertia = (h_total . omega) / |omega|^2 = lambda
    # require lambda >= inertia_ratio_min * min principal of J
    J_min = np.min(np.linalg.eigvalsh(J))
    lam = inertia_ratio_min * J_min
    h_r = lam * omega - J_omega
    return h_r, lam


def dynamics_gyrostat(x, J, h_r):
    """State x = [q(4), omega(3)]. Gyrostat: J*omega_dot = -omega x (J*omega + h_r)."""
    q = x[0:4]
    omega = x[4:7]
    qdot = 0.5 * G(q) @ omega
    h_total = J @ omega + h_r
    omegadot = np.linalg.solve(J, -np.cross(omega, h_total))
    return np.concatenate([qdot, omegadot])


def rk4step_gyrostat(x, h, J, h_r):
    def f(xx):
        return dynamics_gyrostat(xx, J, h_r)
    k1 = f(x)
    k2 = f(x + k1 * h / 2)
    k3 = f(x + k2 * h / 2)
    k4 = f(x + h * k3)
    xn = x + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    xn[0:4] /= np.linalg.norm(xn[0:4])
    return xn


def propagate_gyrostat(omega0, J, h_r, q0=None, h_step=0.01, n_steps=6000):
    if q0 is None:
        q0 = np.array([1.0, 0, 0, 0])
    x0 = np.concatenate([q0, omega0])
    xhist = np.zeros((7, n_steps))
    xhist[:, 0] = x0
    for k in range(n_steps - 1):
        xhist[:, k + 1] = rk4step_gyrostat(xhist[:, k], h_step, J, h_r)
    t = np.linspace(0, n_steps * h_step, n_steps)
    return xhist, t


rng = np.random.default_rng(42)
J, _, _, _, _ = perturb_inertia(J_nominal, d_std=0.02, v_std=np.radians(1.5), rng=rng)
h_r, lam = rotor_momentum_for_superspin(J, omega_des, inertia_ratio_min=1.2)
J_min = np.min(np.linalg.eigvalsh(J))
ratio_eff = lam / J_min
pert_omega = 0.02 * np.array([0.5, -0.3, 0.4])

n_sun_eci = np.array([1.0, 0, 0])


def pointing_error_deg(xhist):
    n = xhist.shape[1]
    err = np.zeros(n)
    for k in range(n):
        q_k = xhist[0:4, k]
        n_body_in_eci = quat_to_rotmat(q_k) @ n_sun_body
        err[k] = np.degrees(np.arccos(np.clip(np.dot(n_body_in_eci, n_sun_eci), -1, 1)))
    return err


if __name__ == "__main__":
    q0 = np.array([1.0, 0, 0, 0])
    h_step = 0.01
    n_steps = 6000
    xhist_nom, t = propagate_gyrostat(omega_des, J, h_r, q0=q0, h_step=h_step, n_steps=n_steps)
    xhist_pert, _ = propagate_gyrostat(omega_des + pert_omega, J, h_r, q0=q0, h_step=h_step, n_steps=n_steps)

    def postprocess(xhist, J_loc, h_r_loc):
        n = xhist.shape[1]
        T_energy = np.zeros(n)
        h_mag = np.zeros(n)
        for k in range(n):
            omega_k = xhist[4:7, k]
            h_total = J_loc @ omega_k + h_r_loc
            T_energy[k] = 0.5 * omega_k @ (J_loc @ omega_k)
            h_mag[k] = np.linalg.norm(h_total)
        return T_energy, h_mag

    T_nom, _ = postprocess(xhist_nom, J, h_r)
    T_pert, _ = postprocess(xhist_pert, J, h_r)
    err_nom = pointing_error_deg(xhist_nom)
    err_pert = pointing_error_deg(xhist_pert)

    out_dir = os.path.join(os.path.dirname(__file__), '..')
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    ax = axes[0, 0]
    ax.plot(t, xhist_nom[4, :], label='omega_1')
    ax.plot(t, xhist_nom[5, :], label='omega_2')
    ax.plot(t, xhist_nom[6, :], label='omega_3')
    ax.set_ylabel('omega [rad/s]')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title('Nominal spin (10 RPM about panel normal)')

    ax = axes[0, 1]
    ax.plot(t, xhist_pert[4, :], label='omega_1')
    ax.plot(t, xhist_pert[5, :], label='omega_2')
    ax.plot(t, xhist_pert[6, :], label='omega_3')
    ax.set_ylabel('omega [rad/s]')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title('Perturbed ICs')

    ax = axes[1, 0]
    ax.plot(t, err_nom, 'b', label='nominal')
    ax.plot(t, err_pert, 'r', alpha=0.8, label='perturbed')
    ax.set_ylabel('Pointing error [deg]')
    ax.set_xlabel('Time [s]')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t, T_nom, 'b', label='T nom')
    ax.plot(t, T_pert, 'r', alpha=0.8, label='T pert')
    ax.set_ylabel('Kinetic energy')
    ax.set_xlabel('Time [s]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.suptitle('HW2.1 Safe mode: gyrostat spin + perturbed ICs')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'hw2_safe_mode.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print('J_nominal diag approx:', np.diag(J_nominal))
    print('J_perturbed diag approx:', np.diag(J))
    print('n_sun_body:', n_sun_body)
    print('omega_des [rad/s]:', omega_des)
    print('|omega_des|:', np.linalg.norm(omega_des))
    print('h_r:', h_r)
    print('effective inertia ratio (lambda/J_min):', ratio_eff)
    print('Pointing error nominal (deg) final:', err_nom[-1])
    print('Pointing error perturbed (deg) final:', err_pert[-1])
