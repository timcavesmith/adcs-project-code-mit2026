# feb 27 2026
# Attitude sim (quaternion kinematics + attitude dynamics with Euler's eqns)
# Euler eqn sim + momentum sphere plot

'''
ADCS HW 1: 

Part 2) Verify stability about each principal axis
        - 10 RPM about major axis, same ||h|| for intermediate and minor
        - Add perturbations to test stability and simulate nutation

Part 3) Momentum sphere with all equilibria and several example trajectories
'''

import numpy as np


# ══════════════════════════════════════════════════════════════════════
# HELPER MATH FXNS
# ══════════════════════════════════════════════════════════════════════

def hat(v):
    """Skew-symmetric (hat) matrix from 3-vector. hat(v) @ x == cross(v, x)."""
    return np.array([[ 0,    -v[2],  v[1]],
                     [ v[2],  0,    -v[0]],
                     [-v[1],  v[0],  0   ]])


H = np.vstack([np.zeros(3), np.eye(3)])   # (4x3) quaternion embedding matrix


def L(q):
    s = q[0]
    v = q[1:]
    return np.block([[s,              -v],
                     [v.reshape(-1,1), s*np.eye(3) + hat(v)]])


def R(q):
    s = q[0]
    v = q[1:]
    return np.block([[s,              -v],
                     [v.reshape(-1,1), s*np.eye(3) - hat(v)]])


def G(q):
    """Attitude Jacobian: qdot = 0.5 * G(q) @ omega.  (4x3)"""
    return L(q) @ H


def quat_to_rotmat(q):
    """Quaternion to rotation matrix (body -> inertial)."""
    return H.T @ L(q) @ R(q).T @ H


# ══════════════════════════════════════════════════════════════════════
# INERTIA & DYNAMICS
# ══════════════════════════════════════════════════════════════════════

J = np.diag([1.0, 2.0, 3.0])            # principal inertia [kg*m^2]
J_diag = np.array([1.0, 2.0, 3.0])


def dynamics(x):
    """State derivative: torque-free Euler + quaternion kinematics.
    State x = [q(4), omega(3)] -> xdot = [qdot(4), omegadot(3)]
    """
    q     = x[0:4]
    omega = x[4:]

    qdot     = 0.5 * G(q) @ omega
    tau      = np.zeros(3)
    omegadot = (tau - hat(omega) @ J @ omega) / J_diag

    return np.concatenate([qdot, omegadot])


def rk4step(x, h):
    """Single RK4 step with quaternion renormalization."""
    k1 = dynamics(x)
    k2 = dynamics(x + k1 * h/2)
    k3 = dynamics(x + k2 * h/2)
    k4 = dynamics(x + h * k3)
    xn = x + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    xn[0:4] /= np.linalg.norm(xn[0:4]) # renormalize
    return xn


def propagate(omega0, h_step=0.1, n_steps=3000, q0=None):
    """Run attitude sim from given omega0. Returns xhist (7 x n_steps), t."""
    if q0 is None:
        q0 = np.array([1.0, 0, 0, 0])
    x0 = np.concatenate([q0, omega0])
    xhist = np.zeros((7, n_steps))
    xhist[:, 0] = x0
    for k in range(n_steps - 1):
        xhist[:, k+1] = rk4step(xhist[:, k], h_step)
    t = np.linspace(0, n_steps * h_step, n_steps)
    return xhist, t


def postprocess(xhist):
    """Compute energy, inertial h, and normalized body h from sim history."""
    n = xhist.shape[1]
    T_energy      = np.zeros(n)
    h_inertial    = np.zeros((3, n))
    h_body_normed = np.zeros((3, n))

    for k in range(n):
        q_k     = xhist[0:4, k]
        omega_k = xhist[4:7, k]
        h_body  = J @ omega_k

        T_energy[k]        = 0.5 * omega_k @ h_body
        h_inertial[:, k]   = quat_to_rotmat(q_k) @ h_body
        h_body_normed[:, k] = h_body / np.linalg.norm(h_body)

    return T_energy, h_inertial, h_body_normed


# ══════════════════════════════════════════════════════════════════════
# PART 2: SETUP CASES
# ══════════════════════════════════════════════════════════════════════
#
# 10 RPM about major axis (axis 3, J=3).
# ||h|| = J3 * omega3.  For other axes, omega_i = ||h|| / Ji
# so angular momentum magnitude is the same for all three.
# Small perturbation on off-axes to excite nutation.

rpm = 10.0
omega_major = rpm * 2 * np.pi / 60   # rad/s

# angular momentum magnitude from major axis spin
h_mag = J_diag[2] * omega_major      # ||h|| = J3 * omega3

# omega for each axis to give same ||h||
omega_ax1 = h_mag / J_diag[0]        # minor axis   (J1 = 1)
omega_ax2 = h_mag / J_diag[1]        # intermediate (J2 = 2)
omega_ax3 = omega_major               # major axis   (J3 = 3)

pert = 0.05  # perturbation [rad/s]

# sim params
h_step  = 0.05
n_steps = 6000    # 300 s total

cases = {
    'Major axis (axis 3, J=3) - STABLE': {
        'omega0': np.array([pert, pert, omega_ax3]),
        'color': 'tab:blue',
    },
    'Intermediate axis (axis 2, J=2) - UNSTABLE': {
        'omega0': np.array([pert, omega_ax2, pert]),
        'color': 'tab:red',
    },
    'Minor axis (axis 1, J=1) - STABLE': {
        'omega0': np.array([omega_ax1, pert, pert]),
        'color': 'tab:green',
    },
}

# Extra trajectories for part 3 momentum sphere
np.random.seed(1)
extra_cases = {}
for i in range(4):
    # random direction in h-space, same ||h||, then omega = J^{-1} h
    direction = np.random.randn(3)
    direction /= np.linalg.norm(direction)
    h_vec = h_mag * direction
    omega_rand = h_vec / J_diag
    omega_rand += 0.02 * np.random.randn(3)
    extra_cases[f'Random trajectory {i+1}'] = {
        'omega0': omega_rand,
        'color': f'C{i+4}',
    }


# ══════════════════════════════════════════════════════════════════════
# PROPAGATE ALL CASES
# ══════════════════════════════════════════════════════════════════════

results = {}

for name, case in {**cases, **extra_cases}.items():
    xhist, t = propagate(case['omega0'], h_step=h_step, n_steps=n_steps)
    T_energy, h_inertial, h_body_normed = postprocess(xhist)
    results[name] = {
        'xhist': xhist,
        't': t,
        'T_energy': T_energy,
        'h_inertial': h_inertial,
        'h_body_normed': h_body_normed,
        'color': case['color'],
        'omega0': case['omega0'],
    }


if __name__ == "__main__":
    print(f"10 RPM = {omega_major:.4f} rad/s")
    print(f"||h|| = {h_mag:.4f} N*m*s")
    print(f"omega for same ||h||:  axis1={omega_ax1:.4f}  axis2={omega_ax2:.4f}  axis3={omega_ax3:.4f}")
    print(f"Perturbation: {pert} rad/s")
    print(f"Sim: {n_steps} steps, h={h_step} s, tf={n_steps*h_step:.1f} s\n")

    for name in cases:
        r = results[name]
        e_drift = abs(r['T_energy'][-1] - r['T_energy'][0])
        h_drift = abs(np.linalg.norm(r['h_inertial'][:,-1]) - np.linalg.norm(r['h_inertial'][:,0]))
        print(f"{name}")
        print(f"  omega0 = {r['omega0']}")
        print(f"  Energy drift: {e_drift:.2e}")
        print(f"  |h| drift:    {h_drift:.2e}\n")
