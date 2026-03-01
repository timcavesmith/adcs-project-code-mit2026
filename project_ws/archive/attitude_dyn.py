# feb 27 2026
# Attitude sim (quaternion kinematics + attitude dynamics with Euler's eqns)
# Euler eqn sim + momentum sphere plot

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

T_conj = np.diag([1, -1, -1, -1])         # quaternion conjugate matrix


def L(q):
    """Left quaternion multiply matrix: L(q) @ p == q ⊗ p."""
    s = q[0]
    v = q[1:]
    return np.block([[s,              -v],
                     [v.reshape(-1,1), s*np.eye(3) + hat(v)]])


def R(q):
    """Right quaternion multiply matrix: R(q) @ p == p ⊗ q."""
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

J = np.diag([1.0, 2.0, 3.0])            # principal inertia [kg·m²]
J_diag = np.array([1.0, 2.0, 3.0])      # diagonal for fast element-wise ops


def dynamics(x):
    """State derivative for attitude dynamics (torque-free Euler + quat kinematics).
    
    State x = [q (4), omega (3)]  ->  xdot = [qdot (4), omegadot (3)]
    """
    q     = x[0:4]
    omega = x[4:]

    qdot     = 0.5 * G(q) @ omega
    tau      = np.zeros(3)                          # zero external torque (for now)
    omegadot = (tau - np.cross(omega, J @ omega)) / J_diag
    
    return np.concatenate([qdot, omegadot])


def rk4step(x, h):
    """Single RK4 integration step with quaternion renormalization."""
    k1 = dynamics(x)
    k2 = dynamics(x + k1 * h/2)
    k3 = dynamics(x + k2 * h/2)
    k4 = dynamics(x + h * k3)
    xn = x + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

    # renormalize quaternion to stay on unit sphere
    xn[0:4] = xn[0:4] / np.linalg.norm(xn[0:4])
    return xn


# ══════════════════════════════════════════════════════════════════════
# SIMULATION SETUP
# ══════════════════════════════════════════════════════════════════════
np.random.seed(1)
h_step  = 0.1          # time step [s]
n_steps = 1000          # number of steps
tf      = n_steps * h_step

# initial conditions — spin about axis 2 with small perturbation
q0     = np.array([1.0, 0, 0, 0])
omega0 = np.array([0.0, 1.0, 0.0]) + 0.1 * np.random.randn(3)
x0     = np.concatenate([q0, omega0])

# propagate
xhist = np.zeros((7, n_steps))
xhist[:, 0] = x0

for k in range(n_steps - 1):
    xhist[:, k+1] = rk4step(xhist[:, k], h_step)

t = np.linspace(0, tf, n_steps)


# ══════════════════════════════════════════════════════════════════════
# POST-PROCESSING: energy, inertial angular momentum, body h for sphere
# ══════════════════════════════════════════════════════════════════════

T_energy = np.zeros(n_steps)       # kinetic energy
h_inertial = np.zeros((3, n_steps))  # angular momentum in inertial frame
h_body_normed = np.zeros((3, n_steps))  # normalized h in body frame (for momentum sphere)

for k in range(n_steps):
    q_k     = xhist[0:4, k]
    omega_k = xhist[4:7, k]
    h_body  = J @ omega_k

    T_energy[k]       = 0.5 * omega_k @ h_body
    h_inertial[:, k]  = quat_to_rotmat(q_k) @ h_body
    h_body_normed[:, k] = h_body / np.linalg.norm(h_body)


if __name__ == "__main__":
    print(f"Sim complete: {n_steps} steps, tf = {tf:.1f} s")
    print(f"Initial omega: {omega0}")
    print(f"Final omega:   {xhist[4:7, -1]}")
    print(f"Energy drift:  {abs(T_energy[-1] - T_energy[0]):.2e}")
    print(f"|h| drift:     {abs(np.linalg.norm(h_inertial[:,-1]) - np.linalg.norm(h_inertial[:,0])):.2e}")
