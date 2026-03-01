# feb 27 2026
# Attitude sim (quaternion kinematics + attitude dynamics with Euler's eqns)
# Euler eqn sim + momentum sphere plot


'''
ADCS HW 1: 

Part 3) 3-DOF 3D attitude dynamics sim 

- ideally later we can expand it to be 6-DOF 3D 

packages:
- use matplotlib for basic output plots
- perhaps try pyvista for attitude dynamics modeling later (whatever david used)
- or can stick with plotly which i used with orbit dynamics (browser and gives TOF slider)


units:
- rad, rad/s

frame: inertial and body

PLAN:

Attitude sim (quaternion kinematics + attitude dynamics with Euler's eqns)
# Euler eqn sim + momentum sphere plot

'''

import numpy as np
import matplotlib.pyplot as plt


# helper math fxns
def hat(v: np.array):
    return np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])

# test that this is equivalent to cross product
''' 
a = np.array([1, 2, 3])
b = np.array([4,5,6])
cross = np.cross(a,b)
hat_op = hat(a) @ b
diff = cross - hat_op
if np.linalg.norm(diff) < 1e-3:
    print(f"hat test passed, diff: {diff}")
else:
    print("test failed")

'''

H = np.vstack([np.zeros(3), np.eye(3)])

# print(H)

# dumb way:
# top = np.hstack([1, np.zeros(3)])
# T = np.vstack([top, -H.T ])

# better way: (quaternion conjugate)
T = np.diag([1, -1, -1, -1])

#print(T)

def L(q):
    s = q[0]
    v = q[1:]
    return np.block([[s,    -v],
                     [v.reshape(-1, 1), s*np.eye(3) + hat(v)]])

qi = np.array([1, 0, 0, 0])

# print(L(qi))

def R(q):
    s = q[0]
    v = q[1:4]
    return np.block([[s,    -v],
                     [v.reshape(-1,1), s*np.eye(3) - hat(v) ]])

# print(R(qi))

def G(q):
    return L(q) @ H

# print(G(qi))

def quat_to_rotmat(q):
    return H.T @ L(q) @ R(q).T @ H

# print(Q(qi))



def dynamics(x):
    q = x[0:4]
    omega = x[4:]
    
    #omega_norm = np.linalg.norm(omega)
    qdot = 0.5 * G(q) @ omega 
    tau = np.zeros(3) # adding torque explicitly for later 
    omegadot = np.linalg.solve(J, tau - hat(omega) @ J @ omega) 
    xdot = np.concatenate([qdot, omegadot])
    return xdot


def rk4step(x, h):
    k1 = dynamics(x)
    k2 = dynamics(x + k1 * h/2)
    k3 = dynamics(x + k2 * h/2)
    k4 = dynamics(x + h * k3)
    xn = x + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

    # renormalize quaternion to stay on unit sphere
    xn[0:4] = xn[0:4] / np.linalg.norm(xn[0:4])
    return xn

# ══════════════════════════════════════════════════════════════════════
# SET PARAMS
# ══════════
np.random.seed(1)
J = np.diag([1.0, 2.0, 3.0]) # principal inertia [kg·m²]

# sample random angular velocity
q0 = np.array([1, 0, 0, 0])
# omega0 = np.random.randn(3)
omega0 = np.array([0.0 , 1.0, 0.0]) + 0.1 * np.random.randn(3)
x0 = np.concatenate([q0, omega0])



# ══════════════════════════════════════════════════════════════════════
# PROPAGATE
# ══════════════════════════════════════════════════════════════════════
h_step = 0.1  # time step [s]
n_steps = 1000 # time steps
tf = n_steps*h_step # final time


xhist = np.zeros((7, n_steps))
xhist[:, 0] = x0

for k in range(n_steps - 1):
    xhist[:, k + 1] = rk4step(xhist[:, k], h_step)

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
