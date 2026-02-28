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

# set params


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

def Q(q):
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
    return xn


# ══════════════════════════════════════════════════════════════════════
# PROPAGATE
# ══════════════════════════════════════════════════════════════════════

h_step = 0.5                          # time step [s]
n_orbits = 2
n_steps = int(n_orbits * T / h_step)
tf = n_steps * h_step

x0 = np.concatenate([r0, v0])
xhist = np.zeros((6, n_steps))
xhist[:, 0] = x0

for k in range(n_steps - 1):
    xhist[:, k + 1] = rk4step(xhist[:, k], h_step)

t = np.linspace(0, tf, n_steps)

    
''' 

#G = 6.674e-11 [m^3/(kg*s^2] # Universal gravitational parameter 
#M = 5.97e24 # [kg] #Earth mass
mu =  398600.44 #[km^3/s^2]  #G*M

# initialize vectors

# case 1: circular equatorial orbit in XY plane (period 5828.516650846224 s)
r0 = np.array([7000, 0, 0]) # [km] shape (3,) which is 1D and shapeless, not row/col
v0 = np.array([0, 7.546053273069307, 0]) # [km/s] 


#case 2: circular polar orbit in XZ plane
r0 = np.array([0, 0, 7000]) # [km] shape (3,) which is 1D and shapeless, not row/col
v0 = np.array([7.546053273069307, 0, 0]) # [km/s] 


# one orbital period given semimajor axis
T = 2 * np.pi * np.sqrt(np.linalg.norm(r0)**3 / (mu))
# 5828.516650846224 sec for one orbit given those params above


# for COE to RV later
r_earth = 6378.137 #[km] #equatorial radius, semimajor axis

# put COE to RV here?


def dynamics(x):
    r = x[0:3]
    v = x[3:6]
    rdot = v
    r_norm = np.linalg.norm(r)
    vdot = -mu / r_norm**3 * r
    xdot = np.concatenate([rdot, vdot]) #shape (6,) and hstack would do same thing
    return xdot

def rk4step(x, h):
    k1 = dynamics(x)
    k2 = dynamics(x + k1 * h/2)
    k3 = dynamics(x + k2 * h/2)
    k4 = dynamics(x + h * k3)
    xn = x + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    return xn

# sim params
h = 0.5 # time step [s]
#n = 3600 * 90   # one orbit is about 90 min 

n = 5828*2 # sec for our params for one orbit

tf = n*h # [s]

# simulate for n time steps
x0 = np.concatenate([r0, v0])
xhist = np.zeros((6, n))
xhist[:, 0] = x0
for k in range(n-1):
    xhist[:, k + 1] = rk4step(xhist[:, k], h)

# TODO (opt) build legend that displays the current RV, the COEs, and the TOF since R0,V0


# plot result in 2D 

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Top-down view (X-Y plane)
axes[0].plot(xhist[0, :], xhist[1, :])
axes[0].set_xlabel("X [km]")
axes[0].set_ylabel("Y [km]")
axes[0].set_title("Top Down (XY)")
axes[0].set_aspect('equal')
axes[0].grid(True)

# Side view (X-Z plane)
axes[1].plot(xhist[0, :], xhist[2, :])
axes[1].set_xlabel("X [km]")
axes[1].set_ylabel("Z [km]")
axes[1].set_title("Side View (XZ)")
axes[1].set_aspect('equal')
axes[1].grid(True)

plt.tight_layout()
#plt.show()

t = np.linspace(0, tf, n)

fig2, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(t, xhist[0, :])
axes[0].set_ylabel("X [km]")

axes[1].plot(t, xhist[1, :])
axes[1].set_ylabel("Y [km]")

axes[2].plot(t, xhist[2, :])
axes[2].set_ylabel("Z [km]")
axes[2].set_xlabel("Time [s]")

for ax in axes:
    ax.grid(True)

plt.tight_layout()
plt.show()


# plot result in 3D (ideally interactive plot with a sphere world model?)
# TODO: ideally either also save a video or have it play the animation while still letting us move
# i think meshcat let us do that. see if pyvista lets us do that

# testing my sim code
print(f'final R is {xhist[0:3, -1]}')
print(f'final V is {xhist[3:6, -1]}')



 '''