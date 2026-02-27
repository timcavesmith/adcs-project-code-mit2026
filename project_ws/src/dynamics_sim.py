# feb 26 2026
# 
# 2-body EOMs for 3-DOF sim
#

'''
ADCS HW 1: 

Part 4) 3-DOF 3D orbital dynamics sim 

- ideally later we can expand it to be 6-DOF 3D with attitude dynamics

packages:
- use matplotlib for basic output plots
- perhaps try pyvista for attitude dynamics modeling later (whatever david used)


units:
- km
- seconds

frame: ECI (earth-centered inertial)

PLAN:

init params:
R, V
celestial body mass and radius



def dynamics(x)
    our DE that takes in x and returns xdot (called 4x for each rk4 step)
    return xdot

n_steps = sim_time / dt 

def rk4(xdot, x, x0, dt, n_steps)
    for i in range(n_steps)
        the rk4 steps

^ will just do rk4step then handle the integration externally in a loop
also using h rather than dt since h is also done to the state


a COE to RV and RV to COEs might be useful for easy initialization and cool output for intuition




plotting and visualization:
- 2D overhead view of an equatorial orbit (perhaps with some eccentricity)
- 
- (opt) 3D visualization with an Earth in the background 


future features:
- (opt) using dataclasses for parameters (like matlab struct) to give dot acacess to attributes

'''

import numpy as np
import matplotlib.pyplot as plt
# set params

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



