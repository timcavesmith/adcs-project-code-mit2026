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



def dynamics()
    our DE

n_steps = sim_time / dt 

def rk4(xdot, x, x0, dt, n_steps)
    for i in range(n_steps)
        the rk4 steps


a COE to RV and RV to COEs might be useful for easy initialization and cool output for intuition




plotting and visualization:
- 2D overhead view of an equatorial orbit (perhaps with some eccentricity)
- 
- (opt) 3D visualization with an Earth in the background 


future features:


'''

