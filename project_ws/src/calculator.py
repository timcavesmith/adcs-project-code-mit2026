import numpy as np
#import matplotlib.pyplot as plt
# set params

#G = 6.674e-11 [m^3/(kg*s^2] # Universal gravitational parameter 
#M = 5.97e24 # [kg] #Earth mass
mu =  398600.44 #[km^3/s^2]  #G*M

# initialize vectors
r0 = np.array([7000, 0, 0]) # [km] shape (3,) which is 1D and shapeless, not row/col
v0 = np.array([0, 8, 0]) # [km/s] 
r_norm = np.linalg.norm(r0)
# one orbital period given semimajor axis for circular orbit
T = 2 * np.pi * np.sqrt(r_norm**3 / (mu))
print(T)

print(5828.516650846224/60)


# velocity req for a circular orbit
v = np.sqrt(mu / r_norm)
print(v)

