# Environmental torques: gravity gradient and atmospheric drag.
# Both expressions are taken from HW4 part 2 of the writeup; the drag formula
# follows Vallado, Fundamentals of Astrodynamics and Applications.

import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from hw1.attitude_dynamics import quat_to_rotmat

MU_EARTH = 398600.4418           # km^3 / s^2
OMEGA_EARTH = 7.2921159e-5       # rad / s
R_EARTH = 6378.137               # km, WGS-84 equatorial radius

# Vallado table 8-4, 400-450 km altitude band.
H0_EXP, RHO0_EXP, H_EXP = 400.0, 3.725e-12, 58.515   # km, kg/m^3, km


def density_exp(altitude_km):
    """Exponential atmosphere density (kg/m^3)."""
    return RHO0_EXP * np.exp(-(altitude_km - H0_EXP) / H_EXP)


def gravity_gradient_torque(r_eci_km, q, J):
    """tau_gg = (3 mu / |r|^5) (r_body x J r_body), in body-frame Nm.
    r_eci_km is converted to metres so J and the output are SI."""
    r_body = quat_to_rotmat(q).T @ (r_eci_km * 1000.0)
    r_norm = np.linalg.norm(r_body)
    return (3.0 * MU_EARTH * 1e9 / r_norm**5) * np.cross(r_body, J @ r_body)


def drag_torque(r_eci_km, v_eci_kms, q, surfaces, Cd=2.2):
    """Sum of flat-plate drag torques over the surface list. Each entry has
    keys 'a' (m^2), 'n_body' (unit normal), 'r_body' (centroid in metres).
    Surfaces with n . v_hat <= 0 are shadowed and contribute nothing."""
    altitude_km = np.linalg.norm(r_eci_km) - R_EARTH
    rho = density_exp(altitude_km)

    omega_e = np.array([0.0, 0.0, OMEGA_EARTH])
    v_rel_eci = v_eci_kms * 1000.0 - np.cross(omega_e, r_eci_km * 1000.0)
    v_rel_body = quat_to_rotmat(q).T @ v_rel_eci
    v_mag = np.linalg.norm(v_rel_body)
    if v_mag < 1e-9:
        return np.zeros(3)
    v_hat = v_rel_body / v_mag

    q_dyn = 0.5 * Cd * rho * v_mag**2
    tau = np.zeros(3)
    for s in surfaces:
        proj = s['n_body'] @ v_hat
        if proj <= 0.0:
            continue
        force = q_dyn * s['a'] * proj * v_hat
        tau += np.cross(s['r_body'], force)
    return tau


# Six-face bounding-box approximation of Dream Chaser (length 9 m, wingspan
# 7 m, height 2 m -- Sierra Space cargo variant). Each face area is 50% of
# its bounding-box area to reflect the lifting-body profile, with a 0.3 m
# offset of the centre of pressure from the geometric centre to give a
# non-trivial moment arm. This is an order-of-magnitude model.
_LX, _LY, _LZ = 9.0, 7.0, 2.0
_FACE_FRAC = 0.5
_OFF = 0.3
DEFAULT_SURFACES = [
    dict(a=_FACE_FRAC * _LY * _LZ, n_body=np.array([ 1.0, 0, 0]), r_body=np.array([ _LX/2,  _OFF, 0])),
    dict(a=_FACE_FRAC * _LY * _LZ, n_body=np.array([-1.0, 0, 0]), r_body=np.array([-_LX/2,  _OFF, 0])),
    dict(a=_FACE_FRAC * _LX * _LZ, n_body=np.array([0,  1.0, 0]), r_body=np.array([0,  _LY/2,  _OFF])),
    dict(a=_FACE_FRAC * _LX * _LZ, n_body=np.array([0, -1.0, 0]), r_body=np.array([0, -_LY/2,  _OFF])),
    dict(a=_FACE_FRAC * _LX * _LY, n_body=np.array([0, 0,  1.0]), r_body=np.array([_OFF, 0,  _LZ/2])),
    dict(a=_FACE_FRAC * _LX * _LY, n_body=np.array([0, 0, -1.0]), r_body=np.array([_OFF, 0, -_LZ/2])),
]


if __name__ == "__main__":
    from hw1.attitude_dynamics import J
    r = np.array([6800.0, 0, 0])
    v = np.array([0, np.sqrt(MU_EARTH / 6800.0), 0])
    q = np.array([1.0, 0, 0, 0])
    print(f"|tau_grav| = {np.linalg.norm(gravity_gradient_torque(r, q, J)):.4e} Nm")
    print(f"|tau_drag| = {np.linalg.norm(drag_torque(r, v, q, DEFAULT_SURFACES)):.4e} Nm")
    print(f"density at 422 km: {density_exp(422):.3e} kg/m^3")
