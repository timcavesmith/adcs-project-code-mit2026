import numpy as np
from utils import *

def getGravityGradientTorque(mu, J, r):
    """Gravity gradient torque
    mu: gravitational parameter of central body
    J: inertia matrix of satellite
    r: position vector of satellite in inertial frame"""
    r = r.reshape(3,)
    return 3 * mu / (r.T @ r)**(5/2) * np.cross(r, J @ r)

def getDragTorque(Cd, v_eci, r_eci, q, surface_positions, surface_normals, surface_areas):
    '''Get drag torque using exponential atmosphere model and surface normal aligned opposite velocity.'''
    v_eci = np.asarray(v_eci, dtype=float).reshape(3,)
    r_eci = np.asarray(r_eci, dtype=float).reshape(3,)
    q = np.asarray(q, dtype=float).reshape(4,)

    earth_rotation_rate_rad_s = np.array([0, 0, 7.2921150e-5])
    v_rel_eci = v_eci - np.cross(earth_rotation_rate_rad_s, r_eci)
    v_rel_body = Q(q).T @ v_rel_eci

    speed = np.linalg.norm(v_rel_eci)
    if speed <= 0.0:
        return np.zeros(3)

    altitude_km = np.linalg.norm(r_eci) - 6378.137
    rho = atmosphericDensityExponential(
        altitude_km,
        rho_ref=3.725e-12,
        h_ref_km=400,
        scale_height_km=58.515,
    )

    return getDragTorqueFromSurfaces(
        Cd=Cd,
        r_eci=r_eci,
        v_rel_body=v_rel_body,
        surface_positions=surface_positions,
        surface_normals=surface_normals,
        surface_areas=surface_areas,
        rho=rho,
    )


def atmosphericDensityExponential(
    altitude_km,
    rho_ref,
    h_ref_km,
    scale_height_km,
):
    """Exponential atmosphere model.

    rho(alt) = rho_ref * exp(-(alt - h_ref) / H)
    Use parameters from Vallado, "Fundamentals of Astrodynamics and Applications" (Table 8-4)

    Args:
        altitude_km: Geometric altitude above reference radius [km].
        rho_ref: Reference density at h_ref [kg/m^3].
        h_ref_km: Reference altitude [km].
        scale_height_km: Atmospheric scale height [km].

    Returns:
        Atmospheric density [kg/m^3].
    """
    altitude_km = float(max(0.0, altitude_km))
    return rho_ref * np.exp(-(altitude_km - h_ref_km) / scale_height_km)


def getDragTorqueFromSurfaces(
    Cd,
    r_eci,
    v_rel_body,
    surface_positions,
    surface_normals,
    surface_areas,
    rho,
    earth_radius_km=6378.137,
):
    """Compute total aerodynamic drag torque from multiple surfaces.

    Args:
        Cd: Drag coefficient (scalar).
        r_eci: Spacecraft position vector in inertial frame [km].
        v_rel_body: Relative velocity wrt atmosphere in body frame [km/s].
        surface_positions: (N,3) lever-arm vectors from COM to each surface [m].
        surface_normals: (N,3) outward unit normals for each surface.
        surface_areas: (N,) panel areas [m^2].
        rho: Atmospheric density at current altitude [kg/m^3].
        earth_radius_km: Reference planetary radius [km].
        

    Returns:
        Total drag torque vector [N*m], summed over all surfaces.
    """
    r_eci = np.asarray(r_eci, dtype=float).reshape(3,)
    v_rel_body = np.asarray(v_rel_body, dtype=float).reshape(3,)
    surface_positions = np.asarray(surface_positions, dtype=float)
    surface_normals = np.asarray(surface_normals, dtype=float)
    surface_areas = np.asarray(surface_areas, dtype=float).reshape(-1,)

    if surface_positions.ndim != 2 or surface_positions.shape[1] != 3:
        raise ValueError("surface_positions must have shape (N, 3).")
    if surface_normals.ndim != 2 or surface_normals.shape[1] != 3:
        raise ValueError("surface_normals must have shape (N, 3).")
    if surface_positions.shape[0] != surface_normals.shape[0] or surface_positions.shape[0] != surface_areas.shape[0]:
        raise ValueError("surface_positions, surface_normals, and surface_areas must have matching length N.")

    speed_kms = np.linalg.norm(v_rel_body)
    if speed_kms <= 0.0:
        return np.zeros(3)

    altitude_km = np.linalg.norm(r_eci) - earth_radius_km
    air_speed = np.linalg.norm(v_rel_body) * 1000.0 # convert km/s to m/s
    v_hat = v_rel_body / speed_kms
    flow_dir = -v_hat
    dynamic_pressure = 0.5 * rho * air_speed * air_speed

    torque_total = np.zeros(3)
    for r_panel, n_panel, area in zip(surface_positions, surface_normals, surface_areas):
        n_norm = np.linalg.norm(n_panel)
        if n_norm <= 0.0 or area <= 0.0:
            continue

        n_hat = n_panel / n_norm
        cos_incidence = np.dot(n_hat, flow_dir)
        if cos_incidence <= 0.0:
            # Surface is facing away from the flow.
            continue

        # Projected-area model: F = q * Cd * A_proj * flow_dir.
        F_drag = dynamic_pressure * Cd * area * cos_incidence * flow_dir
        torque_total += np.cross(r_panel, F_drag)

    return torque_total