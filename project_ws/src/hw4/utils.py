import numpy as np

def hat(v):
    """Skew-symmetric (hat) matrix from 3-vector. hat(v) @ x == cross(v, x)."""
    return np.array([[ 0,    -v[2],  v[1]],
                     [ v[2],  0,    -v[0]],
                     [-v[1],  v[0],  0   ]])

def unhat(S):
    return 0.5*np.array([S[2,1]-S[1,2],
                        S[0,2]-S[2,0],
                        S[1,0]-S[0,1]])


H = np.vstack([np.zeros(3), np.eye(3)])   # (4x3) quaternion embedding matrix

T = np.block([[1, np.zeros(3)],
              [np.zeros((3,1)), -np.eye(3)]])

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


def Q(q):
    """Quaternion to rotation matrix (body -> inertial)."""
    return H.T @ L(q) @ R(q).T @ H

def expq(ϕ):
    θ = np.linalg.norm(ϕ)
    
    #naive way with divide-by-zero badness
    #r = ϕ/θ
    #[cos(θ); r*sin(θ)]

    #using sinc, we can avoid divide-by-zero issues
    return np.concatenate([[np.cos(θ)], ϕ*np.sinc(θ/np.pi)])

def logq(q):
    c = q[0]
    s = np.linalg.norm(q[1:4])
    θ = np.arctan2(s, c)
    return q[1:4]/np.sinc(θ/np.pi)

def coe2rv(a, e, i, raan, argp, nu, mu):
    """Convert classical orbital elements to inertial position and velocity."""
    p = a * (1.0 - e**2)
    r_mag = p / (1.0 + e * np.cos(nu))

    r_pqw = np.array([r_mag * np.cos(nu), r_mag * np.sin(nu), 0.0])
    v_pqw = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    c_raan = np.cos(raan)
    s_raan = np.sin(raan)
    c_i = np.cos(i)
    s_i = np.sin(i)
    c_argp = np.cos(argp)
    s_argp = np.sin(argp)

    r_pqw_to_eci = np.array(
        [
            [c_raan * c_argp - s_raan * s_argp * c_i, -c_raan * s_argp - s_raan * c_argp * c_i, s_raan * s_i],
            [s_raan * c_argp + c_raan * s_argp * c_i, -s_raan * s_argp + c_raan * c_argp * c_i, -c_raan * s_i],
            [s_argp * s_i, c_argp * s_i, c_i],
        ]
    )

    return r_pqw_to_eci @ r_pqw, r_pqw_to_eci @ v_pqw

def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if norm == 0.0:
        raise ValueError("Quaternion norm must be non-zero.")
    return q / norm