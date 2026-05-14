# Minimum-time, minimum-energy slew OCP via direct trapezoidal collocation.
# CasADi builds the NLP, IPOPT solves it. State is (q, omega, w) with w the
# per-wheel momentum (4,), control is u = wheel torques (4,). Sign convention: rho = B_W w, and the
# body sees +B_W u inside the omega-dot bracket.

import numpy as np
import casadi as ca


def _hat(v):
    return ca.vertcat(
        ca.horzcat(    0, -v[2],  v[1]),
        ca.horzcat( v[2],     0, -v[0]),
        ca.horzcat(-v[1],  v[0],     0),
    )


def _G(q):
    """qdot = 0.5 G(q) omega.  G(q) = L(q) H = [-v^T; sI + hat(v)]."""
    s, v = q[0], q[1:4]
    return ca.vertcat(-v.T, s * ca.MX.eye(3) + _hat(v))


def _L(q):
    s, v = q[0], q[1:4]
    return ca.vertcat(
        ca.horzcat(s, -v.T),
        ca.horzcat(v,  s * ca.MX.eye(3) + _hat(v)),
    )


def _rhs(q, omega, w, u, J, J_inv, B_W):
    """Gyrostat RHS, no tau_ext (planning model). The OCP is fed reference
    trajectories that the HW4 PD then tracks against truth-plus-disturbances."""
    rho = B_W @ w
    qdot = 0.5 * _G(q) @ omega
    omega_dot = -J_inv @ (ca.cross(omega, J @ omega + rho) + B_W @ u)
    return qdot, omega_dot, u


def solve_ocp(q0, q_f, J, B_W,
              u_max, w_max, omega_max,
              N=40, w_t=1.0,
              R_diag=None, Q_diag=None,
              t_f_init=1800.0, t_f_min=300.0, t_f_max=3000.0,
              q_init=None, omega_init=None, w_init=None, u_init=None,
              verbose=False):
    """Direct trapezoidal collocation. Decision: x_k for k=0..N, u_k for
    k=0..N, plus the free maneuver time t_f. Returns the optimal state /
    control schedule on the uniform grid t_k = k * (t_f/N)."""
    n_q, n_om, n_w, n_u = 4, 3, 4, 4
    if R_diag is None:
        R_diag = np.ones(n_u)
    if Q_diag is None:
        Q_diag = np.ones(n_w)

    opti = ca.Opti()
    Qv = opti.variable(n_q, N + 1)
    Ov = opti.variable(n_om, N + 1)
    Wv = opti.variable(n_w, N + 1)
    Uv = opti.variable(n_u, N + 1)
    tf = opti.variable()

    J_c = ca.DM(J)
    J_inv_c = ca.DM(np.linalg.inv(J))
    B_c = ca.DM(B_W)
    R = ca.diag(ca.DM(R_diag))
    Q = ca.diag(ca.DM(Q_diag))

    # Boundary conditions
    opti.subject_to(Qv[:, 0] == ca.DM(q0))
    opti.subject_to(Ov[:, 0] == ca.DM.zeros(n_om))
    opti.subject_to(Wv[:, 0] == ca.DM.zeros(n_w))
    # vec(L(q_f)^T q(N)) = 0 -- handles the q vs -q double cover automatically
    q_err = (_L(ca.DM(q_f)).T @ Qv[:, N])[1:4]
    opti.subject_to(q_err == 0)
    opti.subject_to(Ov[:, N] == 0)

    dt = tf / N
    cost = w_t * tf
    for k in range(N):
        f0 = _rhs(Qv[:, k    ], Ov[:, k    ], Wv[:, k    ], Uv[:, k    ],
                  J_c, J_inv_c, B_c)
        f1 = _rhs(Qv[:, k + 1], Ov[:, k + 1], Wv[:, k + 1], Uv[:, k + 1],
                  J_c, J_inv_c, B_c)
        opti.subject_to(Qv[:, k + 1] - Qv[:, k] == 0.5 * dt * (f0[0] + f1[0]))
        opti.subject_to(Ov[:, k + 1] - Ov[:, k] == 0.5 * dt * (f0[1] + f1[1]))
        opti.subject_to(Wv[:, k + 1] - Wv[:, k] == 0.5 * dt * (f0[2] + f1[2]))

        L0 = Uv[:, k    ].T @ R @ Uv[:, k    ] + Wv[:, k    ].T @ Q @ Wv[:, k    ]
        L1 = Uv[:, k + 1].T @ R @ Uv[:, k + 1] + Wv[:, k + 1].T @ Q @ Wv[:, k + 1]
        cost = cost + 0.5 * dt * (L0 + L1)

    # Path constraints. The continuous-time kinematics qdot = 1/2 G(q) omega
    # preserve |q|=1, and trapezoidal drift on a smooth profile is small. We
    # enforce |q|=1 only at the terminal node and renormalize after the solve.
    opti.subject_to(ca.dot(Qv[:, N], Qv[:, N]) == 1)
    for k in range(N + 1):
        opti.subject_to(opti.bounded(-u_max, Uv[:, k], u_max))
        opti.subject_to(opti.bounded(-w_max, Wv[:, k], w_max))
        opti.subject_to(ca.dot(Ov[:, k], Ov[:, k]) <= omega_max**2)

    opti.subject_to(opti.bounded(t_f_min, tf, t_f_max))
    opti.minimize(cost)

    # Initial guesses
    if q_init is not None:
        opti.set_initial(Qv, q_init)
    else:
        opti.set_initial(Qv, np.tile(np.asarray(q0).reshape(-1, 1), (1, N + 1)))
    if omega_init is not None:
        opti.set_initial(Ov, omega_init)
    if w_init is not None:
        opti.set_initial(Wv, w_init)
    if u_init is not None:
        opti.set_initial(Uv, u_init)
    opti.set_initial(tf, t_f_init)

    s_opts = dict(max_iter=3000, tol=1e-6, mu_strategy='adaptive',
                  print_level=(5 if verbose else 0))
    p_opts = dict(print_time=int(verbose))
    opti.solver('ipopt', p_opts, s_opts)
    sol = opti.solve()

    tf_val = float(sol.value(tf))
    return dict(q=sol.value(Qv), omega=sol.value(Ov),
                w=sol.value(Wv), u=sol.value(Uv),
                t_f=tf_val, t=np.linspace(0.0, tf_val, N + 1),
                cost=float(sol.value(cost)), N=N)


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from hw3.utils import expq
    from hw1.attitude_dynamics import J as J_dc
    from hw4.actuators import B_W, WHEEL_TORQUE_MAX, WHEEL_MOMENTUM_MAX

    axis = np.array([1.0, 0.5, -0.3]); axis /= np.linalg.norm(axis)
    q0 = np.array([1.0, 0, 0, 0])
    qf = expq(0.5 * np.pi * axis)
    sol = solve_ocp(q0, qf, J_dc, B_W,
                    u_max=WHEEL_TORQUE_MAX, w_max=WHEEL_MOMENTUM_MAX,
                    omega_max=np.radians(0.3),
                    N=40, w_t=1.0,
                    R_diag=35.0 * np.ones(4),
                    Q_diag=1e-4 * np.ones(4),
                    verbose=True)
    print(f"t_f* = {sol['t_f']:.2f} s")
    print(f"cost = {sol['cost']:.4f}")
    print(f"peak |u| = {np.max(np.abs(sol['u'])):.4f}")
    print(f"peak |w| = {np.max(np.abs(sol['w'])):.4f}")
    print(f"peak |omega| (deg/s) = "
          f"{np.degrees(np.max(np.linalg.norm(sol['omega'], axis=0))):.4f}")
