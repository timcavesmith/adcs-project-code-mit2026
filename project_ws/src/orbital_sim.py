# feb 26 2026
# 
# 2-body EOMs for 3-DOF sim
#

'''
ADCS HW 1: 

Part 4) 3-DOF 3D orbital dynamics sim 
'''

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ── Constants ─────────────────────────────────────────────────────────
mu = 398600.44        # [km^3/s^2]
r_earth = 6378.137    # [km] equatorial radius


# ── COE <-> RV Conversions ────────────────────────────────────────────

def coe2rv(a, e, i, RAAN, omega, nu, mu=mu):
    """
    Classical Orbital Elements to ECI position & velocity.
    
    Args:
        a:     semi-major axis [km]
        e:     eccentricity [-]
        i:     inclination [rad]
        RAAN:  right ascension of ascending node [rad]
        omega: argument of periapsis [rad]
        nu:    true anomaly [rad]
        mu:    gravitational parameter [km^3/s^2]
    
    Returns:
        r_eci: position vector (3,) [km]
        v_eci: velocity vector (3,) [km/s]
    """
    p = a * (1 - e**2)  # semi-latus rectum
    r_mag = p / (1 + e * np.cos(nu))

    # position and velocity in perifocal frame (PQW)
    r_pqw = np.array([r_mag * np.cos(nu),
                       r_mag * np.sin(nu),
                       0.0])

    v_pqw = np.sqrt(mu / p) * np.array([-np.sin(nu),
                                          e + np.cos(nu),
                                          0.0])

    # rotation matrix: perifocal -> ECI (3-1-3: RAAN, i, omega)
    cO = np.cos(RAAN); sO = np.sin(RAAN)
    ci = np.cos(i);     si = np.sin(i)
    cw = np.cos(omega); sw = np.sin(omega)

    R_pqw2eci = np.array([
        [cO*cw - sO*sw*ci,  -cO*sw - sO*cw*ci,   sO*si],
        [sO*cw + cO*sw*ci,  -sO*sw + cO*cw*ci,  -cO*si],
        [sw*si,              cw*si,                ci   ],
    ])

    r_eci = R_pqw2eci @ r_pqw
    v_eci = R_pqw2eci @ v_pqw

    return r_eci, v_eci


def rv2coe(r_vec, v_vec, mu=mu):
    """
    ECI position & velocity to Classical Orbital Elements.
    
    Args:
        r_vec: position vector (3,) [km]
        v_vec: velocity vector (3,) [km/s]
        mu:    gravitational parameter [km^3/s^2]
    
    Returns:
        dict with keys: a, e, i, RAAN, omega, nu, T (period)
        Angles in [rad], a in [km], T in [s]
    """
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    # angular momentum
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)

    # node vector
    K = np.array([0, 0, 1.0])
    n_vec = np.cross(K, h_vec)
    n = np.linalg.norm(n_vec)

    # eccentricity vector
    e_vec = (1/mu) * ((v**2 - mu/r) * r_vec - np.dot(r_vec, v_vec) * v_vec)
    e = np.linalg.norm(e_vec)

    # specific energy -> semi-major axis
    energy = v**2 / 2 - mu / r
    a = -mu / (2 * energy)

    # inclination
    i = np.arccos(np.clip(h_vec[2] / h, -1, 1))

    # RAAN
    if n > 1e-12:
        RAAN = np.arccos(np.clip(n_vec[0] / n, -1, 1))
        if n_vec[1] < 0:
            RAAN = 2*np.pi - RAAN
    else:
        RAAN = 0.0  # equatorial orbit

    # argument of periapsis
    if n > 1e-12 and e > 1e-12:
        omega = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n * e), -1, 1))
        if e_vec[2] < 0:
            omega = 2*np.pi - omega
    else:
        omega = 0.0

    # true anomaly
    if e > 1e-12:
        nu = np.arccos(np.clip(np.dot(e_vec, r_vec) / (e * r), -1, 1))
        if np.dot(r_vec, v_vec) < 0:
            nu = 2*np.pi - nu
    else:
        nu = 0.0

    # period
    T = 2 * np.pi * np.sqrt(a**3 / mu)

    return dict(a=a, e=e, i=i, RAAN=RAAN, omega=omega, nu=nu, T=T)


def coe_string(coe):
    """Pretty-print COEs for plot annotations."""
    return (
        f"a = {coe['a']:.2f} km\n"
        f"e = {coe['e']:.6f}\n"
        f"i = {np.degrees(coe['i']):.2f}°\n"
        f"Ω = {np.degrees(coe['RAAN']):.2f}°\n"
        f"ω = {np.degrees(coe['omega']):.2f}°\n"
        f"ν = {np.degrees(coe['nu']):.2f}°\n"
        f"T = {coe['T']:.1f} s"
    )


def rv_string(r, v):
    """Pretty-print R,V for plot annotations."""
    return (
        f"r = [{r[0]:.2f}, {r[1]:.2f}, {r[2]:.2f}] km\n"
        f"v = [{v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}] km/s\n"
        f"|r| = {np.linalg.norm(r):.2f} km\n"
        f"|v| = {np.linalg.norm(v):.4f} km/s"
    )


# ── Dynamics & RK4 ───────────────────────────────────────────────────

def dynamics(x):
    r = x[0:3]
    v = x[3:6]
    rdot = v
    r_norm = np.linalg.norm(r)
    vdot = -mu / r_norm**3 * r
    xdot = np.concatenate([rdot, vdot])
    return xdot


def rk4step(x, h):
    k1 = dynamics(x)
    k2 = dynamics(x + k1 * h/2)
    k3 = dynamics(x + k2 * h/2)
    k4 = dynamics(x + h * k3)
    xn = x + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    return xn


# ══════════════════════════════════════════════════════════════════════
# SETUP: Define orbit via COEs
# ══════════════════════════════════════════════════════════════════════

# Example: 45° inclined, slightly eccentric orbit
a     = 8000.0              # [km] semi-major axis
e     = 0.1                 # eccentricity
i     = np.radians(45)      # inclination
RAAN  = np.radians(30)      # right ascension of ascending node
omega = np.radians(60)      # argument of periapsis
nu    = np.radians(0)       # true anomaly at epoch

# Convert to R, V
r0, v0 = coe2rv(a, e, i, RAAN, omega, nu)

# Verify round-trip
coe0 = rv2coe(r0, v0)
T = coe0['T']

print("=== Initial Conditions ===")
print(f"r0 = {r0} km")
print(f"v0 = {v0} km/s")
print(f"Period = {T:.2f} s ({T/60:.2f} min)")
print(f"\nCOEs (round-trip check):")
for k, val in coe0.items():
    if k in ['i', 'RAAN', 'omega', 'nu']:
        print(f"  {k:>5s} = {np.degrees(val):.4f}°")
    else:
        print(f"  {k:>5s} = {val:.6f}")


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

# Final state COEs
r_final = xhist[0:3, -1]
v_final = xhist[3:6, -1]
coe_final = rv2coe(r_final, v_final)

print(f"\n=== Final State (t = {tf:.1f} s) ===")
print(rv_string(r_final, v_final))


# ══════════════════════════════════════════════════════════════════════
# 2D MATPLOTLIB PLOTS
# ══════════════════════════════════════════════════════════════════════

# Draw Earth circle helper
earth_theta = np.linspace(0, 2*np.pi, 200)
earth_x = r_earth * np.cos(earth_theta)
earth_y = r_earth * np.sin(earth_theta)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ── Top-down view (XY) ───────────────────────────────────────────────
axes[0].fill(earth_x, earth_y, color='steelblue', alpha=0.5, label='Earth')
axes[0].plot(xhist[0, :], xhist[1, :], 'w', linewidth=1.5)
axes[0].plot(xhist[0, 0], xhist[1, 0], 'go', markersize=8, label='Start')
axes[0].plot(xhist[0, -1], xhist[1, -1], 'r^', markersize=8, label='End')
axes[0].set_xlabel("X [km]")
axes[0].set_ylabel("Y [km]")
axes[0].set_title("Top Down (XY)")
axes[0].set_aspect('equal')
axes[0].grid(True, alpha=0.3)
axes[0].set_facecolor('black')
axes[0].legend(loc='upper right', fontsize=8)

# Annotate with COEs and RV on the left plot
info_text = "Initial State:\n" + rv_string(r0, v0) + "\n\n" + coe_string(coe0)
axes[0].text(0.02, 0.98, info_text, transform=axes[0].transAxes,
             fontsize=7, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.8),
             color='white')

# ── Side view (XZ) ───────────────────────────────────────────────────
axes[1].fill(earth_x, earth_y, color='steelblue', alpha=0.5, label='Earth')
axes[1].plot(xhist[0, :], xhist[2, :], 'w', linewidth=1.5)
axes[1].plot(xhist[0, 0], xhist[2, 0], 'go', markersize=8, label='Start')
axes[1].plot(xhist[0, -1], xhist[2, -1], 'r^', markersize=8, label='End')
axes[1].set_xlabel("X [km]")
axes[1].set_ylabel("Z [km]")
axes[1].set_title("Side View (XZ)")
axes[1].set_aspect('equal')
axes[1].grid(True, alpha=0.3)
axes[1].set_facecolor('black')
axes[1].legend(loc='upper right', fontsize=8)

# Annotate with final state on the right plot
final_text = "Final State:\n" + rv_string(r_final, v_final) + "\n\n" + coe_string(coe_final)
axes[1].text(0.02, 0.98, final_text, transform=axes[1].transAxes,
             fontsize=7, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.8),
             color='white')

plt.tight_layout()
plt.savefig("orbit_2d.png", dpi=200, bbox_inches='tight')
plt.show()


# ── Position components vs time ───────────────────────────────────────
fig2, axes2 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes2[0].plot(t, xhist[0, :], 'c')
axes2[0].set_ylabel("X [km]")

axes2[1].plot(t, xhist[1, :], 'm')
axes2[1].set_ylabel("Y [km]")

axes2[2].plot(t, xhist[2, :], 'y')
axes2[2].set_ylabel("Z [km]")
axes2[2].set_xlabel("Time [s]")

for ax in axes2:
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('black')
    ax.tick_params(colors='white')

fig2.patch.set_facecolor('black')
for ax in axes2:
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')

plt.tight_layout()
plt.savefig("orbit_time.png", dpi=200, bbox_inches='tight')
plt.show()


# ══════════════════════════════════════════════════════════════════════
# 3D PLOTLY INTERACTIVE PLOT
# ══════════════════════════════════════════════════════════════════════

def make_earth_sphere(R=r_earth, N=50):
    phi = np.linspace(0, np.pi, N)
    theta = np.linspace(0, 2 * np.pi, N)
    phi, theta = np.meshgrid(phi, theta)
    xe = R * np.sin(phi) * np.cos(theta)
    ye = R * np.sin(phi) * np.sin(theta)
    ze = R * np.cos(phi)
    return xe, ye, ze


xe, ye, ze = make_earth_sphere()

fig3d = go.Figure()

# Earth
fig3d.add_trace(go.Surface(
    x=xe, y=ye, z=ze,
    colorscale=[[0, 'rgb(30, 80, 160)'], [1, 'rgb(30, 130, 76)']],
    showscale=False, opacity=0.9,
    name='Earth', hoverinfo='skip',
))

# Full orbit path
fig3d.add_trace(go.Scatter3d(
    x=xhist[0, :], y=xhist[1, :], z=xhist[2, :],
    mode='lines',
    line=dict(color='white', width=3),
    name='Orbit', hoverinfo='skip',
))

# Spacecraft marker (animated)
fig3d.add_trace(go.Scatter3d(
    x=[xhist[0, 0]], y=[xhist[1, 0]], z=[xhist[2, 0]],
    mode='markers',
    marker=dict(size=6, color='red'),
    name='Spacecraft',
))

# Start/end markers
fig3d.add_trace(go.Scatter3d(
    x=[xhist[0, 0]], y=[xhist[1, 0]], z=[xhist[2, 0]],
    mode='markers', marker=dict(size=5, color='lime', symbol='diamond'),
    name='Start',
))
fig3d.add_trace(go.Scatter3d(
    x=[xhist[0, -1]], y=[xhist[1, -1]], z=[xhist[2, -1]],
    mode='markers', marker=dict(size=5, color='orange', symbol='diamond'),
    name='End',
))

# ── Animation frames ─────────────────────────────────────────────────
frame_step = max(1, n_steps // 200)  # ~200 frames total
frame_indices = list(range(0, n_steps, frame_step))

frames = []
for idx in frame_indices:
    r_i = xhist[0:3, idx]
    v_i = xhist[3:6, idx]
    coe_i = rv2coe(r_i, v_i)
    tof = t[idx]

    hover_text = (
        f"t = {tof:.1f} s<br>"
        f"r = [{r_i[0]:.1f}, {r_i[1]:.1f}, {r_i[2]:.1f}] km<br>"
        f"v = [{v_i[0]:.3f}, {v_i[1]:.3f}, {v_i[2]:.3f}] km/s<br>"
        f"|r| = {np.linalg.norm(r_i):.1f} km<br>"
        f"ν = {np.degrees(coe_i['nu']):.1f}°"
    )

    frames.append(go.Frame(
        data=[
            go.Surface(x=xe, y=ye, z=ze),
            go.Scatter3d(x=xhist[0, :], y=xhist[1, :], z=xhist[2, :]),
            go.Scatter3d(
                x=[xhist[0, idx]], y=[xhist[1, idx]], z=[xhist[2, idx]],
                hovertext=[hover_text], hoverinfo='text',
            ),
            go.Scatter3d(x=[xhist[0, 0]], y=[xhist[1, 0]], z=[xhist[2, 0]]),
            go.Scatter3d(x=[xhist[0, -1]], y=[xhist[1, -1]], z=[xhist[2, -1]]),
        ],
        name=str(idx),
    ))

fig3d.frames = frames

# ── COE annotation in the 3D scene ───────────────────────────────────
annotation_text = (
    f"a={coe0['a']:.1f}km  e={coe0['e']:.4f}  "
    f"i={np.degrees(coe0['i']):.1f}°  Ω={np.degrees(coe0['RAAN']):.1f}°  "
    f"ω={np.degrees(coe0['omega']):.1f}°  ν₀={np.degrees(coe0['nu']):.1f}°"
)

# ── Play/Pause + slider ──────────────────────────────────────────────
fig3d.update_layout(
    updatemenus=[dict(
        type='buttons', showactive=False,
        x=0.05, y=0.05,
        buttons=[
            dict(label='▶ Play', method='animate',
                 args=[None, dict(frame=dict(duration=30, redraw=True),
                                  fromcurrent=True, mode='immediate')]),
            dict(label='⏸ Pause', method='animate',
                 args=[[None], dict(frame=dict(duration=0, redraw=False),
                                    mode='immediate')]),
        ],
    )],
    sliders=[dict(
        active=0,
        steps=[
            dict(args=[[str(idx)], dict(frame=dict(duration=30, redraw=True),
                                         mode='immediate')],
                 method='animate', label=f"{t[idx]:.0f}s")
            for idx in frame_indices
        ],
        x=0.1, len=0.8,
        currentvalue=dict(prefix='TOF: '),
    )],
)

# ── Scene styling ─────────────────────────────────────────────────────
pad = 1.3 * np.max(np.abs(xhist[0:3, :]))
axis_range = [-pad, pad]
axis_style = dict(range=axis_range, backgroundcolor='black', gridcolor='gray',
                  showbackground=True)

fig3d.update_layout(
    scene=dict(
        xaxis=dict(title='X ECI [km]', **axis_style),
        yaxis=dict(title='Y ECI [km]', **axis_style),
        zaxis=dict(title='Z ECI [km]', **axis_style),
        bgcolor='black',
        aspectmode='data',
    ),
    paper_bgcolor='black',
    font=dict(color='white'),
    title=dict(text=f'3D Orbit — {annotation_text}', font=dict(size=13)),
    width=1000, height=800,
    legend=dict(x=0.85, y=0.95),
)

fig3d.show()
# fig3d.write_html("orbit_3d.html")  # uncomment to save standalone HTML
