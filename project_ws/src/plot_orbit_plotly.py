"""
3D Interactive Orbit Plot with Plotly
- Textured Earth sphere
- Click-drag rotation/zoom
- Play/pause animation slider
"""

import numpy as np
import plotly.graph_objects as go

# ── Fake orbit data for demo (replace with your xhist) ────────────────
n = 500
mu = 3.986004418e5  # km³/s²
R_earth = 6371.0     # km

# Simple circular orbit for testing
theta = np.linspace(0, 2 * np.pi, n)
r_orbit = 7000.0  # km
inc = np.radians(45)

x_orb = r_orbit * np.cos(theta)
y_orb = r_orbit * np.sin(theta) * np.cos(inc)
z_orb = r_orbit * np.sin(theta) * np.sin(inc)


# ── Earth sphere ──────────────────────────────────────────────────────
def make_earth_sphere(R=R_earth, N=50):
    """Create a sphere mesh for Earth."""
    phi = np.linspace(0, np.pi, N)
    theta = np.linspace(0, 2 * np.pi, N)
    phi, theta = np.meshgrid(phi, theta)

    xe = R * np.sin(phi) * np.cos(theta)
    ye = R * np.sin(phi) * np.sin(theta)
    ze = R * np.cos(phi)
    return xe, ye, ze


xe, ye, ze = make_earth_sphere()

# ── Build figure ──────────────────────────────────────────────────────
fig = go.Figure()

# Earth surface
fig.add_trace(go.Surface(
    x=xe, y=ye, z=ze,
    colorscale=[[0, 'rgb(30, 80, 160)'], [1, 'rgb(30, 130, 76)']],
    showscale=False,
    opacity=0.9,
    name='Earth',
    hoverinfo='skip',
))

# Full orbit path
fig.add_trace(go.Scatter3d(
    x=x_orb, y=y_orb, z=z_orb,
    mode='lines',
    line=dict(color='white', width=3),
    name='Orbit',
    hoverinfo='skip',
))

# Spacecraft marker (will be animated)
fig.add_trace(go.Scatter3d(
    x=[x_orb[0]], y=[y_orb[0]], z=[z_orb[0]],
    mode='markers',
    marker=dict(size=6, color='red'),
    name='Spacecraft',
))

# ── Animation frames ─────────────────────────────────────────────────
# Subsample frames so it doesn't lag (every kth point)
frame_step = 5
frame_indices = list(range(0, n, frame_step))

frames = []
for i in frame_indices:
    frames.append(go.Frame(
        data=[
            go.Surface(x=xe, y=ye, z=ze),          # Earth (unchanged)
            go.Scatter3d(x=x_orb, y=y_orb, z=z_orb),  # orbit (unchanged)
            go.Scatter3d(                            # spacecraft moves
                x=[x_orb[i]],
                y=[y_orb[i]],
                z=[z_orb[i]],
            ),
        ],
        name=str(i),
    ))

fig.frames = frames

# ── Play/Pause buttons and slider ────────────────────────────────────
fig.update_layout(
    updatemenus=[dict(
        type='buttons',
        showactive=False,
        x=0.05, y=0.05,
        buttons=[
            dict(label='▶ Play',
                 method='animate',
                 args=[None, dict(
                     frame=dict(duration=30, redraw=True),
                     fromcurrent=True,
                     mode='immediate',
                 )]),
            dict(label='⏸ Pause',
                 method='animate',
                 args=[[None], dict(
                     frame=dict(duration=0, redraw=False),
                     mode='immediate',
                 )]),
        ],
    )],
    sliders=[dict(
        active=0,
        steps=[
            dict(args=[[str(i)], dict(
                frame=dict(duration=30, redraw=True),
                mode='immediate',
            )], method='animate', label=str(i))
            for i in frame_indices
        ],
        x=0.1, len=0.8,
        currentvalue=dict(prefix='Step: '),
    )],
)

# ── Scene styling ─────────────────────────────────────────────────────
axis_range = [-10000, 10000]
fig.update_layout(
    scene=dict(
        xaxis=dict(title='X ECI [km]', range=axis_range, backgroundcolor='black', gridcolor='gray'),
        yaxis=dict(title='Y ECI [km]', range=axis_range, backgroundcolor='black', gridcolor='gray'),
        zaxis=dict(title='Z ECI [km]', range=axis_range, backgroundcolor='black', gridcolor='gray'),
        bgcolor='black',
        aspectmode='data',
    ),
    paper_bgcolor='black',
    font=dict(color='white'),
    title='3D Orbit Visualization (ECI)',
    width=1000,
    height=800,
)

# ── Show or save ──────────────────────────────────────────────────────
fig.show()                         # opens in browser
# fig.write_html("orbit_3d.html")  # standalone file, great for presentations
