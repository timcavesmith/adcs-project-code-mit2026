"""
3D Interactive Orbit Plot with PyVista
- Textured Earth sphere (with actual Earth texture if available)
- Click-drag rotation/zoom (native VTK interactor)
- Animated spacecraft marker
"""

import numpy as np
import pyvista as pv
from pathlib import Path
import os
os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"


# ── Fake orbit data for demo (replace with xhist) ────────────────
n = 500
R_earth = 6371.0  # km

theta = np.linspace(0, 2 * np.pi, n)
r_orbit = 7000.0
inc = np.radians(45)

x_orb = r_orbit * np.cos(theta)
y_orb = r_orbit * np.sin(theta) * np.cos(inc)
z_orb = r_orbit * np.sin(theta) * np.sin(inc)

# Stack into (n, 3)
orbit_pts = np.column_stack([x_orb, y_orb, z_orb])


# ── Earth sphere ──────────────────────────────────────────────────────
earth = pv.Sphere(radius=R_earth, theta_resolution=60, phi_resolution=60)
earth = earth.texture_map_to_sphere()

# If texture image (NASA Blue Marble):
#   texture = pv.read_texture("earth_texture.jpg")
# Otherwise color it blue-green
has_texture = False
texture_path = Path("earth_texture.jpg")
if texture_path.exists():
    texture = pv.read_texture(str(texture_path))
    has_texture = True


# ── Orbit path as a spline ───────────────────────────────────────────
orbit_line = pv.Spline(orbit_pts, n_points=n)


# ── Static plot (interactive rotation built-in) ──────────────────────
def plot_static():
    """Just the orbit + Earth, with mouse rotation/zoom."""
    pl = pv.Plotter()
    pl.set_background('black')

    if has_texture:
        pl.add_mesh(earth, texture=texture)
    else:
        pl.add_mesh(earth, color='steelblue', opacity=0.9)

    pl.add_mesh(orbit_line, color='white', line_width=3, label='Orbit')

    # Starting point
    pl.add_points(
        orbit_pts[0:1],
        color='red',
        point_size=15,
        render_points_as_spheres=True,
        label='Start',
    )

    pl.add_legend()
    pl.add_axes()
    pl.show()


# ── Animated plot (spacecraft moves along orbit) ─────────────────────
def plot_animated(speed=5):
    """
    Animate the spacecraft along the orbit.
    Can rotate/zoom WHILE the animation plays.
    
    Args:
        speed: plot every `speed`-th point (higher = faster)
    """
    pl = pv.Plotter()
    pl.set_background('black')

    # Earth
    if has_texture:
        pl.add_mesh(earth, texture=texture)
    else:
        pl.add_mesh(earth, color='steelblue', opacity=0.9)

    # Orbit path
    pl.add_mesh(orbit_line, color='white', line_width=3)

    # Spacecraft marker — initialize with first point
    sc_point = pv.PolyData(orbit_pts[0:1])
    sc_actor = pl.add_points(
        sc_point,
        color='red',
        point_size=20,
        render_points_as_spheres=True,
    )

    # Trail (shows where spacecraft has been)
    trail_pts = pv.PolyData(orbit_pts[0:1])
    trail_actor = pl.add_points(
        trail_pts,
        color='yellow',
        point_size=3,
        render_points_as_spheres=True,
        opacity=0.5,
    )

    pl.add_axes()
    pl.show(interactive_update=True, auto_close=False)

    # Animation loop — can rotate the view during this
    for i in range(0, n, speed):
        # Update spacecraft position
        sc_point.points = orbit_pts[i:i+1]

        # Update trail
        trail_pts.points = orbit_pts[:i+1]

        pl.update()

        # Small sleep to control animation speed
        import time
        time.sleep(0.01)

    # Keep window open after animation finishes
    pl.show()


# ── Run ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Choose one:
    # plot_static()
    plot_animated(speed=3)
