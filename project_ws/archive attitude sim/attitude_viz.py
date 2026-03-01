# attitude_viz.py
# Visualization for attitude dynamics sim
#
# 2D plots:  matplotlib (energy, inertial h, omega components, momentum sphere)
# 3D viz:    PyVista (desktop OpenGL window, tumbling box)
#
# Usage:  python attitude_viz.py
# Install: pip install pyvista matplotlib numpy
#
# For future coaxcopter viz: same pattern — build meshes, update transforms per frame.
# PyVista is desktop-native (not browser), supports interactive camera while animating.

import numpy as np
import matplotlib.pyplot as plt

from attitude_dynamics import (
    xhist, t, n_steps, h_step, tf,
    T_energy, h_inertial, h_body_normed,
    J, quat_to_rotmat,
)


# ══════════════════════════════════════════════════════════════════════
# 2D PLOTS (matplotlib)
# ══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

# omega components
axes[0].plot(t, xhist[4, :], label=r'$\omega_1$')
axes[0].plot(t, xhist[5, :], label=r'$\omega_2$')
axes[0].plot(t, xhist[6, :], label=r'$\omega_3$')
axes[0].set_ylabel("Angular velocity [rad/s]")
axes[0].set_title("Body-frame angular velocity")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# kinetic energy
axes[1].plot(t, T_energy, 'c')
axes[1].set_ylabel("Kinetic energy [J]")
axes[1].set_title(f"Rotational KE  (drift = {abs(T_energy[-1]-T_energy[0]):.2e})")
axes[1].grid(True, alpha=0.3)

# inertial angular momentum
axes[2].plot(t, h_inertial[0, :], label=r'$h_1^N$')
axes[2].plot(t, h_inertial[1, :], label=r'$h_2^N$')
axes[2].plot(t, h_inertial[2, :], label=r'$h_3^N$')
axes[2].set_ylabel("Ang. momentum")
axes[2].set_xlabel("Time [s]")
axes[2].set_title("Inertial angular momentum (should be constant)")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("attitude_2d_plots.png", dpi=150, bbox_inches='tight')
plt.show()


# ══════════════════════════════════════════════════════════════════════
# MOMENTUM SPHERE (matplotlib 3D)
# ══════════════════════════════════════════════════════════════════════

fig_sphere = plt.figure(figsize=(8, 8))
ax3 = fig_sphere.add_subplot(111, projection='3d')

# unit sphere
u_ang = np.linspace(0, 2*np.pi, 60)
v_ang = np.linspace(0, np.pi, 40)
xs = np.outer(np.cos(u_ang), np.sin(v_ang))
ys = np.outer(np.sin(u_ang), np.sin(v_ang))
zs = np.outer(np.ones_like(u_ang), np.cos(v_ang))
ax3.plot_surface(xs, ys, zs, alpha=0.15, color='cyan')

# equilibria: green = stable (major/minor axis), red = unstable (intermediate)
s = 1.05
ax3.scatter([ s, -s], [0, 0], [0, 0], c='lime', s=60, zorder=5, label='stable (axis 1, minor)')
ax3.scatter([0, 0], [ s, -s], [0, 0], c='red',  s=60, zorder=5, label='unstable (axis 2, intermediate)')
ax3.scatter([0, 0], [0, 0], [ s, -s], c='lime', s=60, zorder=5, label='stable (axis 3, major)')

# momentum trajectory
ax3.plot(h_body_normed[0, :], h_body_normed[1, :], h_body_normed[2, :],
         'r-', linewidth=2, label='h trajectory')

ax3.set_xlabel(r'$\hat{h}_1$')
ax3.set_ylabel(r'$\hat{h}_2$')
ax3.set_zlabel(r'$\hat{h}_3$')
ax3.set_title('Momentum Sphere (body frame)')
ax3.legend(fontsize=8)

plt.tight_layout()
plt.savefig("momentum_sphere.png", dpi=150, bbox_inches='tight')
plt.show()


# ══════════════════════════════════════════════════════════════════════
# 3D PYVISTA TUMBLE ANIMATION (desktop window)
# ══════════════════════════════════════════════════════════════════════

try:
    import pyvista as pv
    import time

    # subsample frames for smooth playback
    frame_step = max(1, n_steps // 500)
    frame_indices = list(range(0, n_steps, frame_step))

    # ── Build scene ───────────────────────────────────────────────────

    pl = pv.Plotter(window_size=[1200, 800])
    pl.set_background('black')

    # spacecraft box (dimensions loosely reflect J = [1,2,3])
    # smaller J axis = longer physical dimension (easier to spin)
    box_mesh = pv.Box(bounds=(-0.7, 0.7, -0.5, 0.5, -0.3, 0.3))
    box_actor = pl.add_mesh(box_mesh, color='steelblue', show_edges=True,
                            edge_color='white', opacity=0.9)

    # inertial frame axes (static)
    pl.add_mesh(pv.Arrow(start=(0,0,0), direction=(1,0,0), scale=1.5),
                color='red',   label='X inertial')
    pl.add_mesh(pv.Arrow(start=(0,0,0), direction=(0,1,0), scale=1.5),
                color='green', label='Y inertial')
    pl.add_mesh(pv.Arrow(start=(0,0,0), direction=(0,0,1), scale=1.5),
                color='blue',  label='Z inertial')

    pl.add_legend(loc='upper right')
    pl.add_text("Torque-Free Tumble  |  close window to exit",
                position='upper_left', font_size=11, color='white')

    # ── Animate ───────────────────────────────────────────────────────

    pl.show(auto_close=False, interactive_update=True)

    for loop in range(5):  # loop animation 5 times
        for idx in frame_indices:
            q_k = xhist[0:4, idx]
            Q   = quat_to_rotmat(q_k)  # 3x3 body->inertial

            # build 4x4 homogeneous transform
            T4 = np.eye(4)
            T4[:3, :3] = Q

            # update box orientation
            box_actor.user_matrix = T4

            pl.update()
            time.sleep(h_step * frame_step * 0.3)

            # check if window was closed
            if not pl.render_window:
                break
        else:
            continue
        break

    pl.close()

except ImportError:
    print("\n  PyVista not installed -- skipping 3D tumble animation.")
    print("  Install with: pip install pyvista")
    print("  2D plots and momentum sphere above still work.\n")

except Exception as e:
    print(f"\n  PyVista 3D error: {e}")
    print("  This can happen in headless/WSL environments.")
    print("  2D plots and momentum sphere above still work.\n")
