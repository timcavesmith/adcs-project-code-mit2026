# attitude_viz.py
# Visualization for HW 1 parts 2 and 3
#
# Part 2: omega vs time for each principal axis case (stability verification)
#          + energy/momentum conservation checks
# Part 3: momentum sphere with 6 equilibria + all trajectories
# Bonus:  PyVista 3D tumble animation (desktop window)
#
# Usage:  python attitude_viz.py
# Install: pip install pyvista matplotlib numpy

import numpy as np
import matplotlib.pyplot as plt

from attitude_dynamics import (
    results, cases, extra_cases,
    h_mag, J, J_diag, quat_to_rotmat,
    h_step, n_steps,
)


# ══════════════════════════════════════════════════════════════════════
# PART 2: OMEGA COMPONENTS VS TIME (one subplot per principal axis case)
# ══════════════════════════════════════════════════════════════════════

case_names = list(cases.keys())

fig, axes = plt.subplots(3, 3, figsize=(16, 10), sharex=True)

for col, name in enumerate(case_names):
    r = results[name]
    t = r['t']
    xhist = r['xhist']

    # omega components
    axes[0, col].plot(t, xhist[4, :], label=r'$\omega_1$')
    axes[0, col].plot(t, xhist[5, :], label=r'$\omega_2$')
    axes[0, col].plot(t, xhist[6, :], label=r'$\omega_3$')
    axes[0, col].set_title(name, fontsize=9)
    axes[0, col].legend(fontsize=7)
    axes[0, col].grid(True, alpha=0.3)
    if col == 0:
        axes[0, col].set_ylabel('Angular velocity [rad/s]')

    # kinetic energy
    axes[1, col].plot(t, r['T_energy'], 'c')
    e_drift = abs(r['T_energy'][-1] - r['T_energy'][0])
    axes[1, col].set_title(f'KE (drift={e_drift:.1e})', fontsize=9)
    axes[1, col].grid(True, alpha=0.3)
    if col == 0:
        axes[1, col].set_ylabel('Kinetic energy')

    # inertial angular momentum
    axes[2, col].plot(t, r['h_inertial'][0, :], label=r'$h_1^N$')
    axes[2, col].plot(t, r['h_inertial'][1, :], label=r'$h_2^N$')
    axes[2, col].plot(t, r['h_inertial'][2, :], label=r'$h_3^N$')
    axes[2, col].legend(fontsize=7)
    axes[2, col].grid(True, alpha=0.3)
    axes[2, col].set_xlabel('Time [s]')
    if col == 0:
        axes[2, col].set_ylabel('Inertial ang. momentum')

fig.suptitle('Part 2: Stability Verification — Spin About Each Principal Axis (same ||h||)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('part2_stability.png', dpi=150, bbox_inches='tight')
plt.show()


# ══════════════════════════════════════════════════════════════════════
# PART 3: MOMENTUM SPHERE
# ══════════════════════════════════════════════════════════════════════

fig_sphere = plt.figure(figsize=(10, 10))
ax = fig_sphere.add_subplot(111, projection='3d')

# unit sphere
u_ang = np.linspace(0, 2*np.pi, 80)
v_ang = np.linspace(0, np.pi, 50)
xs = np.outer(np.cos(u_ang), np.sin(v_ang))
ys = np.outer(np.sin(u_ang), np.sin(v_ang))
zs = np.outer(np.ones_like(u_ang), np.cos(v_ang))
ax.plot_surface(xs, ys, zs, alpha=0.1, color='cyan')

# 6 equilibrium points
# axis 1 (minor, J=1): stable  — green
# axis 2 (intermediate, J=2): unstable — red
# axis 3 (major, J=3): stable  — green
s = 1.05
ax.scatter([ s, -s], [0, 0], [0, 0], c='lime',  s=80, zorder=5,
           label='Stable eq. (axis 1, minor)', edgecolors='black', linewidths=0.5)
ax.scatter([0, 0], [ s, -s], [0, 0], c='red',   s=80, zorder=5,
           label='Unstable eq. (axis 2, intermediate)', edgecolors='black', linewidths=0.5)
ax.scatter([0, 0], [0, 0], [ s, -s], c='lime',  s=80, zorder=5,
           label='Stable eq. (axis 3, major)', edgecolors='black', linewidths=0.5)

# plot all trajectories
all_cases = {**cases, **extra_cases}
for name, case_info in all_cases.items():
    r = results[name]
    hbn = r['h_body_normed']
    color = r['color']
    lw = 2.5 if name in cases else 1.2
    alpha = 1.0 if name in cases else 0.6
    # shorten label for legend
    short = name.split('(')[0].strip() if '(' in name else name
    ax.plot(hbn[0, :], hbn[1, :], hbn[2, :],
            color=color, linewidth=lw, alpha=alpha, label=short)

ax.set_xlabel(r'$\hat{h}_1$ (minor axis)')
ax.set_ylabel(r'$\hat{h}_2$ (intermediate)')
ax.set_zlabel(r'$\hat{h}_3$ (major axis)')
ax.set_title('Part 3: Momentum Sphere — Equilibria and Trajectories', fontsize=13, fontweight='bold')
ax.legend(fontsize=7, loc='upper left')

# set equal aspect
ax.set_xlim([-1.3, 1.3])
ax.set_ylim([-1.3, 1.3])
ax.set_zlim([-1.3, 1.3])

plt.tight_layout()
plt.savefig('part3_momentum_sphere.png', dpi=150, bbox_inches='tight')
plt.show()


# ══════════════════════════════════════════════════════════════════════
# BONUS: Additional view — momentum sphere from different angles
# ══════════════════════════════════════════════════════════════════════

fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6),
                            subplot_kw={'projection': '3d'})

views = [(20, 30), (90, 0), (0, 0)]   # (elevation, azimuth)
view_titles = ['Perspective', 'Top (looking down axis 3)', 'Side (looking down axis 1)']

for ax_i, (elev, azim), title in zip(axes2, views, view_titles):
    # sphere
    ax_i.plot_surface(xs, ys, zs, alpha=0.08, color='cyan')

    # equilibria
    ax_i.scatter([ s, -s], [0, 0], [0, 0], c='lime', s=60, zorder=5, edgecolors='black', linewidths=0.5)
    ax_i.scatter([0, 0], [ s, -s], [0, 0], c='red',  s=60, zorder=5, edgecolors='black', linewidths=0.5)
    ax_i.scatter([0, 0], [0, 0], [ s, -s], c='lime', s=60, zorder=5, edgecolors='black', linewidths=0.5)

    # trajectories
    for name in all_cases:
        r = results[name]
        hbn = r['h_body_normed']
        lw = 2.0 if name in cases else 1.0
        ax_i.plot(hbn[0,:], hbn[1,:], hbn[2,:], color=r['color'], linewidth=lw)

    ax_i.view_init(elev=elev, azim=azim)
    ax_i.set_title(title, fontsize=10)
    ax_i.set_xlim([-1.3, 1.3])
    ax_i.set_ylim([-1.3, 1.3])
    ax_i.set_zlim([-1.3, 1.3])
    ax_i.set_xlabel(r'$\hat{h}_1$')
    ax_i.set_ylabel(r'$\hat{h}_2$')
    ax_i.set_zlabel(r'$\hat{h}_3$')

fig2.suptitle('Momentum Sphere — Multiple Views', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('part3_momentum_views.png', dpi=150, bbox_inches='tight')
plt.show()


# ══════════════════════════════════════════════════════════════════════
# 3D PYVISTA TUMBLE ANIMATION (intermediate axis )
# ══════════════════════════════════════════════════════════════════════

try:
    import pyvista as pv
    import time

    # intermediate axis case
    case_name = list(cases.keys())[1]  # intermediate axis
    xhist_anim = results[case_name]['xhist']
    n = xhist_anim.shape[1]

    #frame_step = max(1, n // 500) #only show about 500 frames
    #frame_indices = list(range(0, n, frame_step))
    frame_indices = list(range(0, n))  # every frame, no skipping
    #frame_indices = list(range(0, n, 2)) #skip every other frame

    pl = pv.Plotter(window_size=[1200, 800])
    pl.set_background('black')

    # box sized to reflect inertia
    box_mesh = pv.Box(bounds=(-0.7, 0.7, -0.5, 0.5, -0.3, 0.3))
    box_actor = pl.add_mesh(box_mesh, color='steelblue', show_edges=True,
                            edge_color='white', opacity=0.9)

    # inertial axes
    pl.add_mesh(pv.Arrow(start=(0,0,0), direction=(1,0,0), scale=1.5), color='red')
    pl.add_mesh(pv.Arrow(start=(0,0,0), direction=(0,1,0), scale=1.5), color='green')
    pl.add_mesh(pv.Arrow(start=(0,0,0), direction=(0,0,1), scale=1.5), color='blue')

    pl.add_text(f"Tumble: {case_name}  |  close window to exit",
                position='upper_left', font_size=10, color='white')
    try:
        pl.show(auto_close=False, interactive_update=True)

        for loop in range(5):
            for idx in frame_indices:
                q_k = xhist_anim[0:4, idx]
                Q = quat_to_rotmat(q_k)
                T4 = np.eye(4)
                T4[:3, :3] = Q
                box_actor.user_matrix = T4
                pl.update()
                #time.sleep(h_step * frame_step * 0.3)
                #time.sleep(0.02)  # 0.02 is about 50 fps, adjust up for slower playback
                #time.sleep(h_step) # (sim time = wall clock time) about 20 fps with 0.05 sec h_step
                time.sleep(h_step / 2) # double wall clock time
            
            
    except (KeyboardInterrupt, Exception):
        pass
    finally:
        pl.close()

except ImportError:
    print("\n  PyVista not installed -- skipping 3D tumble animation.")
    print("  Install with: pip install pyvista")
    print("  All plots above still work.\n")

except Exception as e:
    print(f"\n  PyVista 3D error: {e}")
    print("  All plots above still work.\n")
