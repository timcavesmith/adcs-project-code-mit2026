# Reaction wheel cluster from HW4 part 1: four RSI-68 wheels in a pyramid,
# cant 30 deg above the body x-y plane.
#
# Lecture 16 sign convention: rho_dot = B_W u_W, and the body sees -rho_dot
# from the wheels (Newton's third law). So a desired body torque tau_cmd is
# produced by commanding rho_dot_des = -tau_cmd, i.e. u_W = -pinv(B_W) tau_cmd.

import numpy as np

CANT_DEG = 30.0
_c = np.cos(np.radians(CANT_DEG))    # 0.866, z-component of each spin axis
_xy = np.sin(np.radians(CANT_DEG)) / np.sqrt(2)   # 0.354, x and y components

# Spin-axis unit vectors of the four wheels in body frame; matches the table
# in the HW4 writeup.
B_W = np.column_stack([
    [-_xy, -_xy, _c],
    [-_xy,  _xy, _c],
    [ _xy, -_xy, _c],
    [ _xy,  _xy, _c],
])
B_W_pinv = np.linalg.pinv(B_W)   # 4x3 min-norm right-inverse

# Datasheet limits (Rockwell Collins / Collins Aerospace RSI-68-170/60).
WHEEL_TORQUE_MAX = 0.170      # Nm per wheel
WHEEL_MOMENTUM_MAX = 60.0     # Nms per wheel
N_WHEELS = 4


def body_torque_capability(direction):
    """Largest body torque magnitude along a unit direction while every wheel
    stays inside +/- WHEEL_TORQUE_MAX."""
    d = np.asarray(direction, dtype=float)
    return WHEEL_TORQUE_MAX / np.max(np.abs(B_W_pinv @ (d / np.linalg.norm(d))))


def wheel_command(tau_body):
    """Body torque command (3,) -> per-wheel torques (4,). See sign note above."""
    return -B_W_pinv @ tau_body


def saturate_wheel_torque(u_W):
    """Clip per-wheel torques to the datasheet limit."""
    return np.clip(u_W, -WHEEL_TORQUE_MAX, WHEEL_TORQUE_MAX)


if __name__ == "__main__":
    print("B_W =\n", B_W)
    print("\nBody torque capability along body axes:")
    for name, d in zip("xyz", np.eye(3)):
        print(f"  {name}: {body_torque_capability(d):.3f} Nm")
