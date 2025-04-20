import numpy as np
import os

# Parameters
num_robots = 4
poses_per_robot = 800
noise_std = [0.15, 0.15, np.deg2rad(3)]  # odometry noise [dx, dy, dtheta]

min_x, max_x = -50.0, 50.0
min_y, max_y = -50.0, 50.0

ODOM_PROBS = [0.8, 0.15, 0.05]
ODOM_GRID = np.array([
    [1, 0, 0],              # forward
    [0, 0, np.pi / 2],      # left
    [0, 0, -np.pi / 2],     # right
])

# loop closure parameters
loop_index_threshold = 10
loop_distance_threshold = 6.0
loop_probability = 0.5
loop_noise_std = [0.3, 0.3, np.deg2rad(4)]  # loop closure noise [dx, dy, dtheta]

# inter-robot communication
comm_range = 35
comm_freq = 10
comm_noise_std = [0.3, 0.3, np.deg2rad(4)]

# file setup
pose_file = open("data/poses.txt", "w")
edge_file = open("data/edges.txt", "w")
real_pose_file = open("data/real_poses.txt", "w")

pose_file.write("# robot_id pose_id x y theta\n")
edge_file.write("# from_robot from_pose to_robot to_pose type dx dy dtheta xy_noise_std theta_noise_std\n")
real_pose_file.write("# robot_id pose_id x y theta\n")

def wrap_angle(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi

def compute_relative_pose(from_pose, to_pose):
    fx, fy, ftheta = from_pose
    tx, ty, ttheta = to_pose
    dx = tx - fx
    dy = ty - fy
    dx_body = np.cos(-ftheta) * dx - np.sin(-ftheta) * dy
    dy_body = np.sin(-ftheta) * dx + np.cos(-ftheta) * dy
    dtheta = wrap_angle(ttheta - ftheta)
    return dx_body, dy_body, dtheta


# intra-loop closure should ONLY happen with ground-truth data is nearby
# and then create connection with simulated/noisy data
#
# THIS IS CURRENTLY NOT BEING DONE
#   loops are create around ground-truth
def add_intra_loop_closure(rid, pid, current_pose, real_poses, edge_file):
    if pid <= loop_index_threshold:
        return
    rx, ry, rtheta = current_pose
    for past_pid in range(pid - loop_index_threshold):
        past_pose = real_poses[rid][past_pid]
        dist = np.linalg.norm([rx - past_pose[0], ry - past_pose[1]])
        if dist < loop_distance_threshold and np.random.rand() < loop_probability:
            dx, dy, dtheta = compute_relative_pose(past_pose, current_pose)
            # explicitly adding noise
            dx_noisy, dy_noisy, dtheta_noisy = dx, dy, dtheta
            dx_noisy += np.random.normal(0, loop_noise_std[0])
            dy_noisy += np.random.normal(0, loop_noise_std[1])
            dtheta_noisy += np.random.normal(0, loop_noise_std[2])
            edge_file.write(
                f"{rid} {past_pid} {rid} {pid} loop {dx_noisy:.4f} {dy_noisy:.4f} {dtheta_noisy:.4f} "
                f"{loop_noise_std[0]} {np.rad2deg(loop_noise_std[2]):.4f}\n"
            )
            break


def add_comm_edges(pid, real_poses, edge_file):
    for ra in range(num_robots):
        for rb in range(ra + 1, num_robots):
            xa, ya, ta = real_poses[ra][pid]
            xb, yb, tb = real_poses[rb][pid]
            dist = np.linalg.norm([xa - xb, ya - yb])
            if dist < comm_range:
                dx, dy, dtheta = compute_relative_pose((xa, ya, ta), (xb, yb, tb))
                # Explicitly adding noise
                dx_noisy = dx + np.random.normal(0, comm_noise_std[0])
                dy_noisy = dy + np.random.normal(0, comm_noise_std[1])
                dtheta_noisy = dtheta + np.random.normal(0, comm_noise_std[2])

                edge_file.write(f"{ra} {pid} {rb} {pid} comm {dx_noisy:.4f} {dy_noisy:.4f} {dtheta_noisy:.4f} "
                                f"{comm_noise_std[0]} {np.rad2deg(comm_noise_std[2]):.4f}\n")
                edge_file.write(f"{rb} {pid} {ra} {pid} comm {-dx_noisy:.4f} {-dy_noisy:.4f} {-dtheta_noisy:.4f} "
                                f"{comm_noise_std[0]} {np.rad2deg(comm_noise_std[2]):.4f}\n")

def add_odom_noise(step):
    return step + np.random.normal(loc=0.0, scale=noise_std)

def update_pose_with_ground_truth_bounds(x, y, theta, dx, dy, dtheta,
                                         rx, ry, rtheta, dx_real, dy_real, dtheta_real):
    rtheta_new = (rtheta + dtheta_real) % (2 * np.pi)
    theta_new = (theta + dtheta) % (2 * np.pi)
    rx_next = rx + dx_real * np.cos(rtheta) - dy_real * np.sin(rtheta)
    ry_next = ry + dx_real * np.sin(rtheta) + dy_real * np.cos(rtheta)

    #need to turn around both real and noisy ONLY if real goes out of bounds in next step
    if not (min_x <= rx_next <= max_x and min_y <= ry_next <= max_y):
        rtheta_new = (rtheta + np.pi) % (2 * np.pi)
        theta_new = (theta + np.pi) % (2 * np.pi)
        return (x, y, theta_new), (rx, ry, rtheta_new)
    else:
        x_next = x + dx * np.cos(theta) - dy * np.sin(theta)
        y_next = y + dx * np.sin(theta) + dy * np.cos(theta)
        return (x_next, y_next, theta_new), (rx_next, ry_next, rtheta_new)

# Init states
init_positions = [(10,10,0), (-10,-10,np.pi/2), (10,-10,np.pi), (-10,10,-np.pi/2), (-5, -5, -np.pi/2)]
robot_states = {rid: init_positions[rid] for rid in range(num_robots)}
real_robot_states = {rid: init_positions[rid] for rid in range(num_robots)}

real_poses = {rid: [] for rid in range(num_robots)}

for pid in range(poses_per_robot):
    for rid in range(num_robots):
        x, y, theta = robot_states[rid]
        rx, ry, rtheta = real_robot_states[rid]

        move_idx = np.random.choice([0, 1, 2], p=ODOM_PROBS)
        dx, dy, dtheta = ODOM_GRID[move_idx]
        noisy_dx, noisy_dy, noisy_dtheta = add_odom_noise(ODOM_GRID[move_idx])

        (x, y, theta), (rx, ry, rtheta) = update_pose_with_ground_truth_bounds(
            x, y, theta, noisy_dx, noisy_dy, noisy_dtheta,
            rx, ry, rtheta, dx, dy, dtheta
        )

        pose_file.write(f"{rid} {pid} {x:.4f} {y:.4f} {theta:.4f}\n")
        real_pose_file.write(f"{rid} {pid} {rx:.4f} {ry:.4f} {rtheta:.4f}\n")

        real_poses[rid].append((rx, ry, rtheta))
        robot_states[rid] = (x, y, theta)
        real_robot_states[rid] = (rx, ry, rtheta)

        if pid > 0:
            edge_file.write(
                f"{rid} {pid-1} {rid} {pid} odom {noisy_dx:.4f} {noisy_dy:.4f} {noisy_dtheta:.4f} "
                f"{noise_std[0]} {np.rad2deg(noise_std[2]):.4f}\n"
            )

        add_intra_loop_closure(rid, pid, (rx, ry, rtheta), real_poses, edge_file)

    if pid % comm_freq == 0 and pid > 0:
        add_comm_edges(pid, real_poses, edge_file)

pose_file.close()
edge_file.close()
real_pose_file.close()
print(f"Generated dataset with {num_robots} robots and {poses_per_robot} poses each")
