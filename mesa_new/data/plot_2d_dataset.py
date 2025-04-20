"""
Andrew Snowdy


"""


import matplotlib.pyplot as plt
import numpy as np

def load_poses(file_path):
    poses = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            rid, pid, x, y, theta = line.strip().split()
            poses[(int(rid), int(pid))] = (float(x), float(y))
    return poses

def load_edges(file_path):
    edges = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            fr_rid, fr_pid, to_rid, to_pid, edge_type, *rest = line.strip().split()
            edges.append(((int(fr_rid), int(fr_pid)), (int(to_rid), int(to_pid)), edge_type))
    return edges

def plot_real_dataset(real_poses):
    current_rid = None
    xs, ys = [], []

    for (rid, _), (x, y) in sorted(real_poses.items()):
        if current_rid is None:
            current_rid = rid

        if rid != current_rid:
            plt.plot(xs, ys, color='lightgray', linewidth=1.5, linestyle='-', alpha=0.4)
            xs, ys = [], []
            current_rid = rid

        xs.append(x)
        ys.append(y)

    #plot the last robot's trajectory
    if xs and ys:
        plt.plot(xs, ys, color='lightgray', linewidth=1.5, linestyle='-', alpha=0.4)


def plot_dataset(poses, edges, comm=True, loop=True):
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']

    for (a, b, edge_type) in edges:
        if not comm and edge_type == 'comm':
            continue 

        if not loop and edge_type == 'loop':
            continue

        if a in poses and b in poses:
            x1, y1 = poses[a]
            x2, y2 = poses[b]

            if edge_type == 'odom':
                from_rid = a[0]
                color = colors[from_rid % len(colors)]
                linestyle = '-'
                linewidth = 2
                alpha = 0.9
            elif edge_type == 'loop':
                color = 'gray'
                linestyle = '--'
                linewidth = 1
                alpha = 0.4
            elif edge_type == 'comm':
                color = 'red'
                linestyle = ':'
                linewidth = 1
                alpha = 0.6
            else:
                color = 'black'
                linestyle = ':'
                linewidth = 1
                alpha = 0.3

            plt.plot(
                [x1, x2], [y1, y2],
                linestyle,
                color=color,
                linewidth=linewidth,
                alpha=alpha
            )


def load_optimized(file_path):
    optimized = {}
    with open(file_path, 'r') as f:
        for line in f:
            rid, pid, x, y, theta = line.strip().split()
            optimized[(int(rid), int(pid))] = (float(x), float(y))
    return optimized


def calculate_ate_rmse(ground_truth_poses, estimated_poses):
    all_errors_sq = []
    robot_ate_rmse = {}
    unique_rids = sorted(list(set(rid for rid, pid in ground_truth_poses.keys())))

    if not unique_rids:
        print("Error: No robot IDs found in ground truth poses.")
        return None, {}

    for rid in unique_rids:
        # corresponding timsteps
        gt_points = []
        est_points = []
        common_pids = sorted(list(set(pid for r, pid in ground_truth_poses.keys() if r == rid) &
                               set(pid for r, pid in estimated_poses.keys() if r == rid)))

        for pid in common_pids:
            gt_points.append(ground_truth_poses[(rid, pid)])
            est_points.append(estimated_poses[(rid, pid)])

        gt_array = np.array(gt_points).T #(2, N)
        est_array = np.array(est_points).T #(2, N)

        # center the points - basically horns method
        gt_centroid = np.mean(gt_array, axis=1, keepdims=True)
        est_centroid = np.mean(est_array, axis=1, keepdims=True)
        gt_centered = gt_array - gt_centroid
        est_centered = est_array - est_centroid

        H = est_centered @ gt_centered.T #covariance matrix (2, 2)

        # SVD
        U, S, Vt = np.linalg.svd(H)
        V = Vt.T
        R = V @ U.T #(2, 2)

        # reflection case
        if np.linalg.det(R) < 0:
            Vt[1, :] *= -1 
            R = Vt.T @ U.T

        t = gt_centroid - R @ est_centroid # optimal translation (2, 1)
        est_aligned = R @ est_array + t # estimated pose alignments (2, N)

        # errors
        errors = np.linalg.norm(gt_array - est_aligned, axis=0) 
        errors_sq = errors**2
        all_errors_sq.extend(errors_sq)

        # Calculate RMSE for this robot
        rmse_robot = np.sqrt(np.mean(errors_sq))
        robot_ate_rmse[rid] = rmse_robot
        print(f"Robot {rid} - ATE RMSE: {rmse_robot:.4f}")



    overall_rmse = np.sqrt(np.mean(all_errors_sq))
    print(f"\nOverall ATE RMSE: {overall_rmse:.4f}")

    return overall_rmse, robot_ate_rmse




if __name__ == "__main__":
    real_poses = load_poses("data/real_poses.txt")
    noisy_poses = load_poses("data/poses.txt")
    optimized_poses = load_optimized("data/local_optimized.txt")
    decentralized_poses = load_optimized("data/decentralized_optimization.txt")
    admm_poses = load_optimized("data/admm_optimized.txt")
    edges = load_edges("data/edges.txt")

    # calculate ATE for decentralized and local
    print("\nCalculating ATE for noisy:")
    ate_rmse_nan, robot_rmse_nan = calculate_ate_rmse(real_poses, noisy_poses)

    print("\nCalculating ATE for locally optimized")
    ate_rmse_local, robot_rmse_local = calculate_ate_rmse(real_poses, optimized_poses)

    print("\nCalculating ATE for decentralized optimizer:")
    ate_rmse_decentralized, robot_rmse_decentralized = calculate_ate_rmse(real_poses, decentralized_poses)

    print("\nCalculating ATE for ADMM optimizer:")
    ate_rmse_admm, robot_rmse_admm = calculate_ate_rmse(real_poses, admm_poses)


    fig, axs = plt.subplots(2, 2, figsize=(16, 14))  # Adjust size if needed


    # Top-left: Original noisy
    plt.sca(axs[0, 0])
    plot_real_dataset(real_poses)
    plot_dataset(noisy_poses, edges, comm=False, loop=False)
    axs[0, 0].set_title(f"Noisy Graph, ATE: {ate_rmse_nan:.4f}")
    axs[0, 0].set_xlabel(" ")
    axs[0, 0].set_ylabel("Y")
    axs[0, 0].axis("equal")

    # Top-right: Local optimizer
    plt.sca(axs[0, 1])
    plot_real_dataset(real_poses)
    plot_dataset(optimized_poses, edges, comm=False, loop=False)
    axs[0, 1].set_title(f"Local Optimized, ATE: {ate_rmse_local:.4f}")
    axs[0, 1].set_xlabel(" ")
    axs[0, 1].axis("equal")

    # Bottom-left: Decentralized
    plt.sca(axs[1, 0])
    plot_real_dataset(real_poses)
    plot_dataset(decentralized_poses, edges, comm=False, loop=False)
    axs[1, 0].set_title(f"Decentralized Optimized, ATE: {ate_rmse_decentralized:.4f}")
    axs[1, 0].set_xlabel(" ")
    axs[1, 0].axis("equal")

    # Bottom-right: ADMM Optimizer (MESA)
    plt.sca(axs[1, 1])
    plot_real_dataset(real_poses)
    plot_dataset(admm_poses, edges, comm=True, loop=False)
    axs[1, 1].set_title(f"ADMM Optimized, ATE: {ate_rmse_admm:.4f}")
    axs[1, 1].set_xlabel(" ")
    axs[1, 1].axis("equal")


    # --- TODO: MESA optimization

    plt.tight_layout()
    plt.show()
