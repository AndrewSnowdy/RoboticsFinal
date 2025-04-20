import matplotlib.pyplot as plt

def load_g2o_2d(file_path):
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("VERTEX_SE2"):
                _, idx, x, y, theta = line.strip().split()
                poses.append((float(x), float(y)))
    return poses

def plot_g2o_trajectory(poses):
    xs, ys = zip(*poses)
    plt.figure(figsize=(10, 8))
    plt.plot(xs, ys, '-o', markersize=2, label="Trajectory")
    plt.title("2D Trajectory from .g2o file")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    poses = load_g2o_2d("input_M3500_g2o.g2o")  # Adjust path if needed
    plot_g2o_trajectory(poses)
