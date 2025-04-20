import matplotlib.pyplot as plt

def load_g2o_2d(file_path):
    poses = {}  # key: node ID, value: (x, y, theta)
    edges = []  # list of (from_id, to_id)

    with open(file_path, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue
            if tokens[0] == "VERTEX_SE2":
                idx = int(tokens[1])
                x, y, theta = map(float, tokens[2:5])
                poses[idx] = (x, y, theta)
            elif tokens[0] == "EDGE_SE2":
                from_id = int(tokens[1])
                to_id = int(tokens[2])
                edges.append((from_id, to_id))
    
    return poses, edges

def plot_pose_graph(poses, edges, title="2D Pose Graph"):
    plt.figure(figsize=(10, 8))
    xs, ys = zip(*[(p[0], p[1]) for p in poses.values()])
    plt.scatter(xs, ys, s=10, label="Poses", zorder=2)

    for from_id, to_id in edges:
        if from_id in poses and to_id in poses:
            x1, y1 = poses[from_id][0:2]
            x2, y2 = poses[to_id][0:2]
            plt.plot([x1, x2], [y1, y2], 'gray', linewidth=0.5, zorder=1)

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    g2o_file = "input_M3500_g2o.g2o"  # path to your g2o file
    poses, edges = load_g2o_2d(g2o_file)
    plot_pose_graph(poses, edges)
