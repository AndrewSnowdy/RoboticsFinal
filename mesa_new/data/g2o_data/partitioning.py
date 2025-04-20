import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import random

# Parameters
input_g2o_path = "data/g2o_data/input_M3500_g2o.g2o"
num_robots = 4

# Output text files
pose_output = "poses.txt"
edge_output = "edges.txt"

# Load poses and edges from g2o file
poses = {}
edges = []

with open(input_g2o_path, 'r') as f:
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

# Create graph
G = nx.Graph()
G.add_nodes_from(poses.keys())
G.add_edges_from(edges)

# METIS partitioning (requires networkx-metis)
try:
    import metis
    _, parts = metis.part_graph(G, num_robots)
except ImportError:
    print("networkx-metis is not installed. Install with pip install networkx-metis")
    exit()

# Map node to robot ID
robot_assignments = {node: part for node, part in zip(G.nodes(), parts)}

# Save as text files in the same format as generated data
with open(pose_output, 'w') as pose_file, open(edge_output, 'w') as edge_file:
    pose_file.write("# robot_id pose_id x y theta\n")
    edge_file.write("# from_robot from_pose to_robot to_pose type dx dy dtheta std_dev\n")

    pose_ids = defaultdict(int)  # Keep track of local pose ID for each robot
    global_to_local = {}  # Map global ID to (robot_id, local_id)

    for pid in sorted(poses.keys()):
        rid = robot_assignments[pid]
        local_id = pose_ids[rid]
        x, y, theta = poses[pid]
        pose_file.write(f"{rid} {local_id} {x} {y} {theta}\n")
        global_to_local[pid] = (rid, local_id)
        pose_ids[rid] += 1

    for from_id, to_id in edges:
        if from_id in global_to_local and to_id in global_to_local:
            from_rid, from_local = global_to_local[from_id]
            to_rid, to_local = global_to_local[to_id]
            fx, fy, ftheta = poses[from_id]
            tx, ty, ttheta = poses[to_id]
            dx, dy = tx - fx, ty - fy
            dtheta = ttheta - ftheta
            edge_file.write(f"{from_rid} {from_local} {to_rid} {to_local} odom {dx} {dy} {dtheta} 0.05\n")

print(f"Partitioned and saved to {pose_output} and {edge_output}.")
