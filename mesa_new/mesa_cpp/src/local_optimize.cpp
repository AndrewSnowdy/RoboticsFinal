#include <string>
#include <fstream>
#include "mesa/dataset.h"
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

void runLocalOptimize(const Dataset& dataset, const std::string& output_path) {
    auto subgraphs = dataset.splitByRobot();

    std::ofstream out(output_path);  // Single output file
    if (!out.is_open()) {
        std::cerr << "Error opening output file: " << output_path << std::endl;
        return;
    }

    for (const auto& [rid, robot] : subgraphs) {
        std::cout << "Locally Optimizing Robot " << rid << "..." << std::endl;

        // Filter local-only factors
        gtsam::NonlinearFactorGraph local_graph;
        for (const auto& factor : robot.graph) {
            bool all_local = true;
            for (const auto& key : factor->keys()) {
                if (robotIdFromKey(key) != rid) {
                    all_local = false;
                    break;
                }
            }
            if (all_local) {
                local_graph.add(factor);
            }
        }

        // Optimize
        gtsam::LevenbergMarquardtOptimizer optimizer(local_graph, robot.initial);
        gtsam::Values result = optimizer.optimize();

        // Append to single output file
        for (const auto& key : robot.local_keys) {
            if (result.exists(key)) {
                gtsam::Pose2 pose = result.at<gtsam::Pose2>(key);
                size_t pid = poseIdFromKey(key);
                out << rid << " " << pid << " "
                    << pose.x() << " " << pose.y() << " " << pose.theta() << "\n";
            }
        }
    }

    out.close();
}
