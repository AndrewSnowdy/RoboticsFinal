#include "mesa/dataset.h"
#include <fstream>
#include <sstream>
#include <iostream>

Dataset::Dataset(const std::string& pose_file, const std::string& edge_file) {
    loadPoses(pose_file);
    loadEdges(edge_file);
}

// takes out everthing from pose file adding it to initial_
// which is gtsam Values
void Dataset::loadPoses(const std::string& pose_file) {
    std::ifstream file(pose_file);
    std::string line;
    
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::stringstream ss(line);
        size_t rid, pid;
        double x, y, theta;
        ss >> rid >> pid >> x >> y >> theta;

        size_t key = rid * 100000 + pid; // unique key per robot-pose
        initial_.insert(key, Pose2(x, y, theta));
    }
    file.close();
}

// creates gtsam betweenfactor given edge files
void Dataset::loadEdges(const std::string& edge_file) {
    std::ifstream file(edge_file);
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::stringstream ss(line);
        size_t fr, fp, tr, tp;
        std::string type;
        double dx, dy, dtheta, xy_std, theta_std;

        ss >> fr >> fp >> tr >> tp >> type >> dx >> dy >> dtheta >> xy_std >> theta_std;

        noiseModel::Diagonal::shared_ptr noise = noiseModel::Diagonal::Sigmas(Vector3(xy_std, xy_std, theta_std * M_PI / 180.0));
        size_t from_key = fr * 100000 + fp;
        size_t to_key = tr * 100000 + tp;

        graph_.add(BetweenFactor<Pose2>(from_key, to_key, Pose2(dx, dy, dtheta), noise));
    }

    file.close();

    // adds first pose of each robot with little uncertainty 
    noiseModel::Diagonal::shared_ptr prior_noise = noiseModel::Diagonal::Sigmas(Vector3(0.01, 0.01, 0.01));
    for (size_t rid = 0; rid < 5; ++rid) { // num_robots=5 here
        size_t first_key = rid * 100000;
        if (initial_.exists(first_key))
            graph_.add(PriorFactor<Pose2>(first_key, initial_.at<Pose2>(first_key), prior_noise));
    }
}

NonlinearFactorGraph Dataset::getFactorGraph() const {
    return graph_;
}

Values Dataset::getInitialValues() const {
    return initial_;
}

// separates into per robot graph
std::map<size_t, RobotSubgraph> Dataset::splitByRobot() const {
    std::map<size_t, RobotSubgraph> subgraphs;

    for (const auto& [key, value] : initial_) {
        size_t rid = robotIdFromKey(key);
        subgraphs[rid].initial.insert(key, value);
        subgraphs[rid].local_keys.insert(key);
    }

    for (const auto& factor : graph_) {
        auto keys = factor->keys();
        std::set<size_t> robots_involved;

        for (const auto& k : keys) {
            robots_involved.insert(robotIdFromKey(k));
        }

        //this will run, in case multiple communication broadcast
        if (robots_involved.size() == 1) { //if intra
            size_t rid = *robots_involved.begin();
            subgraphs[rid].graph.add(factor);
        }else{
            
            for (size_t rid : robots_involved) {//if inter
                //need to add to each subgraph a copy
                subgraphs[rid].graph.add(factor);
                for (auto k : keys) {
                    if (robotIdFromKey(k) != rid) //add separator if not owned by rid
                    subgraphs[rid].separator_keys.insert(k);
                }
            }
        }
    }
    
    return subgraphs;

}
