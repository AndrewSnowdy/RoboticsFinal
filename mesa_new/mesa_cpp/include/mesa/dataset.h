#ifndef MESA_DATASET_H
#define MESA_DATASET_H

#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/geometry/Pose2.h>
#include <string>
#include <vector>

#include <set>
#include <map>

using namespace gtsam;

struct RobotSubgraph {
    NonlinearFactorGraph graph;
    Values initial;
    std::set<size_t> local_keys;
    std::set<size_t> separator_keys;
};

inline size_t robotIdFromKey(size_t key) {
    return key/100000; //used to distinguish in gtsam
};

inline size_t poseIdFromKey(size_t key) {
    return key % 100000;
};

struct Edge {
    size_t from_robot, from_pose, to_robot, to_pose;
    std::string type;
    Pose2 measurement;
    noiseModel::Diagonal::shared_ptr noise;
};

class Dataset {
public:
    Dataset(const std::string& pose_file, const std::string& edge_file);

    NonlinearFactorGraph getFactorGraph() const;
    Values getInitialValues() const;
    std::map<size_t, RobotSubgraph> splitByRobot() const;

private:
    void loadPoses(const std::string& pose_file);
    void loadEdges(const std::string& edge_file);

    NonlinearFactorGraph graph_;
    Values initial_;
};

#endif //MESA_DATASET_H
