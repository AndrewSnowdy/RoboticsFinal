#ifndef DECENTRALIZED_H
#define DECENTRALIZED_H

#include <map>
#include <set>
#include <string>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/geometry/Pose2.h>
#include "dataset.h"

using RobotId = size_t;

struct DecentralizedConfig {
    size_t max_iterations = 5;
    double prior_strength = 1.0;
    bool verbose = true;
};

class DecentralizedOptimizer {
public:
    DecentralizedOptimizer(const std::map<RobotId, RobotSubgraph>& subgraphs, const DecentralizedConfig& config);

    void run();
    void saveResultsToFile(const std::string& path) const;

private:
    void runLocal(RobotId rid);
    void exchangeConsensus();

    std::map<RobotId, RobotSubgraph> robots_;
    std::map<RobotId, gtsam::Values> estimates_;
    std::map<gtsam::Key, gtsam::Pose2> global_estimates_;
    DecentralizedConfig config_;
    std::map<RobotId, gtsam::NonlinearFactorGraph> base_graphs_;
};

#endif // DECENTRALIZED_H
