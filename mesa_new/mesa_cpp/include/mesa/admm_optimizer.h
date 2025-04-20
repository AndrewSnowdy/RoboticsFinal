#ifndef ADMM_OPTIMIZER_H
#define ADMM_OPTIMIZER_H

#include <map>
#include <set>
#include <string>
#include <vector>
#include <utility> 
#include <cmath>   

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/base/utilities.h> 

#include "dataset.h" 

using RobotId = size_t;

struct AdmmConfig {
    size_t max_iterations = 14;
    double beta = 0.00015; // .00012 and 16 iter
    bool verbose = false; 
};

class AdmmOptimizer {
public:
    AdmmOptimizer(const std::map<RobotId, RobotSubgraph>& subgraphs, const AdmmConfig& config);

    void run();
    void saveResultsToFile(const std::string& path) const;

private:
    struct SeparatorData {
        gtsam::Pose2 z;        //consensus variable
        gtsam::Vector3 lambda; //dual variable
    };
    using SeparatorKey = gtsam::Key;
    using RobotPair = std::pair<RobotId, RobotId>;
    std::map<RobotPair, std::map<SeparatorKey, SeparatorData>> separator_info_;

    std::map<RobotId, RobotSubgraph> robots_;
    std::map<RobotId, gtsam::Values> estimates_; // holds theta_i
    std::map<RobotId, gtsam::NonlinearFactorGraph> base_graphs_;
    AdmmConfig config_;
    double beta_;

    void initializeAdmmData();
    void exchangeAndConsensusStep();
    void runLocalOptimization(RobotId rid);

    gtsam::Pose2 computeGeodesicConsensus(const gtsam::Pose2& a, const gtsam::Pose2& b) const {
         gtsam::Pose2 delta = a.between(b);
         return a.retract(0.5 * gtsam::Pose2::Logmap(delta));
     }
};

#endif // ADMM_OPTIMIZER_H