#include "mesa/decentralized.h"
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/linear/NoiseModel.h>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace gtsam;

DecentralizedOptimizer::DecentralizedOptimizer(const std::map<RobotId, RobotSubgraph>& subgraphs, const DecentralizedConfig& config)
    : robots_(subgraphs), config_(config) {
    for (const auto& [rid, robot] : robots_) {
        estimates_[rid] = robot.initial;
        base_graphs_[rid] = robot.graph; //store the original graph structure
    }
}

void DecentralizedOptimizer::run() {
    for (size_t iter = 0; iter < config_.max_iterations; ++iter) {
        std::cout << "Decentralized [Iteration " << iter << "]\n";

        exchangeConsensus();
        if (config_.verbose)
            std::cout << "completed consensus" << std::endl;

        for (const auto& [rid, _] : robots_) {
            runLocal(rid);
        }
    }
}

void DecentralizedOptimizer::exchangeConsensus() {
    // reset graphs to base versions (remove priors from previous iteration)
    for (auto& [rid, robot] : robots_) {

        robot.graph = base_graphs_.at(rid);
    }

    // find all unique separator keys and get the owner's current estimate for each
    global_estimates_.clear(); // stores separator keys
    std::set<Key> unique_separator_keys;
    for (const auto& [rid, robot] : robots_) {
        unique_separator_keys.insert(robot.separator_keys.begin(), robot.separator_keys.end());
    }

    for (Key k : unique_separator_keys) {
        RobotId owner_rid = robotIdFromKey(k);
        
        if (config_.verbose) {
            bool key_exists_check = estimates_.at(owner_rid).exists(k); // Check existence FIRST
            std::cout << "  Processing Key " << DefaultKeyFormatter(k) << " (Owner: " << owner_rid
                    << "). Checking estimates_[" << owner_rid << "].exists(" << DefaultKeyFormatter(k) << "): "
                    << std::boolalpha << key_exists_check << std::endl; // Print result
        }

        // for testing multi-comm system, but was too difficult/not necessary
        //   this just checks the owner, kept in but useless in the current case
        if (estimates_.count(owner_rid)) {
             if (estimates_.at(owner_rid).exists(k)) {
                global_estimates_[k] = estimates_.at(owner_rid).at<Pose2>(k);
             } else {
                 if (config_.verbose)
                    std::cerr << "error: " << owner_rid << " missing estimate for its own key " << DefaultKeyFormatter(k) << std::endl;
             }
        } else {
             if (config_.verbose)
                std::cerr << "error: " << owner_rid << " not in estimates map for key " << DefaultKeyFormatter(k) << std::endl;
        }
    }

    // add prior factors to robots that have 'k' as a separator
    //  creating the comm connecetion, and will be updated as the factorgraph adjusts
    noiseModel::Diagonal::shared_ptr noise = noiseModel::Diagonal::Sigmas(Vector3::Constant(1.0 / config_.prior_strength)); 

    for (auto& [rid, robot] : robots_) { // Need non-const robot to modify graph
        for (Key k : robot.separator_keys) {
            // if we have the owner's estimate
            if (global_estimates_.count(k)) { //check if we have the owners estimate, create new factor                
                 robot.graph.add(PriorFactor<Pose2>(k, global_estimates_.at(k), noise));

            }
        }
    }
}

void DecentralizedOptimizer::runLocal(RobotId rid) {
    if (config_.verbose)
        std::cout << "  Running local optimization for Robot " << rid << "..." << std::endl;

    // Get the robot subgraph (graph now includes priors on separator keys)
    const auto& robot = robots_.at(rid);

    if (config_.verbose)
        std::cout << "    Graph size: " << robot.graph.size() << std::endl;

    //create the initial guess for the optimizer:
    // start with the robot's current estimates for its OWN keys
    gtsam::Values initial_guess_for_optimizer = estimates_.at(rid);
    int separator_estimates_added = 0;

    // Add estimates for separator keys from global_estimates_
    // These are needed because the graph contains priors on these keys.

    for (Key k : robot.separator_keys) {
        initial_guess_for_optimizer.insert(k, global_estimates_.at(k));
        separator_estimates_added++;
    }

    if (config_.verbose) {
        std::cout << "    estimate size: " << estimates_.at(rid).size() << std::endl;
        std::cout << "    Separator estimates added to guess: " << separator_estimates_added << std::endl;
        std::cout << "    Total initial guess size for optimizer: " << initial_guess_for_optimizer.size() << std::endl;        
    }


    LevenbergMarquardtParams params; 
    LevenbergMarquardtOptimizer optimizer(robot.graph, initial_guess_for_optimizer, params);
    if (config_.verbose)
        std::cout << "    Optimizer constructed." << std::endl;

    // Run optimization
    gtsam::Values optimization_result = optimizer.optimize();
    if (config_.verbose)
        std::cout << "    Optimization successful." << std::endl;

    // IMPORTANT: Update only the estimates for the keys this robot actually owns.
    // Do not overwrite estimates_[rid] directly with optimization_result,
    // as optimization_result might contain updated values for separator keys (owned by others).
    for(Key local_key : robot.local_keys) {
        if (optimization_result.exists(local_key)) {
                estimates_.at(rid).update(local_key, optimization_result.at(local_key));
        } else {
                std::cerr << "    Warning: Optimization result missing estimate for local key " << DefaultKeyFormatter(local_key) << " for Robot " << rid << std::endl;
        }
    }
    if (config_.verbose)
        std::cout << "    Local estimates updated." << std::endl;

}

void DecentralizedOptimizer::saveResultsToFile(const std::string& path) const {
    std::ofstream out(path);
    if (!out.is_open()) {
        std::cerr << "Could not write results to: " << path << std::endl;
        return;
    }

    for (const auto& [rid, values] : estimates_) {
        for (const auto& kv : values) {
            Key key = kv.key;
            const auto& pose = values.at<Pose2>(key);
            out << robotIdFromKey(key) << " " << poseIdFromKey(key) << " "
                << std::fixed << std::setprecision(6)
                << pose.x() << " " << pose.y() << " " << pose.theta() << "\n";
        }
    }
    if (config_.verbose)
        std::cout << "Saved decentralized results to " << path << std::endl;
}
