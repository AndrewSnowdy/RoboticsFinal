

#include "mesa/admm_optimizer.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <set>

using namespace gtsam;

AdmmOptimizer::AdmmOptimizer(const std::map<RobotId, RobotSubgraph>& subgraphs, const AdmmConfig& config)
    : robots_(subgraphs), config_(config), beta_(config.beta) {
    for (const auto& [rid, robot] : robots_) {
        estimates_[rid] = robot.initial;
        base_graphs_[rid] = robot.graph;
    }
    initializeAdmmData();
}

//initialize z and lambda
void AdmmOptimizer::initializeAdmmData() {
    separator_info_.clear();
    for (const auto& [rid_a, subgraph_a] : robots_) {
        for (const auto& [rid_b, subgraph_b] : robots_) {
            if (rid_a >= rid_b) continue; //improve efficiency

            //creates single, order-independent key
            RobotId r_min = std::min(rid_a, rid_b);
            RobotId r_max = std::max(rid_a, rid_b);
            RobotPair current_pair = {r_min, r_max};

            // keys owned by A, separator for B
            for (const gtsam::Key key_a : subgraph_a.local_keys) {
                if (subgraph_b.separator_keys.count(key_a)) {
                    if (!estimates_.at(rid_a).exists(key_a)) continue;
                    SeparatorData data;
                    data.z = estimates_.at(rid_a).at<Pose2>(key_a);
                    data.lambda = Vector3::Zero();
                    separator_info_[current_pair][key_a] = data; //adds new to separator_info
                }
            }
            // keys owned by B, separator for A
            for (const gtsam::Key key_b : subgraph_b.local_keys) {
                if (subgraph_a.separator_keys.count(key_b)) {
                    if (!estimates_.at(rid_b).exists(key_b)) continue;
                    if (separator_info_.count(current_pair) && separator_info_.at(current_pair).count(key_b)) continue;
                    SeparatorData data;
                    data.z = estimates_.at(rid_b).at<Pose2>(key_b);
                    data.lambda = Vector3::Zero();
                    separator_info_[current_pair][key_b] = data;
                }
            }

        }
    }
}

void AdmmOptimizer::run() {
    for (size_t iter = 0; iter < config_.max_iterations; ++iter) {
        std::cout << "Admm [Iteration " << iter << "]\n";
        exchangeAndConsensusStep();

        for (const auto& [rid, _] : robots_) {
            runLocalOptimization(rid);
        }
    }
}


void AdmmOptimizer::exchangeAndConsensusStep() {
    // update lambda based on owners deviation from PREVIOUS consensus z
    if (config_.verbose) 
        std::cout << "  Updating Lambda" << std::endl;

    int lambda_updates_made = 0;
    for (auto& [pair, key_data_map] : separator_info_) { //need non-const data
        for (auto& [k, data] : key_data_map) {           
            RobotId owner_rid = robotIdFromKey(k);
            if (estimates_.count(owner_rid) && estimates_.at(owner_rid).exists(k)) {
                // compare owner's current estimate to the CURRENT z value (from end of last iteration)
                const Pose2& theta_owner = estimates_.at(owner_rid).at<Pose2>(k);
                // calculate deviation in tangent space
                Vector3 q_deviation = Pose2::Logmap(theta_owner.between(data.z));
                // update lambda: accumulate scaled deviation
                data.lambda += beta_ * q_deviation;
                lambda_updates_made++;
            }
        }
    }

     if (config_.verbose) 
        std::cout << "    updated Lambda for " << lambda_updates_made << " entrie" << std::endl;

     if (config_.verbose) 
        std::cout << "  updating Z " << std::endl;

    int z_updates_made = 0;
    for (auto& [pair, key_data_map] : separator_info_) {
        for (auto& [k, data] : key_data_map) { // need non-const data
            RobotId owner_rid = robotIdFromKey(k);
            if (estimates_.count(owner_rid) && estimates_.at(owner_rid).exists(k)) {
                // update z to track the owner's most recent estimate
                data.z = estimates_.at(owner_rid).at<Pose2>(k);
                z_updates_made++;
            }
        }
    }

    if (config_.verbose) 
        std::cout << "    Updated Z for " << z_updates_made << " entries" << std::endl;

    // local optimization
    if (config_.verbose) 
        std::cout << "  Preparing graphs with biased priors" << std::endl;

    noiseModel::Gaussian::shared_ptr prior_noise =
        noiseModel::Diagonal::Sigmas(Vector3::Constant(1.0 / std::sqrt(beta_))); 

    for (auto& [rid, robot] : robots_) {
        // Reset graph
        robot.graph = base_graphs_.at(rid);
        int priors_added = 0;
        // Add biased priors based on UPDATED z and lambda
        for (const auto& [pair, key_data_map] : separator_info_) {
            if (rid == pair.first || rid == pair.second) {
                for (const auto& [k, data] : key_data_map) {
                    // add prior for shared variable k, using updated z and lambda
                    Vector3 bias = -data.lambda / beta_; // bias term
                    
                    Pose2 biased_z = data.z.retract(bias); // apply bias to consensus value
                    robot.graph.add(PriorFactor<Pose2>(k, biased_z, prior_noise));
                    priors_added++;

                }
            }
        }
        if (config_.verbose)
            std::cout << "    Robot " << rid << ": added " << priors_added << " biased priors." << std::endl;
    }
    if (config_.verbose) 
        std::cout << "  finished graph preparation." << std::endl;
}


void AdmmOptimizer::runLocalOptimization(RobotId rid) {
    if (config_.verbose) 
        std::cout << "  Running local optimization for Robot " << rid << std::endl;
    

    const auto& robot = robots_.at(rid);

    // Create initial guess including z values for separators
    gtsam::Values initial_guess = estimates_.at(rid);
    for (const auto& [pair, key_data_map] : separator_info_) {
            if (rid == pair.first || rid == pair.second) {
                for (const auto& [k, data] : key_data_map) {
                    if (!initial_guess.exists(k)) { // Check if it's a separator key
                        initial_guess.insert(k, data.z); // Use current z as guess
                    }
                }
            }
    }

    LevenbergMarquardtParams params;
    LevenbergMarquardtOptimizer optimizer(robot.graph, initial_guess, params);
    gtsam::Values optimization_result = optimizer.optimize();

    // Update only local estimates
    for(Key local_key : robot.local_keys) {
        if (optimization_result.exists(local_key)) {
                estimates_.at(rid).update(local_key, optimization_result.at(local_key));
        }
    }
    if (config_.verbose) 
        std::cout << "  Optimization complete for Robot " << rid << "." << std::endl;
}

void AdmmOptimizer::saveResultsToFile(const std::string& path) const {
    std::ofstream out(path);
    if (!out.is_open()) {
        std::cerr << "could not writee results to: " << path << std::endl;
        return;
    }
    out << std::fixed << std::setprecision(6);
    for (const auto& [rid, values] : estimates_) {
        for (const auto& kv : values) {
            Key key = kv.key;
            Pose2 pose = values.at<Pose2>(key);
            out << robotIdFromKey(key) << " " << poseIdFromKey(key) << " "
                << pose.x() << " " << pose.y() << " " << pose.theta() << "\n";
            
        }
    }
    std::cout << "saved simple ADMM results to " << path << std::endl;
}