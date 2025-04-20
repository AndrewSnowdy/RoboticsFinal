#include "mesa/dataset.h"
#include "mesa/admm_optimizer.h"
#include "mesa/decentralized.h"


void runLocalOptimize(const Dataset& dataset, const std::string& output_dir);


int main() {
    Dataset dataset("../data/poses.txt", "../data/edges.txt");
    auto subgraphs = dataset.splitByRobot();

    //Local Optimization
    runLocalOptimize(dataset, "../data/local_optimized.txt");


    //Decentralized Optimization
    //1
    //config.max_iterations = 10
    //config.prior_strength = 0.05
    //2
    //config.max_iteration = 10
    ///config.prior_strength = 0.05
    DecentralizedConfig dec_config;
    dec_config.max_iterations = 9;
    dec_config.prior_strength = 0.025; //represents a rather low confidence, priorsigmas(1/prior_strength)
    dec_config.verbose = false;
    DecentralizedOptimizer dec_solver(subgraphs, dec_config);
    dec_solver.run();
    dec_solver.saveResultsToFile("../data/decentralized_optimization.txt");


    //ADMM Optimization
    AdmmConfig admm_config;
    AdmmOptimizer admm_solver(subgraphs, admm_config);
    admm_solver.run();
    admm_solver.saveResultsToFile("../data/admm_optimized.txt");


    for (const auto& [rid, robot] : subgraphs) {
        std::cout << "Robot " << rid << ":\n";
        std::cout << " Poses: " << robot.local_keys.size() << "\n";
        std::cout << " Factors: " << robot.graph.size() << "\n";
        std::cout << " Separators: " << robot.separator_keys.size() << "\n";
    }
    
    
    return 0;
}
