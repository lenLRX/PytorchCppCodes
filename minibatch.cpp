#include "ActorCritic.h"
#include "MiniEnv.h"

#include <stdio.h>
#include <iostream>

int main(int argc, char** argv) {

    /*
        [hero.x hero.y, 
        nearest ally creep xy,  -- others are relative distance
        nearest enemy creep xy
        nearest ally Tower xy,
        nearest enemy Tower xy]
    */
    auto shared_net = std::make_shared<ActorCritic>(10, 9, 64);
    torch::optim::SGD optim(shared_net->parameters(),
        torch::optim::SGDOptions(1E-3));

    std::vector<A3CEnv> vec_env;
    for (int i = 0; i < 3; ++i) {
        vec_env.emplace_back(shared_net);
    }
    A3CEnv Test_net(shared_net);
    int count = 0;
 
    while (true) {
        try {
            bool debug_print = true;
            for (auto& env : vec_env) {
                count++;
                env.reset();
                while (env.step(debug_print, count % 2));

                debug_print = false;
                env.prepare_train();
            }

            for (int i = 0; i < 3; ++i) {
                optim.zero_grad();
                for (int j = 0; j < vec_env.size(); ++j) {
                    auto& env = vec_env[j];
                    env.ppo_train((j == 0) && (i == 0));
                }
                optim.step();
            }
            vec_env.front().print_summary();
            std::cerr << "test case >>>>>>>>>>>>>" << std::endl;
            {
                Test_net.reset();
                while (Test_net.step(true, false));
                Test_net.print_summary();
            }
            std::cerr << "test case <<<<<<<<<<<<<" << std::endl;
        }
        catch (const std::runtime_error& e) {
            std::cerr << e.what() << std::endl;
        }
        
    }
    
    
    getchar();
    return 0;
}