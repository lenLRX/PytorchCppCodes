#include "ActorCritic.h"
#include "A3CEnv.h"

#include <stdio.h>
#include <iostream>

int main(int argc, char** argv) {
    /*
    try {
       
    }
    catch (const c10::Error& e) {
        std::cerr << "got exception: " << e.msg() << std::endl;
        throw;
    }
    */

    /*
        [hero.x hero.y, 
        nearest ally creep xy,  -- others are relative distance
        nearest enemy creep xy
        nearest ally Tower xy,
        nearest enemy Tower xy]
    */
    auto shared_net = std::make_shared<ActorCritic>(10, 9, 16);
    auto reward_net = std::make_shared<RewardNetwork>(9 + 10, 16);

    double lr = 1E-4;

    torch::optim::SGD optimizer(shared_net->parameters(),
        torch::optim::SGDOptions(lr));

    torch::optim::SGD reward_optim(reward_net->parameters(),
        torch::optim::SGDOptions(lr));

    std::vector<A3CEnv> envs;
    for (int i = 0; i < 1; ++i) {
        A3CEnv env(shared_net, reward_net);
        envs.push_back(env);
    }
    while (true) {
        reward_optim.zero_grad();
        optimizer.zero_grad();
        for (auto& env : envs) {
            env.reset();
            while (env.step());

            env.train();
        }
        std::cout << "update param" << std::endl;
        reward_optim.step();
        optimizer.step();
    }
    
    
    getchar();
    return 0;
}