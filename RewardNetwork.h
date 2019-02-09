#pragma once
#include <torch/torch.h>

class RewardNetwork : public torch::nn::Module
{
public:
    RewardNetwork(int input_size, int hidden_size)
    : reward_fc1(input_size, hidden_size),
      reward_fc2(hidden_size, 1) {
        register_module("reward_fc1", reward_fc1);
        register_module("reward_fc2", reward_fc2);

        for (auto& param : named_parameters())
        {
            std::cout << param.key() << ", " << param.value().sizes() << std::endl;
            int dim_ = param.value().dim();
            if (dim_ == 2) {
                torch::nn::init::xavier_normal_(param.value());
            }
            else {
                torch::nn::init::constant_(param.value(), 0);
            }
        }
    }

    torch::Tensor forward(torch::Tensor state, torch::Tensor action) {
        auto x = torch::cat({ state.view({-1}), action.view({-1}) });
        x = torch::relu(reward_fc1->forward(x));
        x = reward_fc2->forward(x);
        return x;
    }

private:
    torch::nn::Linear reward_fc1;
    torch::nn::Linear reward_fc2;
};