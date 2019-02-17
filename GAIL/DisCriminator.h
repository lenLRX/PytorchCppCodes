#pragma once
#include <torch/torch.h>
#include <string>
class DisCriminator : public torch::nn::Cloneable<DisCriminator>
{
public:
    DisCriminator(int state_dim, int action_dim, int hidden_dim)
    : state_dim(state_dim), action_dim(action_dim), hidden_dim(hidden_dim),
        fc1(nullptr), fc2(nullptr)
    {
        reset();
        for (auto& param : named_parameters())
        {
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
        torch::Tensor input = torch::cat({ state, action }, state.dim() - 1);
        
        torch::Tensor o = torch::tanh(fc1->forward(input));
        o = torch::sigmoid(fc2->forward(o));
        return o;
    }

    virtual void reset() override {
        fc1 = torch::nn::Linear(state_dim + action_dim, hidden_dim);
        fc2 = torch::nn::Linear(hidden_dim, 1);
        register_module("fc_1", fc1);
        register_module("fc_2", fc2);
    }

    int state_dim;
    int action_dim;
    int hidden_dim;
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
};
