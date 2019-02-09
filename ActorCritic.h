#pragma once
#include <torch/torch.h>
#include <string>

class ActorCritic : public torch::nn::Cloneable<ActorCritic>
{
public:
    ActorCritic(int dim_in, int dim_out, int dim_hidden)
        :hidden_dim(dim_hidden), fc1(dim_in, dim_hidden), actor_linear(dim_hidden, dim_out),
        critic_linear(dim_hidden, 1){
        reset();

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

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        for (auto& fc : vec_fc) {
            x = torch::relu(fc->forward(x));
        }
        auto actor_out = actor_linear->forward(x);
        torch::Tensor critic_out = critic_linear->forward(x);
        return std::pair<torch::Tensor, torch::Tensor>(actor_out, critic_out);
    }

    virtual void reset() override {
        vec_fc.clear();
        register_module("fc_first", fc1);
        for (int i = 0; i < 2; ++i) {
            auto fc = torch::nn::Linear(hidden_dim, hidden_dim);
            register_module("fc_" + std::to_string(i), fc);
            vec_fc.push_back(fc);
        }

        register_module("actor_linear", actor_linear);
        register_module("critic_linear", critic_linear);
    }
private:
    int hidden_dim;
    torch::nn::Linear fc1;
    std::vector<torch::nn::Linear> vec_fc;
    torch::nn::Linear actor_linear;
    torch::nn::Linear critic_linear;
};