#pragma once
#include <torch/torch.h>

#include "SimDota2/include/SimDota2.h"

#include "Actor.h"
#include "Critic.h"
#include "DisCriminator.h"

template<typename T>
T toNumber(torch::Tensor x) {
    return x.item().to<T>();
}

struct PackedData {
    torch::Tensor state;
    torch::Tensor expert_action;
    torch::Tensor actor_one_hot;
    torch::Tensor reward;
};

class Env
{
public:
    Env(std::shared_ptr<Actor> actor_model,
        std::shared_ptr<Critic> critic_model,
        std::shared_ptr<DisCriminator> d_model);

    ~Env (){
        if (engine) {
            delete engine;
            engine = nullptr;
        }
    }

    void clear_state() {
        fisrt_loop = true;

        total_exp = 0.0;
        total_hp = 0.0;

        states.clear();
        actions.clear();
        expert_actions.clear();
        values.clear();
        one_hot_tensor.clear();
        hp_rewards.clear();
        exp_rewards.clear();
        log_probs.clear();
        reduced_reward.clear();
        actor_prob_with_grad.clear();
        vec_actor_prob.clear();
        vec_expert_prob.clear();
    }

    void reset();

    bool step(bool debug_print = false, bool default_action = false);

    torch::Tensor train_discriminator(bool print_msg = false);

    torch::Tensor train_actor(bool print_msg = false);

    torch::Tensor train_critic(bool print_msg = false);

    torch::Tensor calculate_reward();

    void update_param();

    PackedData prepare_data();

    void evaluate();

    std::string record();

    void print_summary();

    void update();

    torch::Tensor overall_loss;

private:
    cppSimulatorImp* engine;
    Config* config;
    bool fisrt_loop;

    float total_exp;
    float total_hp;

    float df_hp;
    float df_exp;

    std::shared_ptr<Actor> actor_model;
    std::shared_ptr<Critic> critic_model;
    std::shared_ptr<DisCriminator> d_model;

    std::shared_ptr<Actor> actor_model_master;
    std::shared_ptr<Critic> critic_model_master;
    std::shared_ptr<DisCriminator> d_model_master;

    std::shared_ptr<torch::nn::Module> old_actor_model;

    std::vector<torch::Tensor> states;
    std::vector<torch::Tensor> actions;
    std::vector<torch::Tensor> expert_actions;
    std::vector<torch::Tensor> values;
    std::vector<torch::Tensor> one_hot_tensor;
    std::vector<torch::Tensor> actor_prob_with_grad;
    std::vector<torch::Tensor> vec_actor_prob;
    std::vector<torch::Tensor> vec_expert_prob;
    std::vector<float> hp_rewards;
    std::vector<float> exp_rewards;
    std::vector<torch::Tensor> log_probs;
    std::vector<float> reduced_reward;

    torch::Tensor last_action_prob;
    int last_expert_action;
    torch::Tensor last_expert_prob;
    torch::Tensor last_actor_prob;
    torch::Tensor prob_diff;

};
