#pragma once
#include <torch/torch.h>

#include "SimDota2/include/SimDota2.h"
#include "ActorCritic.h"
#include "RewardNetwork.h"

#include <vector>

class A3CEnv
{
public:
    A3CEnv(std::shared_ptr<ActorCritic> model_);
    ~A3CEnv() {
        if (engine) {
            delete engine;
            engine = nullptr;
        }
        teacher_mode = false;
    }

    void clear_state() {
        frame_count = 0;
        fisrt_loop = true;
        states.clear();
        actions.clear();
        values.clear();
        expected_reward.clear();
        one_hot_tensor.clear();
        rewards.clear();
        hp_rewards.clear();
        exp_rewards.clear();
        log_probs.clear();
        reduced_reward.clear();
        teacher_mode = false;
    }

    void reset();

    bool step(bool debug_print = false, bool default_action=false);

    void ppo_train(bool print_msg = false);

    void prepare_train();

    void print_summary();

    void update();

private:
    std::shared_ptr<ActorCritic> model;
    std::vector<torch::Tensor> states;
    std::vector<torch::Tensor> actions;
    std::vector<torch::Tensor> values;
    std::vector<torch::Tensor> expected_reward;
    std::vector<torch::Tensor> one_hot_tensor;
    std::vector<float> rewards;
    std::vector<float> hp_rewards;
    std::vector<float> exp_rewards;
    std::vector<torch::Tensor> log_probs;
    std::vector<float> reduced_reward;

    int frame_count;

    torch::Tensor last_tensor;

    float df_hp;//disconting factor wrt hp
    float df_exp;//disconting factor wrt hp

    bool fisrt_loop;

    cppSimulatorImp* engine;
    Config* config;

    float total_exp;
    float total_hp;
    float total_reward;

    std::shared_ptr<torch::nn::Module> old_model;
    torch::Tensor reward_tensor;

    bool teacher_mode;
};