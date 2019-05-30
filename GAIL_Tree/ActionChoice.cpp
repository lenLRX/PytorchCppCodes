#include "ActionChoice.h"
#include "MovingMode.h"
#include "LastHitMode.h"

#include "util.h"

ActionChoice::ActionChoice(int input_dim, int hidden_dim, int output_dim) :
    ModelNode(input_dim, hidden_dim, output_dim) {
    children.push_back(
        std::dynamic_pointer_cast<ModelNode>(std::make_shared<MovingMode>()));
    children.push_back(
        std::dynamic_pointer_cast<ModelNode>(std::make_shared<LastHitMode>()));
}

void ActionChoice::step_impl(cppSimulatorImp* engine, bool default_action) {
    auto hero = engine->getHero("Radiant", 0);

    auto hero_loc = hero->get_location();

    torch::Tensor x = state_encoding(engine);

    states.push_back(x);

    auto out = actor_->forward(x);

    auto action_prob = torch::softmax(out, 0);
    float max_prob = toNumber<float>(torch::max(action_prob));

    auto action = action_prob.multinomial(1);

    int idx = toNumber<int>(action);

    int default_idx = get_default(x);

    if (default_action)
    {
        idx = default_idx;
    }

    if (states.size() == 1) {
        auto action_logger = spdlog::get("action_logger");
        action_logger->info("{} first tick expert action {}\nmy action\n{}\n-------",
                type(), default_idx, torch_to_string(action_prob));
    }


    VecSpriteDist near_by_enemy = engine->get_nearby_enemy(hero, 2000);
    if (near_by_enemy.empty()) {
        idx = 0;
    }

    torch::Tensor expert_one_hot_action = torch::zeros({ output_dim_ });
    expert_one_hot_action[default_idx] = 1;

    expert_actions.push_back(expert_one_hot_action);

    torch::Tensor one_hot_action = torch::zeros({ output_dim_ });
    one_hot_action[idx] = 1;
    actor_actions.push_back(one_hot_action);

    hero->set_decision(idx + 1);

    children[idx]->step(engine, default_action, last_tick);
}

int ActionChoice::get_default(torch::Tensor x)
{
    float dist_ally_creep_x = toNumber<float>(x[2]) * near_by_scale;
    float dist_ally_creep_y = toNumber<float>(x[3]) * near_by_scale;

    float dist_creep_x = toNumber<float>(x[4]) * near_by_scale;
    float dist_creep_y = toNumber<float>(x[5]) * near_by_scale;

    float dist_tower_x = toNumber<float>(x[8]) * near_by_scale;
    float dist_tower_y = toNumber<float>(x[9]) * near_by_scale;


    if (hypot(dist_tower_x, dist_tower_y) < 1200) {
        return 0;
    }

    if (dist_creep_x < 500) {
        return 1;
    }

    return 0;
}

void ActionChoice::train(const std::vector<PackedData>& data, int episode) {
    if (data.empty()) {
        return;
    }

    if (!is_cuda()) {
        return;
    }

    torch::optim::SGD actor_optim(actor_->parameters(),
                                  torch::optim::SGDOptions(lr));

    torch::optim::SGD critic_optim(critic_->parameters(),
                                   torch::optim::SGDOptions(lr));

    torch::optim::SGD d_optim(discriminator_->parameters(),
                              torch::optim::SGDOptions(lr));
    for (auto& d : data) {
        if (d.empty()) {
            continue;
        }
        d_optim.zero_grad();

        torch::Tensor state = d.at("state").to(torch::kCUDA);
        torch::Tensor expert_action = d.at("expert_action").to(torch::kCUDA);
        torch::Tensor actor_one_hot = d.at("actor_one_hot").to(torch::kCUDA);
        torch::Tensor reward = d.at("reward").to(torch::kCUDA);

        torch::Tensor expert_prob = discriminator_->forward(state, expert_action);
        torch::Tensor expert_label = torch::ones_like(expert_prob);
        torch::Tensor expert_d_loss = torch::binary_cross_entropy(expert_prob, expert_label).mean();

        torch::Tensor actor_action_prob = torch::softmax(actor_->forward(state), 1);
        torch::Tensor actor_prob = discriminator_->forward(state, (actor_action_prob * actor_one_hot).detach());
        torch::Tensor actor_label = torch::zeros_like(actor_prob);
        torch::Tensor actor_d_loss = torch::binary_cross_entropy(actor_prob, actor_label).mean();

        torch::Tensor prob_diff = torch::relu(expert_prob - actor_prob).detach();

        torch::Tensor total_d_loss = expert_d_loss + actor_d_loss;

        total_d_loss.backward();

        d_optim.step();

        actor_optim.zero_grad();
        critic_optim.zero_grad();

        torch::Tensor actor_prob2 = discriminator_->forward(state, actor_action_prob * expert_action);
        torch::Tensor actor_label2 = torch::ones_like(actor_prob2);

        torch::Tensor actor_d_loss2 = (torch::relu(torch::binary_cross_entropy(actor_prob2, actor_label2) - 0.1))*prob_diff.mean();

        torch::Tensor values = critic_->forward(state);
        torch::Tensor critic_loss = reward - values;
        critic_loss = critic_loss * critic_loss;
        critic_loss = critic_loss.mean();

        torch::Tensor adv = reward - values.detach();

        torch::Tensor actor_log_probs = torch::log(torch::sum(actor_action_prob * actor_one_hot, 1));
        torch::Tensor actor_loss = -actor_log_probs * adv;
        actor_loss = actor_loss.mean();

        torch::Tensor total_loss = actor_d_loss2 + actor_loss + critic_loss;
        total_loss.backward();

        auto logger = spdlog::get("loss_logger");

        logger->info("episode {} {}, training loss: {} reward {}", episode, type(), toNumber<float>(total_loss),
                toNumber<float>(reward.mean()));

        critic_optim.step();
        actor_optim.step();
    }
}