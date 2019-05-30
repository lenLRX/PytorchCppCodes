#include "util.h"
#include "LastHitMode.h"

using namespace torch;

static Tensor creep_encoding(Hero* h, Sprite* s) {
    torch::Tensor x = torch::zeros({ 3 });
    auto hloc = h->get_location();
    auto sloc = s->get_location();
    x[0] = (sloc.x - hloc.x) / near_by_scale;
    x[1] = (sloc.y - hloc.y) / near_by_scale;
    x[2] = s->get_HP();
    return x;
}

static Tensor get_expert_action(Hero* h, Sprite* s) {
    torch::Tensor x = torch::zeros({ 1 });
    auto hloc = h->get_location();
    auto sloc = s->get_location();
    float dx = (sloc.x - hloc.x) / near_by_scale;
    float dy = (sloc.y - hloc.y) / near_by_scale;
    
    float dist_ = sqrtf(dx*dx + dy + dy);
    if (dist_ < 500 && s->get_HP() < 40) {
        x[0] = 1;
    }
    return x;
}

void LastHitMode::step_impl(cppSimulatorImp * engine, bool default_action)
{
    auto hero = engine->getHero("Radiant", 0);

    auto hero_loc = hero->get_location();

    VecSpriteDist near_by_enemy = engine->get_nearby_enemy(hero, 2000);
    VecSpriteDist near_by_creep;// no deny now

    std::vector<Tensor> states_cache;
    std::vector<Tensor> expert_action_cache;
    std::vector<Tensor> actor_action_cache;

    float max_value = 0.0;
    int idx = -1;
    int i = 0;

    int expert_idx = -1;

    // get the creep with highest score
    for (auto p : near_by_enemy) {
        auto x = creep_encoding(hero, p.first);
        states_cache.push_back(x);
        auto out = sigmoid(actor_->forward(x));
        actor_action_cache.push_back(out);
        auto out_value = toNumber<float>(out);

        if (out_value > max_value) {
            max_value = out_value;
            idx = i;
        }

        auto expert_act = get_expert_action(hero, p.first);
        expert_action_cache.push_back(expert_act);

        float expert_f = toNumber<float>(expert_act);
        if (expert_idx < 0 && expert_f > 0) {
            expert_idx = i;
        }

        i++;
    }

    if (idx < 0) {
        // we should crash here, because it means something wrong
        //hero->set_decision(decisonType::noop);
        hero->set_target(nullptr);
        return;
    }

    states.push_back(states_cache[idx]);
    actor_actions.push_back(actor_action_cache[idx]);
    expert_actions.push_back(expert_action_cache[idx]);

    if (default_action && expert_idx >= 0) {
        hero->set_target(near_by_enemy[expert_idx].first);
    }
    else {
        hero->set_target(near_by_enemy[idx].first);
    }
    

}

void LastHitMode::train(const std::vector<PackedData> &data, int episode) {
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
        d_optim.zero_grad();

        if (d.empty()) {
            continue;
        }

        torch::Tensor state = d.at("state").to(torch::kCUDA);
        torch::Tensor expert_action = d.at("expert_action").to(torch::kCUDA);
        //torch::Tensor actor_one_hot = d.at("actor_one_hot").to(torch::kCUDA);
        torch::Tensor reward = d.at("reward").to(torch::kCUDA);

        torch::Tensor expert_prob = discriminator_->forward(state, expert_action);
        torch::Tensor expert_label = torch::ones_like(expert_prob);
        torch::Tensor expert_d_loss = torch::binary_cross_entropy(expert_prob, expert_label).mean();

        torch::Tensor actor_action_prob = torch::softmax(actor_->forward(state), 1);
        //torch::Tensor actor_prob = discriminator_->forward(state, (actor_action_prob * actor_one_hot).detach());
        torch::Tensor actor_prob = discriminator_->forward(state, actor_action_prob.detach());
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

        //torch::Tensor actor_log_probs = torch::log(torch::sum(actor_action_prob * actor_one_hot, 1));
        torch::Tensor actor_log_probs = torch::log(torch::sum(actor_action_prob, 1));
        torch::Tensor actor_loss = -actor_log_probs * adv;
        actor_loss = actor_loss.mean();

        torch::Tensor total_loss = actor_d_loss2 + actor_loss + critic_loss;
        total_loss.backward();

        auto logger = spdlog::get("loss_logger");

        logger->info("episode {} {}, training loss: {}", episode, type(), toNumber<float>(total_loss));

        critic_optim.step();
        actor_optim.step();
    }
}
