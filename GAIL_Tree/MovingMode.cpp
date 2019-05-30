#include "MovingMode.h"
#include "util.h"

static int get_default(const torch::Tensor& x) {
    float dist_ally_creep_x = toNumber<float>(x[2]) * near_by_scale;
    float dist_ally_creep_y = toNumber<float>(x[3]) * near_by_scale;

    float dist_creep_x = toNumber<float>(x[4]) * near_by_scale;
    float dist_creep_y = toNumber<float>(x[5]) * near_by_scale;

    float dist_tower_x = toNumber<float>(x[8]) * near_by_scale;
    float dist_tower_y = toNumber<float>(x[9]) * near_by_scale;

    if (hypot(dist_tower_x, dist_tower_y) < 1200) {
        return 1;
    }

    if (fabs(dist_creep_x - near_by_scale) < 0.1 && fabs(dist_creep_y - near_by_scale) < 0.1) {
        return 5;
    }

    if (fabs(dist_ally_creep_x - near_by_scale) < 0.1 && fabs(dist_ally_creep_y - near_by_scale) < 0.1) {
        return 1;
    }


    float mid_x = 0.5f * (dist_ally_creep_x + dist_creep_x);
    float mid_y = 0.5f * (dist_ally_creep_y + dist_creep_y);

    float _500_edge = 353.5f;

    float target_point_x = mid_x - _500_edge;
    float target_point_y = mid_y - _500_edge;

    auto dist2target = hypot(target_point_x, target_point_y);

    if (dist2target < 50) {
        return 0;
    }

    int move_dir = get_move_dir(target_point_x, target_point_y);

    return move_dir;
}

void MovingMode::step_impl(cppSimulatorImp * engine, bool default_action)
{
    torch::Tensor x = torch::ones({ 10 });
    auto hero = engine->getHero("Radiant", 0);

    auto hero_loc = hero->get_location();
    //std::cout << "hero pos " << hero_loc.toString() << std::endl;

    x[0] = hero_loc.x / 7000;
    x[1] = hero_loc.y / 7000;

    float ally_creep_dis = 0.0;
    float ally_tower_dis = 0.0;
    float enemy_creep_dis = 0.0;
    float enemy_tower_dis = 0.0;

    for (auto* s : engine->get_sprites()) {
        int start_idx = -1;
        float dis = s->get_location().distance(hero_loc);
        if (dis > 2000) {
            continue;
        }
        if (s->get_UnitType() == UNITTYPE_LANE_CREEP) {
            if (s->get_side() == hero->get_side()) {
                if (ally_creep_dis == 0 || dis < ally_creep_dis) {
                    ally_creep_dis = dis;
                    start_idx = 2;
                }
            }
            else {
                if (enemy_creep_dis == 0 || dis < enemy_creep_dis) {
                    enemy_creep_dis = dis;
                    start_idx = 4;
                }
            }
        }
        else if (s->get_UnitType() == UNITTYPE_TOWER) {
            if (s->get_side() == hero->get_side()) {
                if (ally_tower_dis == 0 || dis < ally_tower_dis) {
                    ally_tower_dis = dis;
                    start_idx = 6;
                }
            }
            else {
                if (enemy_tower_dis == 0 || dis < enemy_tower_dis) {
                    enemy_tower_dis = dis;
                    start_idx = 8;
                }
            }
        }
        if (start_idx > 0) {
            const auto& loc = s->get_location();
            x[start_idx] = (loc.x - hero_loc.x) / near_by_scale;
            x[start_idx + 1] = (loc.y - hero_loc.y) / near_by_scale;
        }

    }

    //std::cout << "Tensor input " << x << std::endl;
    states.push_back(x);

    auto out = actor_->forward(x);

    auto action_log_prob = torch::log_softmax(out, 0);

    auto action_prob = torch::softmax(out, 0);

    auto action = action_prob.multinomial(1);

    int idx = toNumber<int>(action);

    int default_idx = get_default(x);

    if (default_action)
    {
        idx = default_idx;
    }

    torch::Tensor one_hot_action = torch::zeros({ 9 });
    one_hot_action[idx] = 1;
    actor_actions.push_back(one_hot_action);

    torch::Tensor expert_one_hot_action = torch::zeros({ 9 });
    expert_one_hot_action[default_idx] = 1;

    expert_actions.push_back(expert_one_hot_action);

    hero->set_decision(move);
    pos_tup order = get_move_vec(idx);
    hero->set_move_order(order);
}

void MovingMode::train(const std::vector<PackedData>& data, int episode) {
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

        logger->info("episode {} {}, training loss: {}", episode, type(), toNumber<float>(total_loss));

        critic_optim.step();
        actor_optim.step();
    }
}
