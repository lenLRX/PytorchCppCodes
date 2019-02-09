#include "MiniEnv.h"
#include "util.h"

#include <sstream>

const float near_by_scale = 2000;

A3CEnv::A3CEnv(std::shared_ptr<ActorCritic> model_)
    :model(model_), engine(nullptr), config(nullptr)
{
    df_hp = 0.5f;
    df_exp = 0.99f;

    frame_count = 0;
}

pos_tup get_move_vec(int dir) {
    pos_tup out_(0,0);
    if (dir == 0) {
        return out_;
    }
    double rad = dir * M_PI / 4 - M_PI;
    out_.x = cos(rad) * 300;
    out_.y = sin(rad) * 300;
    return out_;
}

int get_default(torch::Tensor x) {
    float dist_creep_x = toNumber<float>(x[4]) * near_by_scale;
    float dist_creep_y = toNumber<float>(x[5]) * near_by_scale;

    float dist_tower_x = toNumber<float>(x[8]) * near_by_scale;
    float dist_tower_y = toNumber<float>(x[9]) * near_by_scale;

    if (hypot(dist_tower_x, dist_tower_y) < 1500) {
        return 1;
    }

    if (hypot(dist_creep_x, dist_creep_y) < 500) {
        return 1;
    }

    return 5;
}

void A3CEnv::reset() {
    if (engine) {
        delete engine;
        engine = nullptr;
    }

    if (!config) {
        config = ConfigCacheMgr<Config>::getInstance().get("F:\\PytorchCpp\\PytorchCpp\\SimDota2\\config\\Config.json");
    }
    engine = new cppSimulatorImp(config);
    clear_state();
}

bool A3CEnv::step(bool debug_print, bool default_action) {
    teacher_mode = default_action;
    double time_ = engine->get_time();
    //std::cout << "engine time: " << time_ << std::endl;
    if (time_ > 500) {
        return false;
    }
    torch::Tensor x = torch::ones({10});

    auto hero = engine->getHero("Radiant", 0);
    if (hero->isDead()) {
        return false;
    }
    auto hero_loc = hero->get_location();
    float prev_exp = hero->getData().exp;
    float prev_hp = hero->get_HP();
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
            x[start_idx + 1] = (loc.y - hero_loc.y)/ near_by_scale;
        }
        
    }

    //std::cout << "Tensor input " << x << std::endl;

    auto out = model->forward(x);
    states.push_back(x);
    auto action_out = out.first;
    auto value_out = out.second;
    auto action_log_prob = torch::log_softmax(action_out, 0);
    log_probs.push_back(action_log_prob);

    
    
    auto action_prob = torch::softmax(action_out, 0);
    float max_prob = toNumber<float>(torch::max(action_prob));
    
    auto action = action_prob.multinomial(1);

    last_tensor = action_prob;

    if (debug_print && fisrt_loop) {
        std::cout << "first >>>" << std::endl;
        std::cout << " action: " << action_prob.reshape({1,-1}) << std::endl;
        std::cout << " value: " << toNumber<float>(value_out) << std::endl;
        std::cout << "first <<" << std::endl;
        fisrt_loop = false;
    }
    

    int idx = toNumber<int>(action);

    
    if (max_prob > 0.9)
    {
        auto max_action = torch::argmax(action_prob, 0);
        int max_idx = toNumber<int>(max_action);
        //std::cout << max_prob << std::endl;
        idx = max_idx;
    }

    if (default_action) {
        idx = get_default(x);
    }
    

    hero->set_decision(move);
    pos_tup order = get_move_vec(idx);
    hero->set_move_order(order);
    //std::cout << "Tensor action " << idx << std::endl;
    actions.push_back(action);
    values.push_back(value_out);

    torch::Tensor one_hot_action = torch::zeros({ 9 });
    one_hot_action[idx] = 1;
    
    one_hot_tensor.push_back(one_hot_action);

    ///engine running
    engine->loop();
    ///engine running

    auto hero_loc2 = hero->get_location();
    float now_exp = hero->getData().exp;
    float now_hp = hero->get_HP();
    
    double dis1 = hero_loc.distance(pos_tup());
    double dis2 = hero_loc2.distance(pos_tup());
    float reward = (dis1 - dis2) / 1000.0;
    if (dis1 < 1200) {
        reward = 0.0;
    }
    reward = 0.0;

    float exp_reward = now_exp > prev_exp ? 0.2 : 0;
    float hp_reward = now_hp < prev_hp ? -0.2 : 0;

    //std::cout << "reward " << reward << std::endl;
    exp_rewards.push_back(exp_reward);
    hp_rewards.push_back(hp_reward);
    rewards.push_back(reward);

    ++frame_count;
    return true;
}

void A3CEnv::ppo_train(bool print_msg)
{
    torch::Tensor vec_state = torch::stack(states);
    auto new_out = model->forward(vec_state);
    torch::Tensor new_prob = torch::softmax(new_out.first, 1);
    torch::Tensor new_value = new_out.second.view({-1});

    auto old_out = std::dynamic_pointer_cast<ActorCritic>(old_model)->forward(vec_state);
    //auto old_out = model->forward(vec_state);
    torch::Tensor old_prob = torch::softmax(old_out.first, 1).detach();

    torch::Tensor vec_one_hot = torch::stack(one_hot_tensor);

    torch::Tensor ratio = torch::sum(new_prob * vec_one_hot, {1}) / torch::sum(old_prob * vec_one_hot, { 1 });
    torch::Tensor adv = reward_tensor - new_value.detach();
    torch::Tensor surr = ratio * adv;

    torch::Tensor clamped_ratio = torch::clamp(ratio, 0.8, 1.2);
    torch::Tensor actor_loss = torch::min(surr, clamped_ratio * adv);
    actor_loss = -actor_loss.mean();

    torch::Tensor critic_loss = reward_tensor - new_value;
    critic_loss = critic_loss * critic_loss;
    critic_loss = critic_loss.mean();

    torch::Tensor loss = actor_loss;
    //if (!teacher_mode) {
        loss += critic_loss;
    //}
    

    if (print_msg) {
        std::stringstream ss;

        ss << "actor loss " << toNumber<float>(actor_loss)
            << " critic loss " << toNumber<float>(critic_loss)
            << " teacher mode " << teacher_mode;
        std::cout << ss.str() << std::endl;
    }
    
    loss.backward(torch::nullopt, true);
}


void A3CEnv::prepare_train(){
    old_model = model->clone();
    
    reduced_reward = std::vector<float>(rewards.size(), 0.0f);
    torch::Tensor loss = torch::zeros({1});

    float temp_r_exp = 0.0f;
    total_exp = 0.0f;
    float temp_r_hp = 0.0f;
    total_hp = 0.0f;

    total_reward = 0.0;

    for (int i = rewards.size() - 1;i >= 0;--i)
    {
        total_exp += exp_rewards[i];
        total_hp += hp_rewards[i];

        temp_r_exp = df_exp * temp_r_exp + exp_rewards[i];
        temp_r_hp = df_hp * temp_r_hp + hp_rewards[i];
        reduced_reward[i] = temp_r_exp + temp_r_hp + rewards[i];
        total_reward += reduced_reward[i];
    }

    std::vector<int64_t> reward_size;
    reward_size.push_back(reduced_reward.size());

    reward_tensor = torch::from_blob(reduced_reward.data(), c10::IntList(reward_size));

    
}

void A3CEnv::print_summary()
{
    std::stringstream ss;

    ss << "total reward " << total_reward
        << " exp reward " << total_exp
        << " hp reward " << total_hp;

    std::cout << ss.str() << std::endl;

    std::cout << ">>last tensor " << std::endl;
    std::cout << last_tensor.reshape({ 1, -1 }) << std::endl;
    std::cout << ">>last tensor " << std::endl;
}
