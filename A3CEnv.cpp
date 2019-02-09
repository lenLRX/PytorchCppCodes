#include "A3CEnv.h"
#include "util.h"

#include <sstream>

A3CEnv::A3CEnv(std::shared_ptr<ActorCritic> model_,
    std::shared_ptr<RewardNetwork> RNetwork)
    :model(model_), RNetwork(RNetwork), engine(nullptr), config(nullptr)
{
    df_hp = 0.9f;
    df_exp = 0.99f;
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

bool A3CEnv::step() {
    double time_ = engine->get_time();
    //std::cout << "engine time: " << time_ << std::endl;
    if (time_ > 200) {
        return false;
    }
    torch::Tensor x = torch::zeros({10});

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
        if (dis > 1200) {
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
            x[start_idx] = (loc.x - hero_loc.x) / 2000;
            x[start_idx + 1] = (loc.y - hero_loc.y)/ 2000;
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

    if (fisrt_loop) {
        std::cout << action_prob.reshape({1,-1}) << std::endl;
        std::cout << " value: " << toNumber<float>(value_out) << std::endl;
        fisrt_loop = false;
    }
    

    int idx = toNumber<int>(action);

    
    if (max_prob > 0.5)
    {
        auto max_action = torch::argmax(action_prob, 0);
        int max_idx = toNumber<int>(max_action);
        //std::cout << max_prob << std::endl;
        idx = max_idx;
    }

    hero->set_decision(move);
    pos_tup order = get_move_vec(idx);
    hero->set_move_order(order);
    //std::cout << "Tensor action " << idx << std::endl;
    actions.push_back(action);
    values.push_back(value_out);

    torch::Tensor one_hot_action = torch::zeros({ 9 });
    one_hot_action[idx] = 1;
    
    auto reward_net_out = RNetwork->forward(x.view({ -1 }), one_hot_action);
    one_hot_tensor.push_back(one_hot_action);
    rnet.push_back(reward_net_out);

    ///engine running
    engine->loop();
    ///engine running

    auto hero_loc2 = hero->get_location();
    float now_exp = hero->getData().exp;
    float now_hp = hero->get_HP();
    
    double dis1 = hero_loc.distance(pos_tup());
    double dis2 = hero_loc2.distance(pos_tup());
    float reward = (dis1 - dis2) / 100.0;
    if (dis1 < 1200) {
        reward = 0.0;
    }

    float exp_reward = now_exp > prev_exp ? 1 : 0;
    float hp_reward = now_hp < prev_hp ? -1 : 0;

    //std::cout << "reward " << reward << std::endl;
    exp_rewards.push_back(exp_reward);
    hp_rewards.push_back(hp_reward);
    rewards.push_back(reward);
    return true;
}

void A3CEnv::train(){
    std::vector<float> reduced_reward(rewards.size(), 0.0f);
    torch::Tensor loss = torch::zeros({1});

    float temp_r_exp = 0.0f;
    float total_exp = 0.0f;
    float temp_r_hp = 0.0f;
    float total_hp = 0.0f;
    for (int i = rewards.size() - 1;i >= 0;--i)
    {
        total_exp += exp_rewards[i];
        total_hp += hp_rewards[i];

        temp_r_exp = df_exp * temp_r_exp + exp_rewards[i];
        temp_r_hp = df_hp * temp_r_hp + hp_rewards[i];
        reduced_reward[i] = temp_r_exp + temp_r_hp + rewards[i];
    }

    float total_reward = 0.0;

    std::vector<int64_t> reward_size;
    reward_size.push_back(reduced_reward.size());

    torch::Tensor reward_tensor = torch::from_blob(reduced_reward.data(), c10::IntList(reward_size));
    total_reward = toNumber<float>(reward_tensor.sum());
    
    const torch::Tensor vec_value = torch::stack(values).view({-1});
    
    torch::Tensor vec_value_detach = vec_value;
    vec_value_detach = vec_value_detach.detach();

    torch::Tensor vec_adv = reward_tensor - vec_value_detach;
    torch::Tensor vec_one_hot = torch::stack(one_hot_tensor);
    torch::Tensor vec_log_prob = torch::stack(log_probs);

    
    torch::Tensor actor_loss = -(vec_log_prob * vec_one_hot).sum({ 1 });

    actor_loss = actor_loss * vec_adv;

    actor_loss = actor_loss.sum();

    //actor
    /*
    torch::Tensor actor_loss = torch::zeros({ 1 });
    for (size_t i = 0; i < rewards.size(); ++i) {
        float reward_i = reduced_reward[i];
        auto adv = reward_i - toNumber<float>(values[i]);
        actor_loss += (-log_probs[i].view({-1})[toNumber<int>(actions[i])]) * adv;

        total_reward += reward_i;
    }
    */

    torch::Tensor critic_loss = reward_tensor - vec_value;
    
    critic_loss = critic_loss * critic_loss;
    //std::cout << critic_loss << ", " << vec_value.requires_grad() << std::endl;
    critic_loss = critic_loss.sum();

    torch::Tensor vec_rnet = torch::stack(rnet).view({ -1 });

    torch::Tensor rnet_loss = reward_tensor - vec_rnet;

    rnet_loss = rnet_loss * rnet_loss;
    //std::cout << rnet_loss << std::endl;
    rnet_loss = rnet_loss.sum();
    

    //critic
    /*
    torch::Tensor critic_loss = torch::zeros({ 1, 1 });
    torch::Tensor rnet_loss = torch::zeros({ 1 });
    for (size_t i = 0; i < rewards.size(); ++i) {
        auto d = reduced_reward[i] - values[i];
        critic_loss += d * d;

        auto d2 = reduced_reward[i] - rnet[i];
        rnet_loss += d2 * d2;
    }
    */

    critic_loss.reshape({ 1 });
    
    actor_loss /= (float)rewards.size();
    critic_loss /= (float)rewards.size();
    rnet_loss /= (float)rewards.size();

    std::stringstream ss;

    ss << "actor loss " << toNumber<float>(actor_loss)
        << " critic loss " << toNumber<float>(critic_loss)
        << " rnet loss " << toNumber<float>(rnet_loss)
        << " total reward " << total_reward
        << " exp reward " << total_exp
        << " hp reward " << total_hp;

    std::cout << ">>last tensor " << std::endl;
    std::cout << last_tensor.reshape({1, -1}) << std::endl;
    std::cout << ">>last tensor " << std::endl;

    std::cout << ss.str() << std::endl;

    //std::cout << "last state: " << states.back() << std::endl;
    //std::cout << vec_adv << std::endl;

    loss = actor_loss + critic_loss;
    
    loss.backward();
    rnet_loss.backward();
}
