#include "Env.h"

const float near_by_scale = 2000;


Env::Env(std::shared_ptr<Actor> actor_model_,
    std::shared_ptr<Critic> critic_model_, 
    std::shared_ptr<DisCriminator> d_model_)
    :actor_model_master(actor_model_),
    critic_model_master(critic_model_),
    d_model_master(d_model_), engine(nullptr), config(nullptr)
{
    df_hp = 0.5f;
    df_exp = 0.5f;
}

pos_tup get_move_vec(int dir) {
    pos_tup out_(0, 0);
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


    if (toNumber<float>(x[0]) > 0 && toNumber<float>(x[1]) > 0) {
        return 1;
    }

    if (hypot(dist_tower_x, dist_tower_y) < 1500) {
        return 1;
    }

    auto dist2creep = hypot(dist_creep_x, dist_creep_y);

    if (dist2creep < 1000) {
        if (dist2creep < 450) {
            return 1;
        }
        return 0;
    }


    return 5;
}

void Env::reset() {
    if (engine) {
        delete engine;
        engine = nullptr;
    }

    if (!config) {
        //TODO: better solution
#ifdef WIN32
        config = ConfigCacheMgr<Config>::getInstance().get("F:\\PytorchCpp\\PyTorchCpp\\SimDota2\\config\\Config.json");
#else
        config = ConfigCacheMgr<Config>::getInstance().get("/root/PytorchCppCodes/SimDota2/config/Config.json");
#endif
    }
    engine = new cppSimulatorImp(config);
    clear_state();
    update_param();
}

bool Env::step(bool debug_print, bool default_action) {
    double time_ = engine->get_time();
    //std::cout << "engine time: " << time_ << std::endl;
    if (time_ > 300) {
        return false;
    }
    torch::Tensor x = torch::ones({ 10 });

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
            x[start_idx + 1] = (loc.y - hero_loc.y) / near_by_scale;
        }

    }

    //std::cout << "Tensor input " << x << std::endl;
    states.push_back(x);

    auto out = actor_model->forward(x);
    auto value = critic_model->forward(x);
    values.push_back(value);
    

    auto action_out = out;
    auto action_log_prob = torch::log_softmax(action_out, 0);
    log_probs.push_back(action_log_prob);



    auto action_prob = torch::softmax(action_out, 0);
    float max_prob = toNumber<float>(torch::max(action_prob));

    auto action = action_prob.multinomial(1);

    int idx = toNumber<int>(action);


    if (false || max_prob > 0.9)
    {
        auto max_action = torch::argmax(action_prob, 0);
        int max_idx = toNumber<int>(max_action);
        //std::cout << max_prob << std::endl;
        idx = max_idx;
    }

    int default_idx = get_default(x);

    if (default_action)
    {
        idx = default_idx;
    }
    

    torch::Tensor expert_one_hot_action = torch::zeros({ 9 });
    expert_one_hot_action[default_idx] = 1;

    torch::Tensor expert_prob = d_model->forward(x, expert_one_hot_action);
    vec_expert_prob.push_back(expert_prob);

    expert_actions.push_back(expert_one_hot_action);

    hero->set_decision(move);
    pos_tup order = get_move_vec(idx);
    hero->set_move_order(order);
    //std::cout << "Tensor action " << idx << std::endl;
    actions.push_back(action);

    torch::Tensor one_hot_action = torch::zeros({ 9 });
    one_hot_action[idx] = 1;
    one_hot_tensor.push_back(one_hot_action);
    
    //torch::Tensor one_hot_prob = action_prob * one_hot_action;
    torch::Tensor one_hot_prob = action_prob * expert_one_hot_action;
    actor_prob_with_grad.push_back(one_hot_prob);
    torch::Tensor actor_prob = d_model->forward(x, one_hot_prob.detach());
    vec_actor_prob.push_back(actor_prob);

    if (debug_print && fisrt_loop) {
        std::cout << "first >>>" << std::endl;
        std::cout << " action: " << action_prob.reshape({ 1,-1 }) << std::endl;
        std::cout << " expert action: " << default_idx << std::endl;
        std::cout << " expert_prob: " << expert_prob << std::endl;
        std::cout << " actor_prob: " << actor_prob << std::endl;
        std::cout << "first <<" << std::endl;

        fisrt_loop = false;
    }

    last_action_prob = action_prob;
    last_expert_action = default_idx;
    last_expert_prob = expert_prob;
    last_actor_prob = actor_prob;

    ///engine running
    engine->loop();
    ///engine running

    auto hero_loc2 = hero->get_location();
    float now_exp = hero->getData().exp;
    float now_hp = hero->get_HP();

    double dis1 = hero_loc.distance(pos_tup());
    double dis2 = hero_loc2.distance(pos_tup());

    float exp_reward = now_exp > prev_exp ? 1 : 0;
    float hp_reward = now_hp < prev_hp ? -1 : 0;

    total_exp += exp_reward;
    total_hp += hp_reward;

    //std::cout << "reward " << reward << std::endl;
    exp_rewards.push_back(exp_reward);
    hp_rewards.push_back(hp_reward);

    return true;
}

torch::Tensor Env::train_discriminator(bool print_msg)
{
    torch::Tensor v_expert_prob = torch::stack(vec_expert_prob);
    torch::Tensor expert_label = torch::ones_like(v_expert_prob);
    torch::Tensor expert_d_loss = torch::binary_cross_entropy(v_expert_prob, expert_label).mean();

    torch::Tensor v_actor_prob = torch::stack(vec_actor_prob);
    torch::Tensor actor_label = torch::zeros_like(v_actor_prob);
    torch::Tensor actor_d_loss = torch::binary_cross_entropy(v_actor_prob, actor_label).mean();

    prob_diff = torch::relu(v_expert_prob - v_actor_prob).detach();

    torch::Tensor total_d_loss = expert_d_loss + actor_d_loss;
    //total_d_loss.backward(torch::nullopt, true);

    if (print_msg) {
        std::stringstream ss;

        ss << "expert D loss " << toNumber<float>(expert_d_loss)
            << " actor D loss " << toNumber<float>(actor_d_loss)
            << " total exp " << total_exp << " total hp " << total_hp;
        std::cout << ss.str() << std::endl;


        std::cout << "last >>>" << std::endl;
        std::cout << "action: " << last_action_prob.reshape({ 1,-1 }) << std::endl;
        std::cout << " expert action: " << last_expert_action << std::endl;
        std::cout << "expert_prob: " << last_expert_prob << std::endl;
        std::cout << "actor_prob: " << last_actor_prob << std::endl;
        std::cout << "last <<<" << std::endl;
    }
    return total_d_loss;
}


torch::Tensor Env::train_actor(bool print_msg) {
    torch::Tensor v_actor_prob_with_grad = torch::stack(actor_prob_with_grad);
    torch::Tensor v_state = torch::stack(states);
    
    torch::Tensor prob = d_model->forward(v_state, v_actor_prob_with_grad);
    torch::Tensor actor_label = torch::ones_like(prob);
    torch::Tensor actor_loss = (torch::relu(torch::binary_cross_entropy(prob, actor_label) - 0.1))*prob_diff.mean();
    //actor_loss.backward();

    if (print_msg) {
        std::stringstream ss;
        ss << "actor model loss " << toNumber<float>(actor_loss);
        std::cout << ss.str() << std::endl;
    }
    return actor_loss;
}

torch::Tensor Env::train_critic(bool print_msg)
{
    torch::Tensor reward_tensor = calculate_reward();
    return reward_tensor;

    torch::Tensor vec_values = torch::stack(values);
    torch::Tensor critic_loss = reward_tensor - vec_values;

    critic_loss = critic_loss * critic_loss;
    critic_loss = critic_loss.mean();
    //critic_loss.backward(torch::nullopt, true);

    torch::Tensor adv = reward_tensor - vec_values.detach();
    torch::Tensor actor_one_hot = torch::stack(one_hot_tensor);
    torch::Tensor actor_log_probs = torch::sum(torch::stack(log_probs) * actor_one_hot, {1});
    torch::Tensor actor_loss = -actor_log_probs * adv;
    actor_loss = actor_loss.mean();
    //actor_loss.backward(torch::nullopt, true);

    if (print_msg) {
        std::stringstream ss;
        ss << "critic loss " << toNumber<float>(critic_loss);
        ss << " actor loss by critic " << toNumber<float>(critic_loss);
        std::cout << ss.str() << std::endl;
    }
    return critic_loss + actor_loss;
}

torch::Tensor Env::calculate_reward()
{
    reduced_reward = std::vector<float>(exp_rewards.size(), 0.0f);
    float temp_r_exp = 0.0f;
    total_exp = 0.0f;
    float temp_r_hp = 0.0f;
    total_hp = 0.0f;

    float total_reward = 0.0;

    torch::Tensor reward_tensor = torch::zeros({ static_cast<long long>(exp_rewards.size()) });

    for (int i = exp_rewards.size() - 1; i >= 0; --i)
    {
        total_exp += exp_rewards[i];
        total_hp += hp_rewards[i];

        temp_r_exp = df_exp * temp_r_exp + exp_rewards[i];
        temp_r_hp = df_hp * temp_r_hp + hp_rewards[i];
        reduced_reward[i] = temp_r_exp + temp_r_hp;
        reward_tensor[i] = reduced_reward[i];
        total_reward += reduced_reward[i];
    }

    //torch::Tensor reward_tensor = torch::from_blob(reduced_reward.data(), c10::IntList(exp_rewards.size()));
    return reward_tensor;
}

void Env::update_param()
{
    actor_model = std::dynamic_pointer_cast<Actor>(actor_model_master->clone());
    actor_model->eval();
    critic_model = std::dynamic_pointer_cast<Critic>(critic_model_master->clone());
    critic_model->eval();
    d_model = std::dynamic_pointer_cast<DisCriminator>(d_model_master->clone());
    d_model->eval();
}

PackedData Env::prepare_data()
{
    PackedData ret;
    ret.state = torch::stack(states);
    ret.expert_action = torch::stack(expert_actions);
    ret.actor_one_hot = torch::stack(one_hot_tensor);
    ret.reward = calculate_reward();
    return ret;
}

void Env::evaluate() {
    reset();
    std::cout << ">>>>>>>>>>>>>>>>>>>>EVALUATE>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl << std::endl;
    while (step(true, false));
    std::cout << "last >>>" << std::endl;
    std::cout << "action: " << last_action_prob.reshape({ 1,-1 }) << std::endl;
    std::cout << " expert action: " << last_expert_action << std::endl;
    std::cout << "expert_prob: " << last_expert_prob << std::endl;
    std::cout << "actor_prob: " << last_actor_prob << std::endl;
    std::cout << "last <<<" << std::endl;
    std::cout << "evaluate total exp " << total_exp << " total hp " << total_hp << std::endl;
    std::cout << "<<<<<<<<<<<<<<<<<<<<EVALUATE<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl << std::endl;
}

std::string Env::record() {
    reset();
    engine->set_trace(true);
    while (step(false, false));
    return engine->dump_trace();
}

void Env::print_summary()
{

}
