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

void LastHitMode::step(cppSimulatorImp * engine, bool default_action)
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
