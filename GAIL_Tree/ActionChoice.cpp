#include "ActionChoice.h"
#include "MovingMode.h"
#include "LastHitMode.h"

#include "util.h"

ActionChoice::ActionChoice(int input_dim, int hidden_dim, int output_dim) :
    ModelNode(input_dim, hidden_dim, output_dim) {
    children.push_back(
        std::dynamic_pointer_cast<ModelNode>(std::make_shared<MovingMode>(input_dim, hidden_dim)));
    children.push_back(
        std::dynamic_pointer_cast<ModelNode>(std::make_shared<LastHitMode>(input_dim + 2, hidden_dim)));
}

void ActionChoice::step(cppSimulatorImp* engine, bool default_action) {
    auto hero = engine->getHero("Radiant", 0);

    auto hero_loc = hero->get_location();

    torch::Tensor x = state_encoding(engine);

    states.push_back(x);

    auto out = actor_->forward(x);

    auto action_out = out;

    auto action_prob = torch::softmax(action_out, 0);
    float max_prob = toNumber<float>(torch::max(action_prob));

    auto action = action_prob.multinomial(1);

    int idx = toNumber<int>(action);

    int default_idx = get_default(x);

    if (default_action)
    {
        idx = default_idx;
    }

    torch::Tensor expert_one_hot_action = torch::zeros({ output_dim_ });
    expert_one_hot_action[default_idx] = 1;

    expert_actions.push_back(expert_one_hot_action);

    torch::Tensor one_hot_action = torch::zeros({ output_dim_ });
    one_hot_action[idx] = 1;
    actor_actions.push_back(one_hot_action);

    hero->set_decision(idx + 1);

    children[idx]->step(engine, default_action);
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

void ActionChoice::train(const std::vector<PackedData>& data) {

}