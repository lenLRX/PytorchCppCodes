#pragma once

#include <torch/torch.h>
#include <SimDota2/include/SimDota2.h>

const static float lr = 1e-4;

static const float near_by_scale = 2000;
static const int state_dim = 10;

template<typename T>
T toNumber(torch::Tensor x) {
    return x.item().to<T>();
}

template <typename M>
void save_model(M& model, const std::string& prefix, int i) {
    torch::serialize::OutputArchive model_save;
    model->save(model_save);
    model_save.save_to(prefix + "_"  + std::to_string(i) + ".model");
}


template <typename M>
void load_model(M& model, const std::string& prefix, int i) {
    torch::serialize::InputArchive infile;
    infile.load_from(prefix + "_" + std::to_string(i) + ".model");
    model->load(infile);
}

static pos_tup get_move_vec(int dir) {
    pos_tup out_(0, 0);
    if (dir == 0) {
        return out_;
    }
    double rad = dir * M_PI / 4 - M_PI;
    out_.x = cos(rad) * 300;
    out_.y = sin(rad) * 300;
    return out_;
}

static int get_move_dir(float x, float y) {
    if (fabs(x) < 0.1) {
        if (fabs(y) < 0.1) {
            return 0;
        }
        if (y < 0) {
            return 2;
        }

        if (y > 0) {
            return 6;
        }
    }
    if (x < 0) {
        if (fabs(y) < 0.1) {
            return 8;
        }
        if (y < 0) {
            return 1;
        }

        if (y > 0) {
            return 7;
        }
    }

    if (x > 0) {
        if (fabs(y) < 0.1) {
            return 4;
        }
        if (y < 0) {
            return 3;
        }

        if (y > 0) {
            return 5;
        }
    }
}

static torch::Tensor state_encoding(cppSimulatorImp* engine) {
    auto hero = engine->getHero("Radiant", 0);

    auto hero_loc = hero->get_location();

    torch::Tensor x = torch::ones({ state_dim });

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
    return x;
}

