#include "Hero.h"
#include "Creep.h"
#include "log.h"
#include "simulatorImp.h"
#include <cmath>
#include <random>

using namespace rapidjson;

class HeroDataExtFn;

class HeroDataExt
{
private:
    friend void HeroDataExtFn(SpriteData* s, std::string json_path);
    HeroDataExt(std::string json_path)
    {
        std::ifstream ifs(json_path);
        Document document;
        IStreamWrapper isw(ifs);
        document.ParseStream(isw);
        pos_tup tmp_tup;
        double temp_x;
        double temp_y;
        temp_x = document["Radiant"]["init_loc"]["x"].GetDouble();
        temp_y = document["Radiant"]["init_loc"]["y"].GetDouble();
        init_loc[(int)Side::Radiant] = pos_tup(temp_x, temp_y);

        temp_x = document["Dire"]["init_loc"]["x"].GetDouble();
        temp_y = document["Dire"]["init_loc"]["y"].GetDouble();
        init_loc[(int)Side::Dire] = pos_tup(temp_x, temp_y);
    }
public:
    pos_tup init_loc[2];
};

void HeroDataExtFn(SpriteData* s, std::string json_path)
{
    s->ext = new HeroDataExt(json_path);
};


static std::default_random_engine rnd_gen;
static std::uniform_int_distribution<int> pos_distribution(1, 1000);
static std::uniform_int_distribution<int> sign_distribution(-1, 1);

static int get_rand()
{
    return sign_distribution(rnd_gen) * pos_distribution(rnd_gen);
}

DEF_INIT_DATA_FN(Hero)

Hero::Hero(cppSimulatorImp* _Engine, Side _side, std::string type_name)
    :Sprite(INIT_DATA(_Engine, Hero, type_name, HeroDataExtFn)), target(nullptr), decision(noop)
{
    Engine = _Engine;
    unit_type = UNITTYPE_HERO;
    side = _side;

    last_exp = 0.0;
    last_HP = data.HP;

    HeroDataExt* p_ext = (HeroDataExt*)data.ext;
    init_loc = p_ext->init_loc[(int)side];
    if (side == Side::Radiant) {
        color = GET_CFG->Radiant_Colors;
    }
    else {
        color = GET_CFG->Dire_Colors;
    }

    location = init_loc;
    move_order = pos_tup(0,0);
}

Hero::~Hero()
{
    //LOG << "gold:" << data.gold << endl;
    Logger::getInstance().flush();
}

int Hero::get_max_health()
{
    //return 200 + 20*(base_strength + strength_gain*level);
    return 500;
}

int Hero::get_max_mana()
{
    //return 75 + 12*(base_intelligence + intelligence_gain*level);
    return 291;
}

void Hero::step()
{
    if (isAttacking())
        return;
    if (decisonType::noop == decision) {
        ;
    }
    else if (decisonType::move == decision) {
        auto p = pos_tup(move_order.x + location.x,
            move_order.y + location.y);
        set_move(p);
    }
    else if (decisonType::attack == decision) {
        if (nullptr == target) {
            LOG << "null target!\n";
            Logger::getInstance().flush();
            throw "null target!";
        }
        attack(target);
    }
    
}

void Hero::draw()
{
}

void Hero::set_move_order(const pos_tup& order)
{
    move_order = order;
}

void Hero::set_target(Sprite* s)
{
    target = s;
}

const target_list_t& Hero::getTargetList()
{
    return target_list;
}

void Hero::set_decision(int d)
{
    decision = d;
}
