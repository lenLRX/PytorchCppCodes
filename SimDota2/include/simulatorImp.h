#ifndef __SIMULATORIMP_H__
#define __SIMULATORIMP_H__

#include "Config.h"
#include "simulator.h"
#include "Event.h"
#include "util.h"

#include <queue>
#include <list>
#include <vector>

//forward decl
class Hero;
class Sprite;
class Config;

typedef std::vector<std::pair<Sprite*, double>> VecSpriteDist;

class SIMDOTA2_API cppSimulatorImp
{
public:
    cppSimulatorImp() = delete;
    cppSimulatorImp(Config* cfg);
    ~cppSimulatorImp();
    inline Config* get_config() { return cfg; }
    double get_time();
    inline void tick_tick() { tick_time += delta_tick; }
    inline void addSprite(Sprite* s) { Sprites.push_back(s); allSprites.push_back(s); }
    inline const std::list<Sprite*>& get_sprites() { return Sprites; }
    inline double get_deltatick() const { return delta_tick; }
    inline std::priority_queue<Event>& get_queue() { return queue; }
    void loop();
    VecSpriteDist get_nearby_enemy(Sprite* s, double dist);
    VecSpriteDist get_nearby_enemy(Sprite * sprite, double dist, std::function<bool(Sprite*)> filter);
    VecSpriteDist get_nearby_ally(Sprite* s, double dist);
    VecSpriteDist get_nearby_ally(Sprite * sprite, double dist, std::function<bool(Sprite*)> filter);
    VecSpriteDist get_nearby_sprite(pos_tup loc,double dist);
    Hero* getHero(const std::string& side, int idx);
private:
    Config* cfg;
    double tick_time;
    double tick_per_second;
    double delta_tick;
    std::vector<Hero*> RadiantHeros;
    std::vector<Hero*> DireHeros;
    std::list<Sprite*> Sprites;
    std::list<Sprite*> allSprites;
    std::priority_queue<Event> queue;
};

#endif//__SIMULATORIMP_H__