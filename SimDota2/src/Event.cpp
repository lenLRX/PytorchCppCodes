#include "simulatorImp.h"
#include "Event.h"
#include "Sprite.h"

#include "Creep.h"

void EventFactory::CreateAttackEvnt(Sprite* attacker, Sprite* victim)
{
    if (attacker == nullptr || victim == nullptr) {
        return;
    }
    cppSimulatorImp* engine = attacker->get_engine();
    auto fn = [=]() {
        double actualDmg = victim->attakedDmg(attacker, attacker->get_Attack());
        victim->damaged(attacker, actualDmg);
    };
    engine->get_queue().push(
        Event(engine->get_time() + attacker->TimeToDamage(victim), fn));
}

static void spawn_fn(cppSimulatorImp* Engine) {
    for (int i = 0; i < 5; ++i) {
        Engine->addSprite(new Creep(Engine, Side::Radiant, "MeleeCreep"));
        Engine->addSprite(new Creep(Engine, Side::Dire, "MeleeCreep"));
    }

    Engine->addSprite(new Creep(Engine, Side::Radiant, "RangedCreep"));
    Engine->addSprite(new Creep(Engine, Side::Dire, "RangedCreep"));
    
    std::function<void()> fn = std::bind(spawn_fn, Engine);
    Engine->get_queue().push(Event(Engine->get_time() + 30, fn));
}

void EventFactory::CreateSpawnEvnt(cppSimulatorImp* Engine)
{
    std::function<void()> fn = std::bind(spawn_fn, Engine);
    Engine->get_queue().push(Event(Engine->get_time() + 30,fn));
}
