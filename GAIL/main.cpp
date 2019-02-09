#include "Env.h"
#include "util.h"

class my_barrier
{

public:
    my_barrier(int count)
        : thread_count(count)
        , counter(0)
        , waiting(0)
    {}

    void wait()
    {
        //fence mechanism
        std::unique_lock<std::mutex> lk(m);
        ++counter;
        ++waiting;
        if (counter == thread_count) {
            lk.unlock();
            cv.notify_all();
        }
        else {
            cv.wait(lk);
        }
        --waiting;
        if (waiting == 0)
        {
            //reset barrier
            counter = 0;
        }
    }

private:
    std::mutex m;
    std::condition_variable cv;
    int counter;
    int waiting;
    int thread_count;
};

const int thread_num = 1;

my_barrier barrier1(thread_num + 1);
my_barrier barrier2(thread_num + 1);

std::condition_variable cv1;
std::condition_variable cv2;

std::mutex m1;
std::mutex m2;

auto thread_fn = [](Env* env, bool first) {
    while (true) {
        barrier2.wait();
        env->reset();
        while (env->step(first, false));
        barrier1.wait();
    }
};

int main(int argc, char** argv) {
    auto shared_actor_net = std::make_shared<Actor>(10, 9, 16);
    auto shared_critic_net = std::make_shared<Critic>(10, 9, 16);
    auto shared_d_net = std::make_shared<DisCriminator>(10, 9, 16);

    float lr = 1E-3;

    torch::optim::SGD actor_optim(shared_actor_net->parameters(),
        torch::optim::SGDOptions(lr));

    torch::optim::SGD critic_optim(shared_critic_net->parameters(),
        torch::optim::SGDOptions(lr));

    torch::optim::SGD d_optim(shared_d_net->parameters(),
        torch::optim::SGDOptions(lr));


    std::vector<Env*> vec_env;

    std::vector<std::thread> vec_threads;

    bool first = true;
    for (int i = 0; i < thread_num; ++i) {
        Env* penv = new Env(shared_actor_net, shared_critic_net, shared_d_net);
        vec_env.push_back(penv);
        first = false;
    }

    Env* p_test = new Env(shared_actor_net, shared_critic_net, shared_d_net);

    bool expert = true;

    int count = 0;

    while (true) {
        count++;
        first = true;

        for (auto env : vec_env) {
            env->reset();
            while (env->step(first, true));
            first = false;
        }

        first = true;
        d_optim.zero_grad();
        for (auto env : vec_env) {
            env->train_discriminator(first);
            first = false;
        }
        d_optim.step();

        first = true;
        actor_optim.zero_grad();
        critic_optim.zero_grad();
        for (auto env : vec_env) {
            env->train_critic(first);
            first = false;
        }

        critic_optim.step();
        first = true;
        for (auto env : vec_env) {
            env->train_actor(first);
            first = false;
        }
        
        actor_optim.step();

        if (count % 100 == 0) {
            p_test->evaluate();
            save_model(shared_actor_net, "actor", count);
            save_model(shared_critic_net, "discriminator", count);
            save_model(shared_d_net, "critic", count);
        }
        
        /*
        for (const auto& p : d_optim.parameters())
        {
            std::cout << p.grad() << std::endl;
        }
        */
    }

    return 0;
}