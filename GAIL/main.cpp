#include "Env.h"
#include "util.h"
#include "ATen/core/TensorImpl.h"

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

const int thread_num = 2;

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
        torch::Tensor d_loss = env->train_discriminator(first);
        torch::Tensor critic_loss = env->train_critic(first);
        torch::Tensor actor_loss = env->train_actor(first);
        env->overall_loss = d_loss + critic_loss + actor_loss;
        std::stringstream ss;
        ss << std::this_thread::get_id() << " cpu_id :" << c10::CPUTensorId()
            << " tensor type " << env->overall_loss.type_id() << "|";
        std::cerr << ss.str() << std::endl;
        barrier1.wait();
    }
};

int main(int argc, char** argv) {
    auto shared_actor_net = std::make_shared<Actor>(10, 9, 16);
    auto shared_critic_net = std::make_shared<Critic>(10, 9, 16);
    auto shared_d_net = std::make_shared<DisCriminator>(10, 9, 16);

    int count = 0;
    if (argc >= 2) {
        count = std::atoi(argv[1]);
        load_model(shared_actor_net, "actor", count);
        load_model(shared_critic_net, "critic", count);
        load_model(shared_d_net, "discriminator", count);
    }

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
        vec_threads.emplace_back(thread_fn, penv, first);
        first = false;
    }

    Env* p_test = new Env(shared_actor_net, shared_critic_net, shared_d_net);

    if (argc == 3 && std::string(argv[2]) == "test") {
        std::cout << p_test->record() << std::endl;
        return 0;
    }

    torch::Tensor stub = torch::zeros({ 1 });

    bool expert = true;

    while (true) {
        count++;
        std::cerr << "main thread cpu_id :" << c10::CPUTensorId() << std::endl;
        std::cerr << stub.type_id() << std::endl;
        barrier2.wait();
        barrier1.wait();

        d_optim.zero_grad();
        actor_optim.zero_grad();
        critic_optim.zero_grad();

        std::vector<torch::Tensor> vec_tensor;

        for (auto env : vec_env) {
            vec_tensor.push_back(env->overall_loss);
        }

        torch::Tensor total_loss = torch::stack(vec_tensor);
        
        total_loss.sum().backward();

        d_optim.step();
        critic_optim.step();
        actor_optim.step();

        if (count % 100 == 0) {
            p_test->evaluate();
            save_model(shared_actor_net, "actor", count);
            save_model(shared_critic_net, "critic", count);
            save_model(shared_d_net, "discriminator", count);
        }
        
    }

    return 0;
}
