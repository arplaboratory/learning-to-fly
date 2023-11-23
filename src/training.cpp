
#include "training.h"

#include <chrono>


template <typename T_ABLATION_SPEC>
void run(){
    using namespace multirotor_training::config;

    using CONFIG = multirotor_training::config::Config<T_ABLATION_SPEC>;
    using TI = typename CONFIG::TI;

#ifdef LEARNING_TO_FLY_IN_SECONDS_BENCHMARK
    constexpr TI NUM_RUNS = 1;
#else
    constexpr TI NUM_RUNS = 1;
#endif

    for (TI run_i = 0; run_i < NUM_RUNS; run_i++){
        std::cout << "Run " << run_i << "\n";
        auto start = std::chrono::high_resolution_clock::now();
        multirotor_training::operations::TrainingState<CONFIG> ts;
        multirotor_training::operations::init(ts, run_i);

        for(TI step_i=0; step_i < CONFIG::STEP_LIMIT; step_i++){
            multirotor_training::operations::step(ts);
        }

        multirotor_training::operations::destroy(ts);
        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "Training took: " << std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count() << "s" << std::endl;
    }
}


int main(){
    run<multirotor_training::config::DEFAULT_ABLATION_SPEC>();
    return 0;
}