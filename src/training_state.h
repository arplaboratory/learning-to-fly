#include <queue>

namespace learning_to_fly{
    template <typename T_CONFIG>
    struct TrainingState: bpt::rl::algorithms::td3::loop::TrainingState<T_CONFIG>{
        using CONFIG = T_CONFIG;
        std::string run_name;
        std::queue<std::vector<typename CONFIG::ENVIRONMENT::State>> trajectories;
        std::mutex trajectories_mutex;
        std::vector<typename CONFIG::ENVIRONMENT::State> episode;
        // validation
        bpt::rl::utils::validation::Task<typename CONFIG::TASK_SPEC> task;
        typename CONFIG::ENVIRONMENT validation_envs[CONFIG::VALIDATION_N_EPISODES];
        typename CONFIG::ACTOR_TYPE::template DoubleBuffer<CONFIG::VALIDATION_N_EPISODES> validation_actor_buffers;
    };
}
