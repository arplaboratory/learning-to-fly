#include "parameters.h"
#include "actor_and_critic.h"
#include "validation.h"
#include <rl_tools/rl/algorithms/td3/td3.h>
#include <rl_tools/rl/components/off_policy_runner/off_policy_runner.h>

#include "ablation.h"

namespace learning_to_fly{
    namespace config {
        template <typename T_ABLATION_SPEC>
        struct Base{
#ifdef LEARNING_TO_FLY_IN_SECONDS_BENCHMARK
            static constexpr bool BENCHMARK = true;
#else
            static constexpr bool BENCHMARK = false;
#endif
            using ABLATION_SPEC = T_ABLATION_SPEC;
            using LOGGER = rlt::LOGGER_FACTORY<>;
            using DEV_SPEC = rlt::devices::cpu::Specification<rlt::devices::math::CPU, rlt::devices::random::CPU, LOGGER>;
            using DEVICE = rlt::DEVICE_FACTORY<DEV_SPEC>;
            using T = float;
            using TI = typename DEVICE::index_t;

            using ENVIRONMENT = typename parameters::environment<T, TI, ABLATION_SPEC>::ENVIRONMENT;
            using ABLATION_SPEC_EVAL_INSTANCE = ABLATION_SPEC_EVAL<ABLATION_SPEC>;
            static_assert(ABLATION_SPEC_EVAL_INSTANCE::ROTOR_DELAY == true);
            using ENVIRONMENT_EVALUATION = typename parameters::environment<T, TI, ABLATION_SPEC_EVAL_INSTANCE>::ENVIRONMENT;
            static_assert(ENVIRONMENT::OBSERVATION_DIM == ENVIRONMENT_EVALUATION::OBSERVATION_DIM);
            static_assert(ENVIRONMENT::ACTION_DIM == ENVIRONMENT_EVALUATION::ACTION_DIM);
            using UI = bool;

            struct DEVICE_SPEC: rlt::devices::DefaultCPUSpecification {
                using LOGGING = rlt::devices::logging::CPU;
            };
            struct TD3PendulumParameters: rlt::rl::algorithms::td3::DefaultParameters<T, TI>{
                static constexpr TI ACTOR_BATCH_SIZE = 256;
                static constexpr TI CRITIC_BATCH_SIZE = 256;
                static constexpr TI TRAINING_INTERVAL = 10;
                static constexpr TI CRITIC_TRAINING_INTERVAL = 1 * TRAINING_INTERVAL;
                static constexpr TI ACTOR_TRAINING_INTERVAL = 2 * TRAINING_INTERVAL;
                static constexpr TI CRITIC_TARGET_UPDATE_INTERVAL = 1 * TRAINING_INTERVAL;
                static constexpr TI ACTOR_TARGET_UPDATE_INTERVAL = 2 * TRAINING_INTERVAL;
                static constexpr T TARGET_NEXT_ACTION_NOISE_CLIP = 0.5;
                static constexpr T TARGET_NEXT_ACTION_NOISE_STD = 0.2;
                static constexpr T GAMMA = 0.99;
                static constexpr bool IGNORE_TERMINATION = false;
            };

            using TD3_PARAMETERS = TD3PendulumParameters;

            using ACTOR_CRITIC_CONFIG = ActorAndCritic<T, TI, ENVIRONMENT, TD3_PARAMETERS>;
            static constexpr bool ASYMMETRIC_OBSERVATIONS = ACTOR_CRITIC_CONFIG::ASYMMETRIC_OBSERVATIONS;
            using ACTOR_TYPE = typename ACTOR_CRITIC_CONFIG::ACTOR_TYPE;
            using ACTOR_TARGET_TYPE = typename ACTOR_CRITIC_CONFIG::ACTOR_TARGET_TYPE;
            using ACTOR_CHECKPOINT_TYPE = typename ACTOR_CRITIC_CONFIG::ACTOR_CHECKPOINT_TYPE;
            using CRITIC_TYPE = typename ACTOR_CRITIC_CONFIG::CRITIC_TYPE;
            using CRITIC_TARGET_TYPE = typename ACTOR_CRITIC_CONFIG::CRITIC_TARGET_TYPE;
            using OPTIMIZER = typename ACTOR_CRITIC_CONFIG::OPTIMIZER;


            using ACTOR_CRITIC_SPEC = rlt::rl::algorithms::td3::Specification<T, TI, ENVIRONMENT, ACTOR_TYPE, ACTOR_TARGET_TYPE, CRITIC_TYPE, CRITIC_TARGET_TYPE, OPTIMIZER, TD3_PARAMETERS>;
            using ACTOR_CRITIC_TYPE = rlt::rl::algorithms::td3::ActorCritic<ACTOR_CRITIC_SPEC>;


            static constexpr bool ACTOR_ENABLE_CHECKPOINTS = !BENCHMARK;
            static constexpr TI ACTOR_CHECKPOINT_INTERVAL = 100000;
            static constexpr bool DETERMINISTIC_EVALUATION = !BENCHMARK;
            static constexpr TI EVALUATION_INTERVAL = 10000;
            static constexpr TI NUM_EVALUATION_EPISODES = 1000;
            static constexpr bool COLLECT_EPISODE_STATS = false;
            static constexpr TI EPISODE_STATS_BUFFER_SIZE = 1000;
            static constexpr TI N_ENVIRONMENTS = 1;
            static constexpr TI STEP_LIMIT = 300001;
//            static constexpr TI REPLAY_BUFFER_LIMIT = 3000000;
            static constexpr TI REPLAY_BUFFER_CAP = STEP_LIMIT;
            static constexpr TI ENVIRONMENT_STEP_LIMIT = 500;
            static constexpr TI ENVIRONMENT_STEP_LIMIT_EVALUATION = 500;
            static constexpr TI BASE_SEED = 0;
            static constexpr bool CONSTRUCT_LOGGER = false;
            using OFF_POLICY_RUNNER_SPEC = rlt::rl::components::off_policy_runner::Specification<T, TI, ENVIRONMENT, N_ENVIRONMENTS, ASYMMETRIC_OBSERVATIONS, REPLAY_BUFFER_CAP, ENVIRONMENT_STEP_LIMIT, rlt::rl::components::off_policy_runner::DefaultParameters<T>, false, true, 1000>;
            using OFF_POLICY_RUNNER_TYPE = rlt::rl::components::OffPolicyRunner<OFF_POLICY_RUNNER_SPEC>;
            static constexpr rlt::rl::components::off_policy_runner::DefaultParameters<T> off_policy_runner_parameters = {
                    0.1
            };

            static constexpr TI N_WARMUP_STEPS_CRITIC = 15000;
            static constexpr TI N_WARMUP_STEPS_ACTOR = 30000;
            static_assert(ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE == ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE);
        };
        template <typename T_ABLATION_SPEC>
        using Config = Validation<Base<T_ABLATION_SPEC>>;
    }
}
