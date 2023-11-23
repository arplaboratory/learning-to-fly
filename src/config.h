
#include "parameters.h"

namespace learning_to_fly{
    namespace config {
        using namespace rlt::nn_models::sequential::interface; // to simplify the model definition we import the sequential interface but we don't want to pollute the global namespace hence we do it in a model definition namespace
        struct DEFAULT_ABLATION_SPEC{
            static constexpr bool DISTURBANCE = true;
            static constexpr bool OBSERVATION_NOISE = true;
            static constexpr bool ASYMMETRIC_ACTOR_CRITIC = true;
            static constexpr bool ROTOR_DELAY = true;
            static constexpr bool ACTION_HISTORY = true;
            static constexpr bool ENABLE_CURRICULUM = true;
            static constexpr bool RECALCULATE_REWARDS = true;
            static constexpr bool USE_INITIAL_REWARD_FUNCTION = true;
            static constexpr bool INIT_NORMAL = true;
            static constexpr bool EXPLORATION_NOISE_DECAY = true;
        };
        template <typename T_ABLATION_SPEC>
        struct ABLATION_SPEC_EVAL: T_ABLATION_SPEC{
            // override everything but ACTION_HISTORY because that changes the observation space
            static constexpr bool DISTURBANCE = true;
            static constexpr bool OBSERVATION_NOISE = true;
            static constexpr bool ASYMMETRIC_ACTOR_CRITIC = true;
            static constexpr bool ROTOR_DELAY = true;
            static constexpr bool ENABLE_CURRICULUM = true;
            static constexpr bool RECALCULATE_REWARDS = true;
            static constexpr bool USE_INITIAL_REWARD_FUNCTION = false; // Use target reward function as metric
            static constexpr bool INIT_NORMAL = true;
            static constexpr bool EXPLORATION_NOISE_DECAY = true;
        };
        template <typename T_ABLATION_SPEC>
        struct CoreConfig{
#ifdef LEARNING_TO_FLY_IN_SECONDS_BENCHMARK
            static constexpr bool BENCHMARK = true;
#else
            static constexpr bool BENCHMARK = false;
#endif
            using ABLATION_SPEC = T_ABLATION_SPEC;
#ifdef RL_TOOLS_ENABLE_TENSORBOARD
            using LOGGER = rlt::utils::typing::conditional_t<BENCHMARK, rlt::devices::logging::CPU , rlt::devices::logging::CPU_TENSORBOARD<rlt::devices::logging::CPU_TENSORBOARD_FREQUENCY_EXTENSION>>;
#else
            using LOGGER = rlt::devices::logging::CPU;
#endif
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
//                constexpr static typename TI CRITIC_BATCH_SIZE = 100;
//                constexpr static typename TI ACTOR_BATCH_SIZE = 100;
//                constexpr static T GAMMA = 0.997;
                static constexpr TI ACTOR_BATCH_SIZE = 256;
                static constexpr TI CRITIC_BATCH_SIZE = 256;
                static constexpr TI TRAINING_INTERVAL = 10;
                static constexpr TI CRITIC_TRAINING_INTERVAL = 1 * TRAINING_INTERVAL;
                static constexpr TI ACTOR_TRAINING_INTERVAL = 2 * TRAINING_INTERVAL;
                static constexpr TI CRITIC_TARGET_UPDATE_INTERVAL = 1 * TRAINING_INTERVAL;
                static constexpr TI ACTOR_TARGET_UPDATE_INTERVAL = 2 * TRAINING_INTERVAL;
//            static constexpr T TARGET_NEXT_ACTION_NOISE_CLIP = 1.0;
//            static constexpr T TARGET_NEXT_ACTION_NOISE_STD = 0.5;
                static constexpr T TARGET_NEXT_ACTION_NOISE_CLIP = 0.5;
                static constexpr T TARGET_NEXT_ACTION_NOISE_STD = 0.2;
                static constexpr T GAMMA = 0.99;
                static constexpr bool IGNORE_TERMINATION = false;
            };

            using TD3_PARAMETERS = TD3PendulumParameters;

            static constexpr bool ASYMMETRIC_OBSERVATIONS = ENVIRONMENT::PRIVILEGED_OBSERVATION_AVAILABLE;
            static constexpr TI CRITIC_OBSERVATION_DIM = ASYMMETRIC_OBSERVATIONS ? ENVIRONMENT::OBSERVATION_DIM_PRIVILEGED : ENVIRONMENT::OBSERVATION_DIM;

            template <typename PARAMETER_TYPE>
            struct ACTOR{
                static constexpr TI HIDDEN_DIM = 64;
                static constexpr TI BATCH_SIZE = TD3_PARAMETERS::ACTOR_BATCH_SIZE;
                static constexpr auto ACTIVATION_FUNCTION = rlt::nn::activation_functions::FAST_TANH;
                using LAYER_1_SPEC = rlt::nn::layers::dense::Specification<T, TI, ENVIRONMENT::OBSERVATION_DIM, HIDDEN_DIM, ACTIVATION_FUNCTION, PARAMETER_TYPE, BATCH_SIZE, rlt::nn::parameters::groups::Input>;
                using LAYER_1 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
                using LAYER_2_SPEC = rlt::nn::layers::dense::Specification<T, TI, HIDDEN_DIM, HIDDEN_DIM, ACTIVATION_FUNCTION, PARAMETER_TYPE, BATCH_SIZE, rlt::nn::parameters::groups::Normal>;
                using LAYER_2 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;
                using LAYER_3_SPEC = rlt::nn::layers::dense::Specification<T, TI, HIDDEN_DIM, ENVIRONMENT::ACTION_DIM, rlt::nn::activation_functions::FAST_TANH, PARAMETER_TYPE, BATCH_SIZE, rlt::nn::parameters::groups::Output>;
                using LAYER_3 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_3_SPEC>;

                using MODEL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3>>>;
            };

            template <typename ACTOR>
            struct ACTOR_CHECKPOINT{
                using LAYER_1_SPEC = rlt::nn::layers::dense::Specification<T, TI, ENVIRONMENT::OBSERVATION_DIM, ACTOR::HIDDEN_DIM, ACTOR::ACTIVATION_FUNCTION, rlt::nn::parameters::Plain, 1, rlt::nn::parameters::groups::Input>;
                using LAYER_1 = rlt::nn::layers::dense::Layer<LAYER_1_SPEC>;
                using LAYER_2_SPEC = rlt::nn::layers::dense::Specification<T, TI, ACTOR::HIDDEN_DIM, ACTOR::HIDDEN_DIM, ACTOR::ACTIVATION_FUNCTION, rlt::nn::parameters::Plain, 1, rlt::nn::parameters::groups::Normal>;
                using LAYER_2 = rlt::nn::layers::dense::Layer<LAYER_2_SPEC>;
                using LAYER_3_SPEC = rlt::nn::layers::dense::Specification<T, TI, ACTOR::HIDDEN_DIM, ENVIRONMENT::ACTION_DIM, rlt::nn::activation_functions::FAST_TANH, rlt::nn::parameters::Plain, 1, rlt::nn::parameters::groups::Output>;
                using LAYER_3 = rlt::nn::layers::dense::Layer<LAYER_3_SPEC>;

                using MODEL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3>>>;
            };

            template <typename PARAMETER_TYPE>
            struct CRITIC{
                static constexpr TI HIDDEN_DIM = 64;
                static constexpr TI BATCH_SIZE = TD3_PARAMETERS::CRITIC_BATCH_SIZE;

                static constexpr auto ACTIVATION_FUNCTION = rlt::nn::activation_functions::FAST_TANH;
                using LAYER_1_SPEC = rlt::nn::layers::dense::Specification<T, TI, CRITIC_OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM, HIDDEN_DIM, ACTIVATION_FUNCTION, PARAMETER_TYPE, BATCH_SIZE, rlt::nn::parameters::groups::Input>;
                using LAYER_1 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
                using LAYER_2_SPEC = rlt::nn::layers::dense::Specification<T, TI, HIDDEN_DIM, HIDDEN_DIM, ACTIVATION_FUNCTION, PARAMETER_TYPE, BATCH_SIZE, rlt::nn::parameters::groups::Normal>;
                using LAYER_2 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;
                using LAYER_3_SPEC = rlt::nn::layers::dense::Specification<T, TI, HIDDEN_DIM, 1, rlt::nn::activation_functions::ActivationFunction::IDENTITY, PARAMETER_TYPE, BATCH_SIZE, rlt::nn::parameters::groups::Output>;
                using LAYER_3 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_3_SPEC>;

                using MODEL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3>>>;
            };

            struct OPTIMIZER_PARAMETERS: rlt::nn::optimizers::adam::DefaultParameters<T, TI>{
                static constexpr T WEIGHT_DECAY = 0.0001;
                static constexpr T WEIGHT_DECAY_INPUT = 0.0001;
                static constexpr T WEIGHT_DECAY_OUTPUT = 0.0001;
                static constexpr T BIAS_LR_FACTOR = 1;
            };
            using OPTIMIZER = rlt::nn::optimizers::Adam<OPTIMIZER_PARAMETERS>;
            using ACTOR_TYPE = typename ACTOR<rlt::nn::parameters::Adam>::MODEL;
            using ACTOR_CHECKPOINT_TYPE = typename ACTOR_CHECKPOINT<ACTOR<rlt::nn::parameters::Plain>>::MODEL;
            using ACTOR_TARGET_TYPE = typename ACTOR<rlt::nn::parameters::Plain>::MODEL;
            using CRITIC_TYPE = typename CRITIC<rlt::nn::parameters::Adam>::MODEL;
            using CRITIC_TARGET_TYPE = typename CRITIC<rlt::nn::parameters::Plain>::MODEL;

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
        struct Config: CoreConfig<T_ABLATION_SPEC>{
            using SUPER = CoreConfig<T_ABLATION_SPEC>;
            using T = typename SUPER::T;
            using TI = typename SUPER::TI;
            using ENVIRONMENT = typename SUPER::ENVIRONMENT;
            using VALIDATION_SPEC = rlt::rl::utils::validation::Specification<T, TI, ENVIRONMENT>;
            static constexpr TI VALIDATION_N_EPISODES = 10;
            static constexpr TI VALIDATION_MAX_EPISODE_LENGTH = SUPER::ENVIRONMENT_STEP_LIMIT;
            using TASK_SPEC = rlt::rl::utils::validation::TaskSpecification<VALIDATION_SPEC, VALIDATION_N_EPISODES, VALIDATION_MAX_EPISODE_LENGTH>;
            using ADDITIONAL_METRICS = rlt::rl::utils::validation::set::Component<
            rlt::rl::utils::validation::metrics::SettlingFractionPosition<TI, 200>,
            rlt::rl::utils::validation::set::Component<rlt::rl::utils::validation::metrics::MaxErrorMean<rlt::rl::utils::validation::metrics::multirotor::POSITION, TI, 100>,
            rlt::rl::utils::validation::set::Component<rlt::rl::utils::validation::metrics::MaxErrorStd <rlt::rl::utils::validation::metrics::multirotor::POSITION, TI, 100>,
            rlt::rl::utils::validation::set::Component<rlt::rl::utils::validation::metrics::MaxErrorMean<rlt::rl::utils::validation::metrics::multirotor::POSITION, TI, 200>,
            rlt::rl::utils::validation::set::Component<rlt::rl::utils::validation::metrics::MaxErrorStd <rlt::rl::utils::validation::metrics::multirotor::POSITION, TI, 200>,
            rlt::rl::utils::validation::set::Component<rlt::rl::utils::validation::metrics::MaxErrorMean<rlt::rl::utils::validation::metrics::multirotor::ANGLE, TI, 100>,
            rlt::rl::utils::validation::set::Component<rlt::rl::utils::validation::metrics::MaxErrorStd <rlt::rl::utils::validation::metrics::multirotor::ANGLE, TI, 200>,
            rlt::rl::utils::validation::set::Component<rlt::rl::utils::validation::metrics::MaxErrorMean<rlt::rl::utils::validation::metrics::multirotor::LINEAR_VELOCITY, TI, 100>,
            rlt::rl::utils::validation::set::Component<rlt::rl::utils::validation::metrics::MaxErrorStd <rlt::rl::utils::validation::metrics::multirotor::LINEAR_VELOCITY, TI, 100>,
            rlt::rl::utils::validation::set::Component<rlt::rl::utils::validation::metrics::MaxErrorMean<rlt::rl::utils::validation::metrics::multirotor::LINEAR_VELOCITY, TI, 200>,
            rlt::rl::utils::validation::set::Component<rlt::rl::utils::validation::metrics::MaxErrorStd <rlt::rl::utils::validation::metrics::multirotor::LINEAR_VELOCITY, TI, 200>,
            rlt::rl::utils::validation::set::Component<rlt::rl::utils::validation::metrics::MaxErrorMean<rlt::rl::utils::validation::metrics::multirotor::ANGULAR_VELOCITY, TI, 100>,
            rlt::rl::utils::validation::set::Component<rlt::rl::utils::validation::metrics::MaxErrorStd <rlt::rl::utils::validation::metrics::multirotor::ANGULAR_VELOCITY, TI, 100>,
            rlt::rl::utils::validation::set::Component<rlt::rl::utils::validation::metrics::MaxErrorMean<rlt::rl::utils::validation::metrics::multirotor::ANGULAR_VELOCITY, TI, 200>,
            rlt::rl::utils::validation::set::Component<rlt::rl::utils::validation::metrics::MaxErrorStd <rlt::rl::utils::validation::metrics::multirotor::ANGULAR_VELOCITY, TI, 200>,
            rlt::rl::utils::validation::set::Component<rlt::rl::utils::validation::metrics::MaxErrorMean<rlt::rl::utils::validation::metrics::multirotor::ANGULAR_ACCELERATION, TI, 100>,
            rlt::rl::utils::validation::set::Component<rlt::rl::utils::validation::metrics::MaxErrorStd <rlt::rl::utils::validation::metrics::multirotor::ANGULAR_ACCELERATION, TI, 100>,
            rlt::rl::utils::validation::set::Component<rlt::rl::utils::validation::metrics::MaxErrorMean<rlt::rl::utils::validation::metrics::multirotor::ANGULAR_ACCELERATION, TI, 200>,
            rlt::rl::utils::validation::set::Component<rlt::rl::utils::validation::metrics::MaxErrorStd <rlt::rl::utils::validation::metrics::multirotor::ANGULAR_ACCELERATION, TI, 200>,
            rlt::rl::utils::validation::set::FinalComponent>>>>>>>>>>>>>>>>>>>;
            using METRICS = rlt::rl::utils::validation::DefaultMetrics<ADDITIONAL_METRICS>;
        };
    }
}
