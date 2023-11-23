#include <learning_to_fly/simulator/parameters/reward_functions/abs_exp.h>
#include <learning_to_fly/simulator/parameters/reward_functions/squared.h>
#include <learning_to_fly/simulator/parameters/reward_functions/default.h>
#include <learning_to_fly/simulator/parameters/dynamics/crazy_flie.h>
#include <learning_to_fly/simulator/parameters/init/default.h>
#include <learning_to_fly/simulator/parameters/termination/default.h>

namespace parameters_sim2real{
    namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;
    template<typename T, typename TI>
    struct environment{
//        static constexpr auto reward_function = rlt::rl::environments::multirotor::parameters::reward_functions::reward_old_but_gold_4<T>;
//        static constexpr auto reward_function = rlt::rl::environments::multirotor::parameters::reward_functions::reward_mm<T, TI>;
//        static constexpr auto reward_function = rlt::rl::environments::multirotor::parameters::reward_functions::sq_exp_position_action_only_3<T>;
//        static constexpr auto reward_function = rlt::rl::environments::multirotor::parameters::reward_functions::sq_exp_reward_mm<T, TI>;
        static constexpr auto reward_function = rlt::rl::environments::multirotor::parameters::reward_functions::reward_squared_position_only_torque<T>;
        using REWARD_FUNCTION_CONST = typename rlt::utils::typing::remove_cv_t<decltype(reward_function)>;
        using REWARD_FUNCTION = typename rlt::utils::typing::remove_cv<REWARD_FUNCTION_CONST>::type;

        using PARAMETERS_TYPE = rlt::rl::environments::multirotor::ParametersDisturbances<T, TI, rlt::rl::environments::multirotor::ParametersBase<T, TI, 4, REWARD_FUNCTION>>;
        static constexpr PARAMETERS_TYPE parameters = {
                rlt::rl::environments::multirotor::parameters::dynamics::crazy_flie_old_reduced_inertia<T, TI, REWARD_FUNCTION>,
                {0.01}, // integration dt
                {
//                        rlt::rl::environments::multirotor::parameters::init::all_around_orientation_only<T, TI, 4, REWARD_FUNCTION>,
                        rlt::rl::environments::multirotor::parameters::init::all_around_2<T, TI, 4, REWARD_FUNCTION>,
//                        rlt::rl::environments::multirotor::parameters::init::orientation_all_around<T, TI, 4, REWARD_FUNCTION>,
//                        rlt::rl::environments::multirotor::parameters::init::simple<T, TI, 4, REWARD_FUNCTION>,
                        reward_function,
                        {   // Observation noise
                            0.001, // position
                            0.001, // orientation
                            0.002, // linear_velocity
                            0.002, // angular_velocity
                        },
                        {   // Action noise
                            0, // std of additive gaussian noise onto the normalized action (-1, 1)
                        },
                        rlt::rl::environments::multirotor::parameters::termination::fast_learning<T, TI, 4, REWARD_FUNCTION>
                },
                typename PARAMETERS_TYPE::Disturbances{
                        typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 0.027 * 9.81 / 20}, // random_force;
//                        typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 0} // random_force;
                        typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 0.027 * 9.81 / 10000} // random_torque;
                }

        };

        using PARAMETERS = typename rlt::utils::typing::remove_cv_t<decltype(parameters)>;

//        struct ENVIRONMENT_STATIC_PARAMETERS: rlt::rl::environments::multirotor::StaticParametersDefault<T, TI>{
//            static constexpr bool ENFORCE_POSITIVE_QUATERNION = false;
//            static constexpr bool RANDOMIZE_QUATERNION_SIGN = false;
//            static constexpr rlt::rl::environments::multirotor::LatentStateType LATENT_STATE_TYPE = rlt::rl::environments::multirotor::LatentStateType::Empty;
//            static constexpr rlt::rl::environments::multirotor::StateType STATE_TYPE = rlt::rl::environments::multirotor::StateType::BaseRotors;
//            static constexpr rlt::rl::environments::multirotor::ObservationType OBSERVATION_TYPE = rlt::rl::environments::multirotor::ObservationType::RotationMatrix;
//        };

        using ENVIRONMENT_SPEC = rlt::rl::environments::multirotor::Specification<T, TI, PARAMETERS, rlt::rl::environments::multirotor::StaticParametersDefault<T, TI>>;
        using ENVIRONMENT = rlt::rl::environments::Multirotor<ENVIRONMENT_SPEC>;
    };
}

namespace parameters_fast_learning{
    namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;
    template<typename T, typename TI>
    struct environment{
        static constexpr auto reward_function = rlt::rl::environments::multirotor::parameters::reward_functions::reward_old_but_gold<T>;
//        static constexpr auto reward_function = rlt::rl::environments::multirotor::parameters::reward_functions::reward_mm<T, TI>;
//        static constexpr auto reward_function = rlt::rl::environments::multirotor::parameters::reward_functions::reward_squared_2<T>;
//        static constexpr auto reward_function = rlt::rl::environments::multirotor::parameters::reward_functions::reward_squared_4<T>;
        using REWARD_FUNCTION_CONST = typename rlt::utils::typing::remove_cv_t<decltype(reward_function)>;
        using REWARD_FUNCTION = typename rlt::utils::typing::remove_cv<REWARD_FUNCTION_CONST>::type;

        using PARAMETERS_TYPE = rlt::rl::environments::multirotor::ParametersDisturbances<T, TI, rlt::rl::environments::multirotor::ParametersBase<T, TI, 4, REWARD_FUNCTION>>;
        static constexpr PARAMETERS_TYPE parameters = {
                rlt::rl::environments::multirotor::parameters::dynamics::crazy_flie_old<T, TI, REWARD_FUNCTION>,
                {0.01}, // integration dt
                {
                        rlt::rl::environments::multirotor::parameters::init::all_around<T, TI, 4, REWARD_FUNCTION>,
                        reward_function,
                        {   // Observation noise
                            0, // position
                            0, // orientation
                            0, // linear_velocity
                            0, // angular_velocity
                        },
                        {   // Action noise
                            0, // std of additive gaussian noise onto the normalized action (-1, 1)
                        },
//                        rlt::rl::environments::multirotor::parameters::init::all_around_simplified<T, TI, 4, REWARD_FUNCTION>,
//                        rlt::rl::environments::multirotor::parameters::init::simple<T, TI, 4, REWARD_FUNCTION>,
                        rlt::rl::environments::multirotor::parameters::termination::fast_learning<T, TI, 4, REWARD_FUNCTION>
                },
                typename PARAMETERS_TYPE::Disturbances{
                        typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 0.027 * 9.81 / 10}, // random_force;
                        typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 0} // random_torque;
                }
        };

        using PARAMETERS = typename rlt::utils::typing::remove_cv_t<decltype(parameters)>;

//        struct ENVIRONMENT_STATIC_PARAMETERS: rlt::rl::environments::multirotor::StaticParametersDefault<T, TI>{
//            static constexpr bool ENFORCE_POSITIVE_QUATERNION = false;
//            static constexpr bool RANDOMIZE_QUATERNION_SIGN = false;
//            static constexpr rlt::rl::environments::multirotor::StateType STATE_TYPE = rlt::rl::environments::multirotor::StateType::Base;
//            static constexpr rlt::rl::environments::multirotor::ObservationType OBSERVATION_TYPE = rlt::rl::environments::multirotor::ObservationType::Normal;
//        };

        using ENVIRONMENT_SPEC = rlt::rl::environments::multirotor::Specification<T, TI, PARAMETERS, rlt::rl::environments::multirotor::StaticParametersDefault<T, TI>>;
        using ENVIRONMENT = rlt::rl::environments::Multirotor<ENVIRONMENT_SPEC>;
    };
}
