#ifndef RL_TOOLS_SRC_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_H
#define RL_TOOLS_SRC_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_H

#include <learning_to_fly/simulator/parameters/reward_functions/abs_exp.h>
#include <learning_to_fly/simulator/parameters/reward_functions/squared.h>
#include <learning_to_fly/simulator/parameters/reward_functions/absolute.h>
#include <learning_to_fly/simulator/parameters/reward_functions/default.h>
#include <learning_to_fly/simulator/parameters/dynamics/crazy_flie.h>
#include <learning_to_fly/simulator/parameters/init/default.h>
#include <learning_to_fly/simulator/parameters/termination/default.h>

#include <rl_tools/utils/generic/typing.h>

namespace parameters{
    namespace bpt = rl_tools;
    struct DefaultAblationSpec{
        static constexpr bool DISTURBANCE = true;
        static constexpr bool OBSERVATION_NOISE = true;
        static constexpr bool ASYMMETRIC_ACTOR_CRITIC = true;
        static constexpr bool ROTOR_DELAY = true;
        static constexpr bool ACTION_HISTORY = true;
        static constexpr bool ENABLE_CURRICULUM = true;
        static constexpr bool USE_INITIAL_REWARD_FUNCTION = true;
        static constexpr bool INIT_NORMAL = true;
    };
    namespace sim2real{
        namespace builder {
            namespace bpt = RL_TOOLS_NAMESPACE_WRAPPER::rl_tools;
            using namespace bpt::rl::environments::multirotor;
            template<typename T, typename TI, typename T_ABLATION_SPEC>
            struct environment {
                using ABLATION_SPEC = T_ABLATION_SPEC;
                static constexpr auto initial_reward_function = rl_tools::rl::environments::multirotor::parameters::reward_functions::reward_squared_position_only_torque<T>;
                static constexpr auto target_reward_function = rl_tools::rl::environments::multirotor::parameters::reward_functions::reward_squared_position_only_torque_curriculum_target<T>;
                static constexpr auto reward_function = ABLATION_SPEC::USE_INITIAL_REWARD_FUNCTION ? initial_reward_function : target_reward_function;


                using REWARD_FUNCTION_CONST = typename rl_tools::utils::typing::remove_cv_t<decltype(reward_function)>;
                using REWARD_FUNCTION = typename rl_tools::utils::typing::remove_cv<REWARD_FUNCTION_CONST>::type;

                using PARAMETERS_TYPE = rl_tools::rl::environments::multirotor::ParametersDisturbances<T, TI, rl_tools::rl::environments::multirotor::ParametersBase<T, TI, 4, REWARD_FUNCTION>>;

                static_assert(ABLATION_SPEC::INIT_NORMAL);
                static constexpr auto init_params = ABLATION_SPEC::INIT_NORMAL ?
                                                    rl_tools::rl::environments::multirotor::parameters::init::orientation_biggest_angle<T, TI, 4, REWARD_FUNCTION>
                                                                               :
                                                    rl_tools::rl::environments::multirotor::parameters::init::all_positions<T, TI, 4, REWARD_FUNCTION>;

                static constexpr PARAMETERS_TYPE parameters = {
                        rl_tools::rl::environments::multirotor::parameters::dynamics::crazy_flie<T, TI, REWARD_FUNCTION>,
                        {0.01}, // integration dt
                        {
                                init_params,
                                reward_function,
                                {   // Observation noise
                                        0.001 * ABLATION_SPEC::OBSERVATION_NOISE, // position
                                        0.001 * ABLATION_SPEC::OBSERVATION_NOISE, // orientation
                                        0.002 * ABLATION_SPEC::OBSERVATION_NOISE, // linear_velocity
                                        0.002 * ABLATION_SPEC::OBSERVATION_NOISE, // angular_velocity
                                },
                                {   // Action noise
                                        0, // std of additive gaussian noise onto the normalized action (-1, 1)
                                },
                                rl_tools::rl::environments::multirotor::parameters::termination::fast_learning<T, TI, 4, REWARD_FUNCTION>
                        },
                        typename PARAMETERS_TYPE::Disturbances{
                                typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 0.027 * 9.81 / 20 * ABLATION_SPEC::DISTURBANCE}, // random_force;
                                typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 0.027 * 9.81 / 10000 * ABLATION_SPEC::DISTURBANCE} // random_torque;
                        }

                };

                using PARAMETERS = typename rl_tools::utils::typing::remove_cv_t<decltype(parameters)>;

                struct ENVIRONMENT_STATIC_PARAMETERS{
                    static constexpr TI ACTION_HISTORY_LENGTH = 32;
                    using STATE_TYPE = bpt::utils::typing::conditional_t<ABLATION_SPEC::ROTOR_DELAY,
                        bpt::utils::typing::conditional_t<ABLATION_SPEC::ACTION_HISTORY,
                            StateRotorsHistory<T, TI, ACTION_HISTORY_LENGTH, bpt::utils::typing::conditional_t<ABLATION_SPEC::DISTURBANCE, StateRandomForce<T, TI, StateBase<T, TI>>, StateBase<T, TI>>>,
                            StateRotors<T, TI, bpt::utils::typing::conditional_t<ABLATION_SPEC::DISTURBANCE, StateRandomForce<T, TI, StateBase<T, TI>>, StateBase<T, TI>>>>,
                        bpt::utils::typing::conditional_t<ABLATION_SPEC::DISTURBANCE, StateRandomForce<T, TI, StateBase<T, TI>>, StateBase<T, TI>>>;
                    using OBSERVATION_TYPE = observation::Position<observation::PositionSpecification<T, TI,
                            observation::OrientationRotationMatrix<observation::OrientationRotationMatrixSpecification<T, TI,
                                    observation::LinearVelocity<observation::LinearVelocitySpecification<T, TI,
                                            observation::AngularVelocity<observation::AngularVelocitySpecification<T, TI,
                                                    bpt::utils::typing::conditional_t<ABLATION_SPEC::ACTION_HISTORY, observation::ActionHistory<observation::ActionHistorySpecification<T, TI, ACTION_HISTORY_LENGTH>>, observation::LastComponent<TI>>>>>>>>>>;
                    using OBSERVATION_TYPE_PRIVILEGED = bpt::utils::typing::conditional_t<ABLATION_SPEC::ASYMMETRIC_ACTOR_CRITIC,
                        observation::Position<observation::PositionSpecificationPrivileged<T, TI,
                            observation::OrientationRotationMatrix<observation::OrientationRotationMatrixSpecificationPrivileged<T, TI,
                                observation::LinearVelocity<observation::LinearVelocitySpecificationPrivileged<T, TI,
                                    observation::AngularVelocity<observation::AngularVelocitySpecificationPrivileged<T, TI,
                                        bpt::utils::typing::conditional_t<ABLATION_SPEC::DISTURBANCE,
                                            observation::RandomForce<observation::RandomForceSpecification<T, TI,
                                                bpt::utils::typing::conditional_t<ABLATION_SPEC::ROTOR_DELAY,
                                                    observation::RotorSpeeds<observation::RotorSpeedsSpecification<T, TI>>,
                                                    observation::LastComponent<TI>
                                                >
                                            >>,
                                            bpt::utils::typing::conditional_t<ABLATION_SPEC::ROTOR_DELAY,
                                                    observation::RotorSpeeds<observation::RotorSpeedsSpecification<T, TI>>,
                                                    observation::LastComponent<TI>
                                            >
                                        >
                                    >>
                                >>
                            >>
                        >>,
                        observation::NONE<TI>
                    >;
                    static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
                };

                using ENVIRONMENT_SPEC = bpt::rl::environments::multirotor::Specification<T, TI, PARAMETERS, ENVIRONMENT_STATIC_PARAMETERS>;
                using ENVIRONMENT = bpt::rl::environments::Multirotor<ENVIRONMENT_SPEC>;
            };
        }
        template<typename T, typename TI, typename ABLATION_SPEC>
        using environment = builder::environment<T, TI, ABLATION_SPEC>;
    }
}

#endif
