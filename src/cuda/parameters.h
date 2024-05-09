#include <learning_to_fly/simulator/parameters/reward_functions/abs_exp.h>
#include <learning_to_fly/simulator/parameters/reward_functions/squared.h>
#include <learning_to_fly/simulator/parameters/reward_functions/default.h>
#include <learning_to_fly/simulator/parameters/dynamics/crazy_flie.h>
#include <learning_to_fly/simulator/parameters/init/default.h>
#include <learning_to_fly/simulator/parameters/termination/default.h>

namespace parameters_crazyflie{
    namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;
    template<typename T, typename TI>
    struct environment{
        static constexpr auto reward_function = rlt::rl::environments::multirotor::parameters::reward_functions::reward_old_but_gold<T>;
        using REWARD_FUNCTION_CONST = typename rlt::utils::typing::remove_cv_t<decltype(reward_function)>;
        using REWARD_FUNCTION = typename rlt::utils::typing::remove_cv<REWARD_FUNCTION_CONST>::type;

        using PARAMETERS_TYPE = rlt::rl::environments::multirotor::ParametersDisturbances<T, TI, rlt::rl::environments::multirotor::ParametersBase<T, TI, 4, REWARD_FUNCTION>>;
        static constexpr PARAMETERS_TYPE parameters = {
                rlt::rl::environments::multirotor::parameters::dynamics::crazy_flie<T, TI, REWARD_FUNCTION>,
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
                        rlt::rl::environments::multirotor::parameters::termination::fast_learning<T, TI, 4, REWARD_FUNCTION>
                },
                typename PARAMETERS_TYPE::Disturbances{
                        typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 0.027 * 9.81 / 10}, // random_force;
                        typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 0} // random_torque;
                }
        };

        using PARAMETERS = typename rlt::utils::typing::remove_cv_t<decltype(parameters)>;
        using ENVIRONMENT_SPEC = rlt::rl::environments::multirotor::Specification<T, TI, PARAMETERS, rlt::rl::environments::multirotor::StaticParametersDefault<T, TI>>;
        using ENVIRONMENT = rlt::rl::environments::Multirotor<ENVIRONMENT_SPEC>;
    };
}
