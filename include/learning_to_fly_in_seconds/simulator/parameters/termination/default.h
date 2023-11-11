
#include "../../multirotor.h"

#include <rl_tools/math/operations_generic.h>

namespace rl_tools::rl::environments::multirotor::parameters::termination{
    template<typename T, typename TI, TI ACTION_DIM, typename REWARD_FUNCTION>
    constexpr typename rl_tools::rl::environments::multirotor::ParametersBase<T, TI, 4, REWARD_FUNCTION>::MDP::Termination classic = {
            true,           // enable
            0.6,            // position
            10,         // linear velocity
            10 // angular velocity
    };
    template<typename T, typename TI, TI ACTION_DIM, typename REWARD_FUNCTION>
    constexpr typename rl_tools::rl::environments::multirotor::ParametersBase<T, TI, 4, REWARD_FUNCTION>::MDP::Termination fast_learning = {
        true,           // enable
        0.6,            // position
        1000,         // linear velocity
        1000 // angular velocity
    };
}