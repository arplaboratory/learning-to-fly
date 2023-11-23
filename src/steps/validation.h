#include <rl_tools/rl/utils/validation_analysis.h>

namespace learning_to_fly {
    namespace steps {
        template <typename CONFIG>
        void validation(TrainingState<CONFIG>& ts){
            if(ts.step % 50000 == 0){
                bpt::reset(ts.device, ts.task, ts.rng_eval);
                bool completed = false;
                while(!completed){
                    completed = bpt::step(ts.device, ts.task, ts.actor_critic.actor, ts.validation_actor_buffers, ts.rng_eval);
                }
                bpt::analyse_log(ts.device, ts.task, typename TrainingState<CONFIG>::SPEC::METRICS{});
            }
        }
    }
}
