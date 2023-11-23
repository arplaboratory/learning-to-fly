#include <rl_tools/rl/utils/validation_analysis.h>

namespace learning_to_fly {
    namespace steps {
        template <typename CONFIG>
        void validation(TrainingState<CONFIG>& ts){
            if(ts.step % 50000 == 0){
                rlt::reset(ts.device, ts.task, ts.rng_eval);
                bool completed = false;
                while(!completed){
                    completed = rlt::step(ts.device, ts.task, ts.actor_critic.actor, ts.validation_actor_buffers, ts.rng_eval);
                }
                rlt::analyse_log(ts.device, ts.task, typename TrainingState<CONFIG>::SPEC::METRICS{});
            }
        }
    }
}
