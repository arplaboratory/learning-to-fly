namespace learning_to_fly{
    namespace steps{
        template <typename T_CONFIG>
        void log_reward(TrainingState<T_CONFIG>& ts) {
            using T = typename T_CONFIG::T;
            using TI = typename T_CONFIG::TI;
            auto &rb = ts.off_policy_runner.replay_buffers[0];
            if (rb.position > 0 && rlt::random::uniform_real_distribution(ts.device.random, (T) 0, (T) 1, ts.rng_eval) < 0.01) {
                TI last_position = rb.position - 1;
                auto state = rlt::get(rb.states, last_position, 0);
                auto action = rlt::row(ts.device, rb.actions, last_position);
                auto next_state = rlt::get(rb.next_states, last_position, 0);
                rlt::log_reward(ts.device, ts.off_policy_runner.envs[0], state, action, next_state, ts.rng_eval);
            }
        }
    }
}
