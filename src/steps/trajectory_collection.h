namespace learning_to_fly {
    namespace steps {
        template<typename CONFIG>
        void trajectory_collection(TrainingState <CONFIG> &ts) {
            static_assert(CONFIG::N_ENVIRONMENTS == 1);
            using TI = typename CONFIG::TI;
            auto &rb = ts.off_policy_runner.replay_buffers[0];
            TI current_pos = rb.position == 0 ? CONFIG::REPLAY_BUFFER_CAP - 1 : rb.position - 1;
            typename CONFIG::ENVIRONMENT::State s = get(rb.states, current_pos, 0);
            ts.episode.push_back(s);
            if (rlt::get(rb.terminated, current_pos, 0) == 1.0) {
                {
                    std::lock_guard <std::mutex> lock(ts.trajectories_mutex);
                    ts.trajectories.push(ts.episode);
                }
                ts.episode.clear();
            }
        }

    }
}