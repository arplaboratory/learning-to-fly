namespace learning_to_fly {
    namespace steps {
        template <typename T_CONFIG>
        void logger(TrainingState<T_CONFIG>& ts){
            bpt::set_step(ts.device, ts.device.logger, ts.step);
        }
    }
}
