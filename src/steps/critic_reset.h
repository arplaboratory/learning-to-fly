namespace learning_to_fly {
    namespace steps {
        template <typename CONFIG>
        void critic_reset(TrainingState<CONFIG>& ts){
            using T = typename CONFIG::T;
            using TI = typename CONFIG::TI;
            if(ts.step == 500000) {
                std::cout << "Resetting critic" << std::endl;
                rlt::init_weights(ts.device, ts.actor_critic.critic_1, ts.rng);
                rlt::init_weights(ts.device, ts.actor_critic.critic_2, ts.rng);
                rlt::reset_optimizer_state(ts.device, ts.actor_critic.critic_optimizers[0], ts.actor_critic.critic_1);
                rlt::reset_optimizer_state(ts.device, ts.actor_critic.critic_optimizers[1], ts.actor_critic.critic_2);
            }
            if(ts.step == 600000){
                std::cout << "Resetting actor" << std::endl;
                rlt::init_weights(ts.device, ts.actor_critic.actor, ts.rng);
                rlt::reset_optimizer_state(ts.device, ts.actor_critic.actor_optimizer, ts.actor_critic.actor);
            }
        }
    }
}

