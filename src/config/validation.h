namespace learning_to_fly{
    namespace config{
        template <typename T_SUPER_CONFIG>
        struct Validation: T_SUPER_CONFIG{
            using SUPER = T_SUPER_CONFIG;
            using T = typename SUPER::T;
            using TI = typename SUPER::TI;
            using ENVIRONMENT = typename SUPER::ENVIRONMENT;
            using VALIDATION_SPEC = rlt::rl::utils::validation::Specification<T, TI, ENVIRONMENT>;
            static constexpr TI VALIDATION_N_EPISODES = 10;
            static constexpr TI VALIDATION_MAX_EPISODE_LENGTH = SUPER::ENVIRONMENT_STEP_LIMIT;
            using TASK_SPEC = rlt::rl::utils::validation::TaskSpecification<VALIDATION_SPEC, VALIDATION_N_EPISODES, VALIDATION_MAX_EPISODE_LENGTH>;
            using ADDITIONAL_METRICS = rlt::rl::utils::validation::set::Component<
            rlt::rl::utils::validation::metrics::SettlingFractionPosition<TI, 200>,
            rlt::rl::utils::validation::set::Component<rlt::rl::utils::validation::metrics::MaxErrorMean<rlt::rl::utils::validation::metrics::multirotor::POSITION, TI, 100>,
            rlt::rl::utils::validation::set::Component<rlt::rl::utils::validation::metrics::MaxErrorStd <rlt::rl::utils::validation::metrics::multirotor::POSITION, TI, 100>,
            rlt::rl::utils::validation::set::Component<rlt::rl::utils::validation::metrics::MaxErrorMean<rlt::rl::utils::validation::metrics::multirotor::POSITION, TI, 200>,
            rlt::rl::utils::validation::set::Component<rlt::rl::utils::validation::metrics::MaxErrorStd <rlt::rl::utils::validation::metrics::multirotor::POSITION, TI, 200>,
            rlt::rl::utils::validation::set::Component<rlt::rl::utils::validation::metrics::MaxErrorMean<rlt::rl::utils::validation::metrics::multirotor::ANGLE, TI, 100>,
            rlt::rl::utils::validation::set::Component<rlt::rl::utils::validation::metrics::MaxErrorStd <rlt::rl::utils::validation::metrics::multirotor::ANGLE, TI, 200>,
            rlt::rl::utils::validation::set::Component<rlt::rl::utils::validation::metrics::MaxErrorMean<rlt::rl::utils::validation::metrics::multirotor::LINEAR_VELOCITY, TI, 100>,
            rlt::rl::utils::validation::set::Component<rlt::rl::utils::validation::metrics::MaxErrorStd <rlt::rl::utils::validation::metrics::multirotor::LINEAR_VELOCITY, TI, 100>,
            rlt::rl::utils::validation::set::Component<rlt::rl::utils::validation::metrics::MaxErrorMean<rlt::rl::utils::validation::metrics::multirotor::LINEAR_VELOCITY, TI, 200>,
            rlt::rl::utils::validation::set::Component<rlt::rl::utils::validation::metrics::MaxErrorStd <rlt::rl::utils::validation::metrics::multirotor::LINEAR_VELOCITY, TI, 200>,
            rlt::rl::utils::validation::set::Component<rlt::rl::utils::validation::metrics::MaxErrorMean<rlt::rl::utils::validation::metrics::multirotor::ANGULAR_VELOCITY, TI, 100>,
            rlt::rl::utils::validation::set::Component<rlt::rl::utils::validation::metrics::MaxErrorStd <rlt::rl::utils::validation::metrics::multirotor::ANGULAR_VELOCITY, TI, 100>,
            rlt::rl::utils::validation::set::Component<rlt::rl::utils::validation::metrics::MaxErrorMean<rlt::rl::utils::validation::metrics::multirotor::ANGULAR_VELOCITY, TI, 200>,
            rlt::rl::utils::validation::set::Component<rlt::rl::utils::validation::metrics::MaxErrorStd <rlt::rl::utils::validation::metrics::multirotor::ANGULAR_VELOCITY, TI, 200>,
            rlt::rl::utils::validation::set::Component<rlt::rl::utils::validation::metrics::MaxErrorMean<rlt::rl::utils::validation::metrics::multirotor::ANGULAR_ACCELERATION, TI, 100>,
            rlt::rl::utils::validation::set::Component<rlt::rl::utils::validation::metrics::MaxErrorStd <rlt::rl::utils::validation::metrics::multirotor::ANGULAR_ACCELERATION, TI, 100>,
            rlt::rl::utils::validation::set::Component<rlt::rl::utils::validation::metrics::MaxErrorMean<rlt::rl::utils::validation::metrics::multirotor::ANGULAR_ACCELERATION, TI, 200>,
            rlt::rl::utils::validation::set::Component<rlt::rl::utils::validation::metrics::MaxErrorStd <rlt::rl::utils::validation::metrics::multirotor::ANGULAR_ACCELERATION, TI, 200>,
            rlt::rl::utils::validation::set::FinalComponent>>>>>>>>>>>>>>>>>>>;
            using METRICS = rlt::rl::utils::validation::DefaultMetrics<ADDITIONAL_METRICS>;
        };
    }
}