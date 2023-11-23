namespace learning_to_fly{
    namespace helpers{
        template <typename ABLATION_SPEC>
        std::string ablation_name(){
            std::string n = "";
            n += std::string("d") + (ABLATION_SPEC::DISTURBANCE ? "+"  : "-");
            n += std::string("o") + (ABLATION_SPEC::OBSERVATION_NOISE ? "+"  : "-");
            n += std::string("a") + (ABLATION_SPEC::ASYMMETRIC_ACTOR_CRITIC ? "+"  : "-");
            n += std::string("r") + (ABLATION_SPEC::ROTOR_DELAY ? "+"  : "-");
            n += std::string("h") + (ABLATION_SPEC::ACTION_HISTORY ? "+"  : "-");
            n += std::string("c") + (ABLATION_SPEC::ENABLE_CURRICULUM ? "+"  : "-");
            n += std::string("f") + (ABLATION_SPEC::USE_INITIAL_REWARD_FUNCTION ? "+"  : "-");
            n += std::string("w") + (ABLATION_SPEC::RECALCULATE_REWARDS ? "+"  : "-");
            n += std::string("e") + (ABLATION_SPEC::EXPLORATION_NOISE_DECAY ? "+"  : "-");
            return n;
        }
        template <typename ABLATION_SPEC, typename CONFIG>
        std::string run_name(typename CONFIG::TI seed) {
            std::stringstream run_name_ss;
            run_name_ss << "";
            auto now = std::chrono::system_clock::now();
            auto local_time = std::chrono::system_clock::to_time_t(now);
            std::tm *tm = std::localtime(&local_time);
            run_name_ss << "" << std::put_time(tm, "%Y_%m_%d_%H_%M_%S");
            if constexpr (CONFIG::BENCHMARK) {
                run_name_ss << "_BENCHMARK";
            }
            run_name_ss << "_" << helpers::ablation_name<ABLATION_SPEC>();
            run_name_ss << "_" << std::setw(3) << std::setfill('0') << seed;
            return run_name_ss.str();
        }
    }
}
