#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>

#include <learning_to_fly/simulator/operations_cpu.h>
#include <learning_to_fly/simulator/metrics.h>

#include <rl_tools/nn_models/sequential/operations_generic.h>

#include <rl_tools/rl/algorithms/td3/loop.h>
namespace bpt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

#include "config.h"

#include "training_state.h"

#include "steps/checkpoint.h"
#include "steps/critic_reset.h"
#include "steps/curriculum.h"
#include "steps/log_reward.h"
#include "steps/logger.h"
#include "steps/trajectory_collection.h"
#include "steps/validation.h"

#include "helpers.h"





#include <vector>
#include <mutex>
#include <filesystem>
#include <fstream>


namespace learning_to_fly{

    template <typename T_CONFIG>
    void init(TrainingState<T_CONFIG>& ts, typename T_CONFIG::TI seed = 0){
        using CONFIG = T_CONFIG;
        using T = typename CONFIG::T;
        using TI = typename CONFIG::TI;
        using ABLATION_SPEC = typename CONFIG::ABLATION_SPEC;
        auto env_parameters = parameters::environment<T, TI, ABLATION_SPEC>::parameters;
        auto env_parameters_eval = parameters::environment<T, TI, config::template ABLATION_SPEC_EVAL<ABLATION_SPEC>>::parameters;
        for (auto& env : ts.envs) {
            env.parameters = env_parameters;
        }
        ts.env_eval.parameters = env_parameters_eval;
        TI effective_seed = CONFIG::BASE_SEED + seed;
        {
            std::stringstream run_name_ss;
            run_name_ss << "";
            auto now = std::chrono::system_clock::now();
            auto local_time = std::chrono::system_clock::to_time_t(now);
            std::tm* tm = std::localtime(&local_time);
            run_name_ss << "" << std::put_time(tm, "%Y_%m_%d_%H_%M_%S");
            if constexpr(CONFIG::BENCHMARK){
                run_name_ss << "_BENCHMARK";
            }
            run_name_ss << "_" << helpers::ablation_name<ABLATION_SPEC>();
            run_name_ss << "_" << std::setw(3) << std::setfill('0') << effective_seed;
            ts.run_name = run_name_ss.str();
        }
        {
            bpt::construct(ts.device, ts.device.logger, std::string("logs"), ts.run_name);
        }
        bpt::set_step(ts.device, ts.device.logger, 0);
        bpt::add_scalar(ts.device, ts.device.logger, "loop/seed", effective_seed);
        bpt::rl::algorithms::td3::loop::init(ts, effective_seed);
        ts.off_policy_runner.parameters = CONFIG::off_policy_runner_parameters;

        for(typename CONFIG::ENVIRONMENT& env: ts.validation_envs){
            env.parameters = parameters::environment<typename CONFIG::T, TI, ABLATION_SPEC>::parameters;
        }
        bpt::malloc(ts.device, ts.validation_actor_buffers);
        bpt::init(ts.device, ts.task, ts.validation_envs, ts.rng_eval);

        // info

        std::cout << "Environment Info: \n";
        std::cout << "\t" << "Observation dim: " << CONFIG::ENVIRONMENT::OBSERVATION_DIM << std::endl;
        std::cout << "\t" << "Observation dim privileged: " << CONFIG::ENVIRONMENT::OBSERVATION_DIM_PRIVILEGED << std::endl;
        std::cout << "\t" << "Action dim: " << CONFIG::ENVIRONMENT::ACTION_DIM << std::endl;
    }

    template <typename CONFIG>
    void step(TrainingState<CONFIG>& ts){
        using TI = typename CONFIG::TI;
        using T = typename CONFIG::T;
        if(ts.step % 10000 == 0){
            std::cout << "Step: " << ts.step << std::endl;
        }
        steps::logger(ts);
        steps::checkpoint(ts);
        steps::validation(ts);
        steps::curriculum(ts);
        bpt::rl::algorithms::td3::loop::step(ts);
        steps::trajectory_collection(ts);
    }
    template <typename CONFIG>
    void destroy(TrainingState<CONFIG>& ts){
        bpt::rl::algorithms::td3::loop::destroy(ts);
        bpt::destroy(ts.device, ts.task);
        bpt::free(ts.device, ts.validation_actor_buffers);
    }
}
