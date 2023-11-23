#include <rl_tools/operations/cpu.h>

#include <learning_to_fly/simulator/operations_cpu.h>
#include <learning_to_fly/simulator/ui.h>
#include <rl_tools/nn_models/operations_cpu.h>
#include <rl_tools/nn_models/persist.h>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

#include "parameters_training.h"
#include "training.h"

#include <chrono>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <thread>
#include <highfive/H5File.hpp>
#include <CLI/CLI.hpp>

namespace TEST_DEFINITIONS{
    using DEVICE = rlt::devices::DefaultCPU;
    using T = float;
    using TI = typename DEVICE::index_t;
    namespace parameter_set = parameters_0;
    template <typename BASE_SPEC>
    struct SpecEval: BASE_SPEC{
        static constexpr bool DISTURBANCE = true;
        static constexpr bool OBSERVATION_NOISE = true;
        static constexpr bool ROTOR_DELAY = true;
        static constexpr bool ACTION_HISTORY = BASE_SPEC::ROTOR_DELAY && BASE_SPEC::ACTION_HISTORY;
        static constexpr bool USE_INITIAL_REWARD_FUNCTION = false;
        static constexpr bool INIT_NORMAL = true;
    };
    using EVAL_SPEC = SpecEval<parameters::DefaultAblationSpec>;
    using CONFIG = learning_to_fly::config::Config<EVAL_SPEC>;

    using penv = parameter_set::environment<T, TI, EVAL_SPEC>;
    using ENVIRONMENT = penv::ENVIRONMENT;
    using UI = rlt::rl::environments::multirotor::UI<ENVIRONMENT>;

    constexpr bool TRAJECTORY_TRACKING = false;
    constexpr TI MAX_EPISODE_LENGTH = TRAJECTORY_TRACKING ? 3000 : 600;
    constexpr bool SAME_CONFIG_AS_IN_TRAINING = false;
    constexpr bool RANDOMIZE_DOMAIN_PARAMETERS = false;
    constexpr bool INIT_SIMPLE = false;
    constexpr bool DEACTIVATE_OBSERVATION_NOISE = true;
    constexpr bool INJECT_EXPLORATION_NOISE = false;
    constexpr bool DISABLE_DISTURBANCES = true;
    constexpr bool AMPLIFY_DISTURBANCES = false;
    constexpr TI N_ENVIRONMENTS = 100;
    constexpr T max_pos_diff = 0.6;
    constexpr T max_vel_diff = 5;
    constexpr T time_lapse = 0.05;
}

template <typename T>
void trajectory(T t, T position[3], T velocity[3]){
    constexpr T figure_eight_scale = 2;
    constexpr T figure_eight_interval = 5.5;
    T speed = 1 / figure_eight_interval;
    T progress = t * speed;
    position[1] = cosf(progress*2*M_PI + M_PI / 2) * figure_eight_scale;
    velocity[1] = -sinf(progress*2*M_PI + M_PI / 2) * speed * 2 * M_PI;
    position[0] = sinf(2*(progress*2*M_PI + M_PI / 2)) / 2.0f * figure_eight_scale;
    velocity[0] = cosf(2*(progress*2*M_PI + M_PI / 2)) / 2.0f * speed * 4 * M_PI;
    position[2] = 0;
    velocity[2] = 0;
}


int main(int argc, char** argv) {
    using namespace TEST_DEFINITIONS;
    CLI::App app;
    std::string arg_run = "", arg_checkpoint = "";
    DEVICE::index_t startup_timeout = 0;
    app.add_option("--run", arg_run, "path to the run's directory");
    app.add_option("--checkpoint", arg_checkpoint, "path to the checkpoint");
    app.add_option("--timeout", startup_timeout, "time to wait after first render");

    CLI11_PARSE(app, argc, argv);
    DEVICE dev;
    ENVIRONMENT env;
    env.parameters = penv::parameters;
    UI uis[N_ENVIRONMENTS];
    typename CONFIG::ACTOR_TYPE actor;
    typename CONFIG::ACTOR_TYPE::template DoubleBuffer<1> actor_buffer;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> action;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>> observation;
    typename ENVIRONMENT::State states[N_ENVIRONMENTS], target_states[N_ENVIRONMENTS], observation_states[N_ENVIRONMENTS], observation_states_clamped[N_ENVIRONMENTS], next_states[N_ENVIRONMENTS];
    auto rng = rlt::random::default_engine(DEVICE::SPEC::RANDOM(), 10);

    rlt::malloc(dev, env);
    rlt::malloc(dev, actor);
    rlt::malloc(dev, actor_buffer);
    rlt::malloc(dev, action);
    rlt::malloc(dev, observation);

    for(TI env_i=0; env_i < N_ENVIRONMENTS; env_i++){
        auto& ui = uis[env_i];
        ui.host = "localhost";
        ui.port = "8080";
        ui.id = env_i;
        rlt::init(dev, env, ui);
    }
    DEVICE::index_t episode_i = 0;
    std::string run = arg_run;
    std::string checkpoint = arg_checkpoint;
    while(true){
        std::filesystem::path actor_run;
        if(run == "" && checkpoint == ""){
            std::filesystem::path actor_checkpoints_dir = std::filesystem::path("checkpoints") / "multirotor_td3";
            std::vector<std::filesystem::path> actor_runs;

            for (const auto& run : std::filesystem::directory_iterator(actor_checkpoints_dir)) {
                if (run.is_directory()) {
                    actor_runs.push_back(run.path());
                }
            }
            std::sort(actor_runs.begin(), actor_runs.end());
            actor_run = actor_runs.back();
        }
        else{
            actor_run = run;
        }
        if(checkpoint == ""){
            std::vector<std::filesystem::path> actor_checkpoints;
            for (const auto& checkpoint : std::filesystem::directory_iterator(actor_run)) {
                if (checkpoint.is_regular_file()) {
                    if(checkpoint.path().extension() == ".h5" || checkpoint.path().extension() == ".hdf5"){
                        actor_checkpoints.push_back(checkpoint.path());
                    }
                }
            }
            std::sort(actor_checkpoints.begin(), actor_checkpoints.end());
            checkpoint = actor_checkpoints.back().string();
        }

        std::cout << "Loading actor from " << checkpoint << std::endl;
        {
            try{
                auto data_file = HighFive::File(checkpoint, HighFive::File::ReadOnly);
                rlt::load(dev, actor, data_file.getGroup("actor"));
            }
            catch(HighFive::FileException& e){
                std::cout << "Failed to load actor from " << checkpoint << std::endl;
                std::cout << "Error: " << e.what() << std::endl;
                continue;
            }
        }
        if(arg_checkpoint == ""){
            checkpoint = "";
        }
        if(arg_run == ""){
            run = "";
        }

        T reward_acc = 0;
        env.parameters = penv::parameters;
        if(!SAME_CONFIG_AS_IN_TRAINING && INIT_SIMPLE){
            env.parameters.mdp.init = rlt::rl::environments::multirotor::parameters::init::simple<T, TI, 4, penv::REWARD_FUNCTION>;
        }
        if(!SAME_CONFIG_AS_IN_TRAINING && DEACTIVATE_OBSERVATION_NOISE){
            env.parameters.mdp.observation_noise.position = 0;
            env.parameters.mdp.observation_noise.orientation = 0;
            env.parameters.mdp.observation_noise.linear_velocity = 0;
            env.parameters.mdp.observation_noise.angular_velocity = 0;
        }
        if(!SAME_CONFIG_AS_IN_TRAINING && RANDOMIZE_DOMAIN_PARAMETERS && episode_i % 2 == 0){
//            T mass_factor = rlt::random::uniform_real_distribution(DEVICE::SPEC::RANDOM(), (T)0.5, (T)1.5, rng);
//            T J_factor = rlt::random::uniform_real_distribution(DEVICE::SPEC::RANDOM(), (T)0.5, (T)5.0, rng);
//            T max_rpm_factor = rlt::random::uniform_real_distribution(DEVICE::SPEC::RANDOM(), (T)0.8, (T)1.2, rng);
            T mass_factor = 1;
            T J_factor = 1;
            T max_rpm_factor = 1;
//            T rpm_time_constant_factor = rlt::random::uniform_real_distribution(DEVICE::SPEC::RANDOM(), (T)1.0, (T)2, rng);
            T rpm_time_constant_factor = 1;
            std::cout << "Randomizing domain parameters" << std::endl;
            std::cout << "Mass factor: " << mass_factor << std::endl;
            std::cout << "J factor: " << J_factor << std::endl;
            std::cout << "Max RPM factor: " << max_rpm_factor << std::endl;
            std::cout << "RPM time constant factor: " << rpm_time_constant_factor << std::endl;
            env.parameters.dynamics.mass *= mass_factor;
            env.parameters.dynamics.J[0][0] *= J_factor;
            env.parameters.dynamics.J[1][1] *= J_factor;
            env.parameters.dynamics.J[2][2] *= J_factor;
            env.parameters.dynamics.J_inv[0][0] /= J_factor;
            env.parameters.dynamics.J_inv[1][1] /= J_factor;
            env.parameters.dynamics.J_inv[2][2] /= J_factor;
            env.parameters.dynamics.action_limit.max *= max_rpm_factor;
            env.parameters.dynamics.rpm_time_constant *= rpm_time_constant_factor;
        }
        else{
            std::cout << "Using nominal domain parameters" << std::endl;
        }
        if(!SAME_CONFIG_AS_IN_TRAINING && INJECT_EXPLORATION_NOISE){
            env.parameters.mdp.action_noise.normalized_rpm = 0.1;
        }
        if(!SAME_CONFIG_AS_IN_TRAINING && DISABLE_DISTURBANCES){
            env.parameters.disturbances.random_force.mean = 0;
            env.parameters.disturbances.random_force.std = 0;
        }
        else{
            if(AMPLIFY_DISTURBANCES){
                env.parameters.disturbances.random_force.std *= 2;
            }
        }
        env.parameters.mdp.init.guidance = 0;
        for(TI env_i=0; env_i < N_ENVIRONMENTS; env_i++){
            rlt::sample_initial_state(dev, env, states[env_i], rng);
//            states[env_i].position[0] *= 3;
//            states[env_i].position[1] *= 3;
//            states[env_i].position[2] *= 3;
            rlt::set_state(dev, uis[env_i], states[env_i]);
        }
        std::this_thread::sleep_for(std::chrono::seconds(10));
        std::cout << "Random force: " << states[0].force[0] << ", " << states[0].force[1] << ", " << states[0].force[2] << std::endl;
        T max_speed = 0;
        constexpr TI TRACKING_START_STEP = 100;

        T tracking_error = 0;
        for(int step_i = 0; step_i < MAX_EPISODE_LENGTH; step_i++){
            auto start = std::chrono::high_resolution_clock::now();
            for(TI env_i=0; env_i < N_ENVIRONMENTS; env_i++){
                auto& ui = uis[env_i];
                auto& state = states[env_i];
                auto& target_state = target_states[env_i];
                auto& observation_state = observation_states[env_i];
                auto& observation_state_clamped = observation_states_clamped[env_i];
                auto& next_state = next_states[env_i];

                if(TRAJECTORY_TRACKING && step_i >= TRACKING_START_STEP){
                    target_state = state;
                    trajectory(((T)step_i - TRACKING_START_STEP) * env.parameters.integration.dt, target_state.position, target_state.linear_velocity);
                    observation_state = state;
                    observation_state_clamped = state;
                    observation_state.position[0] = state.position[0] - target_state.position[0];
                    observation_state.position[1] = state.position[1] - target_state.position[1];
                    observation_state.position[2] = state.position[2] - target_state.position[2];
                    observation_state_clamped.position[0] = rlt::math::clamp(dev.math, observation_state.position[0], -max_pos_diff, max_pos_diff);
                    observation_state_clamped.position[1] = rlt::math::clamp(dev.math, observation_state.position[1], -max_pos_diff, max_pos_diff);
                    observation_state_clamped.position[2] = rlt::math::clamp(dev.math, observation_state.position[2], -max_pos_diff, max_pos_diff);

                    observation_state.linear_velocity[0] = state.linear_velocity[0] - target_state.linear_velocity[0];
                    observation_state.linear_velocity[1] = state.linear_velocity[1] - target_state.linear_velocity[1];
                    observation_state.linear_velocity[2] = state.linear_velocity[2] - target_state.linear_velocity[2];
                    observation_state_clamped.linear_velocity[0] = rlt::math::clamp(dev.math, observation_state.linear_velocity[0], -max_vel_diff, max_vel_diff);
                    observation_state_clamped.linear_velocity[1] = rlt::math::clamp(dev.math, observation_state.linear_velocity[1], -max_vel_diff, max_vel_diff);
                    observation_state_clamped.linear_velocity[2] = rlt::math::clamp(dev.math, observation_state.linear_velocity[2], -max_vel_diff, max_vel_diff);

                    tracking_error += rlt::math::sqrt(dev.math, observation_state.position[0] * observation_state.position[0] + observation_state.position[1] * observation_state.position[1]); // + observation_state.position[2] * observation_state.position[2]);
                    std::cout << "Tracking error: " << tracking_error/(step_i - TRACKING_START_STEP + 1) << std::endl;
                }
                else{
                    observation_state = state;
                    observation_state_clamped.position[0] = rlt::math::clamp(dev.math, observation_state.position[0], -max_pos_diff, max_pos_diff);
                    observation_state_clamped.position[1] = rlt::math::clamp(dev.math, observation_state.position[1], -max_pos_diff, max_pos_diff);
                    observation_state_clamped.position[2] = rlt::math::clamp(dev.math, observation_state.position[2], -max_pos_diff, max_pos_diff);
                    observation_state_clamped = state;
                }
                rlt::observe(dev, env, observation_state_clamped, observation, rng);
                rlt::evaluate(dev, actor, observation, action, actor_buffer);
                rlt::clamp(dev, action, (T)-1, (T)1);
                T dt = rlt::step(dev, env, state, action, next_state, rng);
                bool terminated_flag = rlt::terminated(dev, env, observation_state, rng);
                reward_acc += rlt::reward(dev, env, state, action, next_state, rng);
                rlt::set_state(dev, ui, state, action);
                state = next_state;
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> diff = end-start;
                if(startup_timeout > 0 && episode_i == 0 && step_i == 0){
                    for(int timeout_step_i = 0; timeout_step_i < startup_timeout; timeout_step_i++){
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        if(timeout_step_i % 100 == 0){
                            rlt::set_state(dev, ui, state, action);
                        }
                    }
                }
                T speed = rlt::math::sqrt(dev.math, next_state.linear_velocity[0] * next_state.linear_velocity[0] + next_state.linear_velocity[1] * next_state.linear_velocity[1] + next_state.linear_velocity[2] * next_state.linear_velocity[2]);
                if(speed > max_speed){
                    max_speed = speed;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds((int)((dt/time_lapse - diff.count())*1000)));
                if(terminated_flag || step_i == (MAX_EPISODE_LENGTH - 1)){
                    std::cout << "Episode terminated after " << step_i << " steps with reward " << reward_acc << "(max speed: " << max_speed << ")" << std::endl;
                }
            }
        }
        episode_i++;
    }
}

