#ifdef RL_TOOLS_ENABLE_HDF5
#include <rl_tools/containers/persist.h>
#include <rl_tools/nn/parameters/persist.h>
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>
#endif

#include <rl_tools/containers/persist_code.h>
#include <rl_tools/nn/parameters/persist_code.h>
#include <rl_tools/nn/layers/dense/persist_code.h>
#include <rl_tools/nn_models/sequential/persist_code.h>

#include <filesystem>
#include <fstream>
namespace learning_to_fly {
    namespace steps {
        template <typename T_CONFIG>
        void checkpoint(TrainingState<T_CONFIG>& ts){
            using CONFIG = T_CONFIG;
            using T = typename CONFIG::T;
            using TI = typename CONFIG::TI;
            if(CONFIG::ACTOR_ENABLE_CHECKPOINTS && (ts.step % CONFIG::ACTOR_CHECKPOINT_INTERVAL == 0)){
                const std::string ACTOR_CHECKPOINT_DIRECTORY = "checkpoints/multirotor_td3";
                std::filesystem::path actor_output_dir = std::filesystem::path(ACTOR_CHECKPOINT_DIRECTORY) / ts.run_name;
                try {
                    std::filesystem::create_directories(actor_output_dir);
                }
                catch (std::exception& e) {
                }
                std::stringstream checkpoint_name_ss;
                checkpoint_name_ss << "actor_" << std::setw(15) << std::setfill('0') << ts.step;
                std::string checkpoint_name = checkpoint_name_ss.str();

#if defined(RL_TOOLS_ENABLE_HDF5) && !defined(RL_TOOLS_DISABLE_HDF5)
                std::filesystem::path actor_output_path_hdf5 = actor_output_dir / (checkpoint_name + ".h5");
                std::cout << "Saving actor checkpoint " << actor_output_path_hdf5 << std::endl;
                try{
                    auto actor_file = HighFive::File(actor_output_path_hdf5.string(), HighFive::File::Overwrite);
                    rlt::save(ts.device, ts.actor_critic.actor, actor_file.createGroup("actor"));
                }
                catch(HighFive::Exception& e){
                    std::cout << "Error while saving actor: " << e.what() << std::endl;
                }
#endif
                {
                    // Since checkpointing a full Adam model to code (including gradients and moments of the weights and biases currently does not work)
                    typename CONFIG::ACTOR_CHECKPOINT_TYPE actor_checkpoint;
                    typename decltype(ts.actor_critic.actor)::template DoubleBuffer<1> actor_buffer;
                    typename decltype(actor_checkpoint)::template DoubleBuffer<1> actor_checkpoint_buffer;
                    rlt::malloc(ts.device, actor_checkpoint);
                    rlt::malloc(ts.device, actor_buffer);
                    rlt::malloc(ts.device, actor_checkpoint_buffer);
                    rlt::copy(ts.device, ts.device, ts.actor_critic.actor, actor_checkpoint);
                    std::filesystem::path actor_output_path_code = actor_output_dir / (checkpoint_name + ".h");
                    auto actor_weights = rlt::save_code(ts.device, actor_checkpoint, std::string("rl_tools::checkpoint::actor"), true);
                    std::cout << "Saving checkpoint at: " << actor_output_path_code << std::endl;
                    std::ofstream actor_output_file(actor_output_path_code);
                    actor_output_file << actor_weights;
                    {
                        typename CONFIG::ENVIRONMENT_EVALUATION::State state;
                        rlt::sample_initial_state(ts.device, ts.envs[0], state, ts.rng_eval);
                        rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, CONFIG::ENVIRONMENT_EVALUATION::OBSERVATION_DIM>> observation;
                        rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, CONFIG::ENVIRONMENT::ACTION_DIM>> action;
                        rlt::malloc(ts.device, observation);
                        rlt::malloc(ts.device, action);
                        auto rng_copy = ts.rng_eval;
                        rlt::observe(ts.device, ts.env_eval, state, observation, rng_copy);
                        rlt::evaluate(ts.device, ts.actor_critic.actor, observation, action, actor_buffer);
                        rlt::evaluate(ts.device, actor_checkpoint, observation, action, actor_checkpoint_buffer);
                        actor_output_file << "\n" << rlt::save_code(ts.device, observation, std::string("rl_tools::checkpoint::observation"), true);
                        actor_output_file << "\n" << rlt::save_code(ts.device, action, std::string("rl_tools::checkpoint::action"), true);
                        actor_output_file << "\n" << "namespace rl_tools::checkpoint::meta{";
                        actor_output_file << "\n" << "   " << "char name[] = \"" << ts.run_name << "_" << checkpoint_name << "\";";
                        actor_output_file << "\n" << "   " << "char commit_hash[] = \"" << RL_TOOLS_STRINGIFY(RL_TOOLS_COMMIT_HASH) << "\";";
                        actor_output_file << "\n" << "}";
                        rlt::free(ts.device, observation);
                        rlt::free(ts.device, action);
                    }
                    rlt::free(ts.device, actor_checkpoint);
                    rlt::free(ts.device, actor_buffer);
                    rlt::free(ts.device, actor_checkpoint_buffer);
                }
            }
        }
    }
}
