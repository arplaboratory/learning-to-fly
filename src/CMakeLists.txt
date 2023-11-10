
if(BACKPROP_TOOLS_ENABLE_JSON AND BACKPROP_TOOLS_ENABLE_BOOST_BEAST)
find_package(Boost REQUIRED)
message(STATUS "Boost version: ${Boost_VERSION}")
add_executable(rl_environments_multirotor_ui ui/ui.cpp)
target_link_libraries(
        rl_environments_multirotor_ui
        PRIVATE
        backprop_tools
        learning_to_fly_in_seconds
        Boost::boost
)
endif()

add_executable(rl_environments_multirotor_training_headless training.cpp)
target_link_libraries(
        rl_environments_multirotor_training_headless
        PRIVATE
        backprop_tools
        learning_to_fly_in_seconds
)

add_executable(rl_environments_multirotor_ablation_study ablation_study.cpp)
target_link_libraries(
        rl_environments_multirotor_ablation_study
        PRIVATE
        backprop_tools
        learning_to_fly_in_seconds
)


if(BACKPROP_TOOLS_BACKEND_ENABLE_CUDA)
    add_subdirectory(cuda)
endif()