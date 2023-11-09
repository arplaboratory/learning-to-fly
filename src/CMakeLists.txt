
if(BACKPROP_TOOLS_ENABLE_JSON AND BACKPROP_TOOLS_ENABLE_BOOST_BEAST)
find_package(Boost REQUIRED)
message(STATUS "Boost version: ${Boost_VERSION}")
add_executable(rl_environments_multirotor_ui ui/ui.cpp)
target_link_libraries(
        rl_environments_multirotor_ui
        PRIVATE
        backprop_tools
        Boost::boost
        nlohmann_json::nlohmann_json
)
endif()

add_executable(rl_environments_multirotor_ui_training training.cpp)
target_link_libraries(
        rl_environments_multirotor_ui_training
        PRIVATE
        backprop_tools
)

add_executable(rl_environments_multirotor_ui_ablation_study ablation_study.cpp)
target_link_libraries(
        rl_environments_multirotor_ui_ablation_study
        PRIVATE
        backprop_tools
)


add_subdirectory(cuda)