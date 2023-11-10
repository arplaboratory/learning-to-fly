
if(BACKPROP_TOOLS_ENABLE_JSON AND BACKPROP_TOOLS_ENABLE_BOOST_BEAST)
find_package(Boost REQUIRED)
message(STATUS "Boost version: ${Boost_VERSION}")
add_executable(ui ui/ui.cpp)
target_link_libraries(
        ui
        PRIVATE
        backprop_tools
        learning_to_fly_in_seconds
        Boost::boost
)
endif()

add_executable(training_headless training.cpp)
target_link_libraries(
        training_headless
        PRIVATE
        backprop_tools
        learning_to_fly_in_seconds
)
add_executable(training_benchmark training.cpp)
target_link_libraries(
        training_benchmark
        PRIVATE
        backprop_tools
        learning_to_fly_in_seconds
)
target_compile_definitions(training_benchmark PRIVATE LEARNING_TO_FLY_IN_SECONDS_BENCHMARK)

add_executable(ablation_study ablation_study.cpp)
target_link_libraries(
        ablation_study
        PRIVATE
        backprop_tools
        learning_to_fly_in_seconds
)


if(BACKPROP_TOOLS_BACKEND_ENABLE_CUDA)
    add_subdirectory(cuda)
endif()