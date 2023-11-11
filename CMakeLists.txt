cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0003 NEW)


project(learning_to_fly)

set(CMAKE_CXX_STANDARD 17)

find_package(HDF5)
set(RL_TOOLS_ENABLE_HDF5 ON)
set(RL_TOOLS_ENABLE_TENSORBOARD ON)
set(RL_TOOLS_ENABLE_JSON ON)
set(RL_TOOLS_ENABLE_BOOST_BEAST ON)

if(APPLE)
    set(RL_TOOLS_BACKEND_ENABLE_ACCELERATE ON)
endif()

add_library(learning_to_fly INTERFACE)
target_include_directories(learning_to_fly INTERFACE include)

#set(RL_TOOLS_ENABLE_TESTS ON)
add_subdirectory(external/rl_tools)
add_subdirectory(src)

set(LEARNING_TO_FLY_IN_SECONDS_ENABLE_TESTS ON)

if(LEARNING_TO_FLY_IN_SECONDS_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()