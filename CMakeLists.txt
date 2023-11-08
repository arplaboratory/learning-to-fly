cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0003 NEW)


project(learning_to_fly_in_seconds)

set(CMAKE_CXX_STANDARD 17)

find_package(HDF5)

if(HDF5_FOUND)
    message(STATUS "HDF5 found")
    set(BACKPROP_TOOLS_ENABLE_HDF5 ON)
endif()

set(TENSORBOARD_CONFIG_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/backprop_tools/external/tensorboard/CMakeLists.txt")
if(EXISTS "${TENSORBOARD_CONFIG_PATH}")
    message(STATUS "Tensorboard found")
    set(BACKPROP_TOOLS_ENABLE_TENSORBOARD ON)
endif()


set(JSON_CONFIG_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/backprop_tools/external/json/CMakeLists.txt")
if(EXISTS "${JSON_CONFIG_PATH}")
    set(BACKPROP_TOOLS_ENABLE_JSON ON)
endif()

find_package(Boost)
message(STATUS "Boost version: ${Boost_VERSION}")
if(Boost_FOUND)
    message(STATUS "Boost found")
    set(BACKPROP_TOOLS_ENABLE_BOOST_BEAST ON)
endif()

if(APPLE)
    set(BACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE ON)
endif()

add_subdirectory(external/backprop_tools)
add_subdirectory(src)