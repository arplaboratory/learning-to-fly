
set(INSTALL_GTEST OFF CACHE BOOL "")
set(BUILD_GMOCK OFF CACHE BOOL "")
add_subdirectory(lib/googletest EXCLUDE_FROM_ALL)
include(GoogleTest)

add_library(rl_tools_tests INTERFACE)
target_link_libraries(rl_tools_tests INTERFACE GTest::gtest_main)
target_compile_definitions(rl_tools_tests INTERFACE RL_TOOLS_TESTS_DATA_PATH=${RL_TOOLS_TESTS_DATA_PATH})


    # Environment Multirotor test
add_executable(
        test_rl_environments_multirotor_multirotor
        multirotor.cpp
)
target_link_libraries(
        test_rl_environments_multirotor_multirotor
        rl_tools
        rl_tools_tests
        learning_to_fly
)
gtest_discover_tests(test_rl_environments_multirotor_multirotor)



# Multirotor UI test
if(${RL_TOOLS_TESTS_RL_ENVIRONMENTS_MULTIROTOR_UI_ENABLE})
    add_executable(
            test_rl_environments_multirotor_ui_test
            src/rl/environments/multirotor/ui.cpp
    )
    target_link_libraries(
            test_rl_environments_multirotor_ui_test
            rl_tools
            rl_tools_tests
            learning_to_fly
    )
    target_include_directories(
            test_rl_environments_multirotor_ui_test
            PRIVATE
            lib/nlohmann/include
    )
    RL_TOOLS_ADD_DEFINITIONS(test_rl_environments_multirotor_ui_test)
    gtest_discover_tests(test_rl_environments_multirotor_ui_test)
endif()


if(NOT APPLE)
#    add_subdirectory(regression_test_example_old)
endif()