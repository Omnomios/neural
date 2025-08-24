cmake_minimum_required(VERSION 3.28)
project (nn)

set(PROJECT_SOURCE_DIR  "src")
set(PROJECT_BINARY_DIR  "bin")
set(BINARY_NAME "nn")

get_filename_component(FINAL_BIN "${CMAKE_SOURCE_DIR}/bin" ABSOLUTE)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${FINAL_BIN})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${FINAL_BIN})

file(GLOB_RECURSE BINARY_SRC "${PROJECT_SOURCE_DIR}/*.cpp")

add_executable(${BINARY_NAME} ${BINARY_SRC})

# Set default build type
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)

if(WIN32)
   set(SFML_STATIC_LIBRARIES TRUE)
   set(WINDOWS_FLAG "-mwindows")
endif()

# Select flags.
set(CMAKE_CXX_FLAGS "-Wall -Wno-switch -Wno-long-long -Wno-unknown-pragmas -Wno-narrowing -pthread -pedantic ${WINDOWS_FLAG}")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG  "-Og -g -ggdb")
set(CMAKE_CXX_FLAGS_PROFILE  "-O0 -pg -g -ggdb")

# Ensure at least C++17 for SFML 3
target_compile_features(${BINARY_NAME} PRIVATE cxx_std_17)

# FetchContent SFML 3.0.1 per official guide
include(FetchContent)
FetchContent_Declare(SFML
    GIT_REPOSITORY https://github.com/SFML/SFML.git
    GIT_TAG 3.0.1
    GIT_SHALLOW ON
    EXCLUDE_FROM_ALL
    SYSTEM)
FetchContent_MakeAvailable(SFML)

# Include our own headers
include_directories(include)

# Link against SFML 3 namespaced target
target_link_libraries(${BINARY_NAME} PRIVATE SFML::Graphics)
