cmake_minimum_required(VERSION 3.9.1)
project (nn)

set(PROJECT_SOURCE_DIR  "src")
set(PROJECT_BINARY_DIR  "bin")
set(BINARY_NAME "nn")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake_modules")

get_filename_component(FINAL_BIN "${CMAKE_SOURCE_DIR}/bin" ABSOLUTE)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${FINAL_BIN})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${FINAL_BIN})

include_directories(${SFML_ROOT}/include include)
link_directories(${SFML_ROOT}/lib)

file(GLOB_RECURSE BINARY_SRC "${PROJECT_SOURCE_DIR}/*.cpp")

add_executable(${BINARY_NAME} ${BINARY_SRC})

# Set default build type
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)

# Select flags.
set(CMAKE_CXX_FLAGS "-Wall -Wno-switch -Wno-long-long -Wno-unknown-pragmas -Wno-narrowing -std=c++17 -pthread -pedantic")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG  "-Og -g -ggdb")
set(CMAKE_CXX_FLAGS_PROFILE  "-O0 -pg -g -ggdb")

find_package(SFML COMPONENTS graphics window system)
target_link_libraries(${BINARY_NAME} ${SFML_LIBRARIES} ${SFML_DEPENDENCIES})
