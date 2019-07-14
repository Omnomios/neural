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

set(MAKE_FLAGS "${CMAKE_CXX_FLAGS} -g -ggdb -O2 -Wall -Wno-long-long -Wno-unknown-pragmas -std=c++17 -pthread -pedantic")

#-finstrument-functions
set(CMAKE_CXX_FLAGS "${MAKE_FLAGS}")

find_package(SFML COMPONENTS graphics window system)
target_link_libraries(${BINARY_NAME} ${SFML_LIBRARIES} ${SFML_DEPENDENCIES})
