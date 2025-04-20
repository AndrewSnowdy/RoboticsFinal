cmake_minimum_required(VERSION 3.10)
project(mesa_cpp VERSION 0.1.0)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find GTSAM
find_package(GTSAM REQUIRED)

# Include project headers
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/mesa_cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/mesa_cpp/src
)

# Gather all source files (excluding main.cpp for now)
file(GLOB_RECURSE MESA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/mesa_cpp/src/*.cpp")

# Remove main.cpp from the shared source list to avoid double inclusion
list(FILTER MESA_SOURCES EXCLUDE REGEX ".*main.cpp")

# Build main executable using main.cpp and all other sources
add_executable(mesa_exec
    ${MESA_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/mesa_cpp/src/main.cpp
)
target_link_libraries(mesa_exec gtsam)
