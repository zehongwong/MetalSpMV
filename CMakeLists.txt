cmake_minimum_required(VERSION 3.8)
project(MetalSpmv)

set(CMAKE_CXX_STANDARD 17)

# Download metal-cpp from Apple
# Link: https://developer.apple.com/metal/cpp/
include_directories(${GSL_INCLUDE_DIRS})
include_directories(include)
include_directories(include/metal-cpp)

# Use llvm version of clang to support OpenMP
# brew install llvm
# brew install libomp
set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")

# Set -O3 optimization flag for C++ compiler
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

set(SOURCE_FILES src/main.cpp src/spmv_calculator.cpp src/logger.cpp)
add_executable(spmv ${SOURCE_FILES})

find_library(METAL Metal)
find_library(FOUNDATION Foundation)
find_library(QUARTZCORE QuartzCore)
find_package(GSL REQUIRED)

target_link_libraries(spmv -fopenmp GSL::gsl GSL::gslcblas
        "-framework Metal" "-framework Foundation" "-framework QuartzCore")

add_custom_target(kernel
        COMMAND xcrun -sdk macosx metal -O0 -c src/spmv.metal -o spmv.air
        COMMAND xcrun -sdk macosx metallib spmv.air -o spmv.metallib
        DEPENDS src/spmv.metal
        )