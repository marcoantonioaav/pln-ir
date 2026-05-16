cmake_minimum_required(VERSION 3.14)
project(pln_ir_initialization)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# This assumes pybind11 is installed, e.g., via pip install pybind11
# For cmake to find it, you might need to run:
# cmake -DCMAKE_PREFIX_PATH=$(python3 -m pybind11 --includes) ..
find_package(pybind11 REQUIRED)

pybind11_add_module(initialization_cpp 
    src/cpp/initialization.cpp
    src/cpp/bindings.cpp
)

target_include_directories(initialization_cpp PRIVATE src/cpp/include)
