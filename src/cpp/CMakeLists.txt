
cmake_minimum_required(VERSION 3.5)
project(hific_cpp)

add_subdirectory(pybind11)

pybind11_add_module(hific_cpp
    ans.cpp
)

