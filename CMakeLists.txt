cmake_minimum_required(VERSION 3.17)
project(armadilo_study)

set(CMAKE_CXX_STANDARD 17)
include_directories(header)
link_libraries(armadillo)
#add_executable(ocsim main.cpp header/dsp.h source/dsp.cpp source/dsp.cpp)
add_executable(armadilo_study main.cpp )