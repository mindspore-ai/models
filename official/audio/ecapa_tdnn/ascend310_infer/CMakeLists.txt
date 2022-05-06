cmake_minimum_required(VERSION 3.14.1)
project(310infer)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fPIE -Wl,--allow-shlib-undefined")

option(MINDSPORE_PATH "mindspore install path" "")
include_directories(${MINDSPORE_PATH})
include_directories(${MINDSPORE_PATH}/include)
find_library(MS_LIB libmindspore.so ${MINDSPORE_PATH}/lib)
file(GLOB_RECURSE MD_LIB ${MINDSPORE_PATH}/_c_dataengine*)

add_executable(ecapa_sample main.cc)
target_link_libraries(ecapa_sample ${MS_LIB} ${MD_LIB})