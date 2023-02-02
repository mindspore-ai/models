cmake_minimum_required(VERSION 3.14.1)
project(Ascend310Infer)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g -std=c++17 -Werror -Wall -fPIE -Wl,--allow-shlib-undefined")
set(PROJECT_SRC_ROOT ${CMAKE_CURRENT_LIST_DIR}/)
option(MINDSPORE_PATH "mindspore install path" "")
include_directories(${MINDSPORE_PATH})
include_directories(${MINDSPORE_PATH}/include)
include_directories(${PROJECT_SRC_ROOT})
set(TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)
include_directories(${TOP_DIR}/utils/cpp_infer/example/)  # common_inc in top dir
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)  # common_inc in local dir
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../)  # common_inc in local dir

if(EXISTS ${MINDSPORE_PATH}/lib/libmindspore-lite.so)
    message(--------------- Compile-with-MindSpore-Lite ----------------)
    set(MS_LIB ${MINDSPORE_PATH}/lib/libmindspore-lite.so)
    set(MD_LIB ${MINDSPORE_PATH}/lib/libminddata-lite.so)
    add_compile_definitions(ENABLE_LITE)
else()
    message(--------------- Compile-with-MindSpore ----------------)
    set(MS_LIB ${MINDSPORE_PATH}/lib/libmindspore.so)
    file(GLOB_RECURSE MD_LIB ${MINDSPORE_PATH}/_c_dataengine*)
endif()

add_executable(main src/main.cc)
target_link_libraries(main ${MS_LIB} ${MD_LIB})
