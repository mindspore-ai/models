cmake_minimum_required(VERSION 3.10)
project(retinanet)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

include_directories(retinanetDetection)
file(GLOB_RECURSE RETINANET_DETECTION ${PROJECT_SOURCE_DIR}/retinanetDetection/*cpp)
add_compile_options(-std=c++11 -fPIE -fstack-protector-all -fPIC -Wl,-z,relro,-z,now,-z,noexecstack -s -pie -Wall)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0 -Dgoogle=mindxsdk_private)

set(MX_SDK_HOME $ENV{MX_SDK_HOME})
set(DEBUG cmake-build-debug)
set(RELEASE release)
set(OUTPUT_NAME "retinanet")

include_directories(
        ${MX_SDK_HOME}/include
        ${MX_SDK_HOME}/opensource/include
        ${MX_SDK_HOME}/opensource/include/opencv4
)

link_directories(
        ${MX_SDK_HOME}/lib
        ${MX_SDK_HOME}/opensource/lib
        ${MX_SDK_HOME}/lib/modelpostprocessors
        /usr/local/Ascend/ascend-toolkit/latest/acllib/lib64
        /usr/local/Ascend/driver/lib64/
)

add_executable(${OUTPUT_NAME} main.cpp retinanetDetection/RetinanetDetection.cpp)
target_link_libraries(${OUTPUT_NAME}
        glog
        mxbase
        cpprest
        opencv_world
        )

install(TARGETS ${TARGET}
        CONFIGURATIONS ${DEBUG}
        RUNTIME DESTINATION ${DEBUG}/bin)
install(TARGETS ${TARGET}
        CONFIGURATIONS ${RELEASE}
        RUNTIME DESTINATION ${RELEASE}/bin)
