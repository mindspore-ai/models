cmake_minimum_required(VERSION 3.5.2)
project(resnet18)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
add_definitions(-DENABLE_DVPP_INTERFACE)
set(TARGET resnet)
set(ACL_LIB_PATH $ENV{ASCEND_HOME}/nnrt/latest/acllib)
set(ACL_INC_DIR $ENV{ASCEND_HOME}/$ENV{ASCEND_VERSION}/$ENV{ARCH_PATTERN}/acllib/include)

set(MXBASE_ROOT_DIR $ENV{MX_SDK_HOME})
set(MXBASE_INC ${MXBASE_ROOT_DIR}/src/include)
set(MXBASE_LIB_DIR ${MXBASE_ROOT_DIR}/dist/lib)
set(MXBASE_POST_LIB_DIR ${MXBASE_ROOT_DIR}/dist/lib/modelpostprocessors)
set(MXBASE_POST_PROCESS_DIR ${MXBASE_ROOT_DIR}/postprocess/include)
if(DEFINED ENV{MXSDK_OPENSOURCE_DIR})
    set(OPENSOURCE_DIR $ENV{MXSDK_OPENSOURCE_DIR})
else()
    set(OPENSOURCE_DIR ${MXBASE_ROOT_DIR}/opensource/dist)
endif()

include_directories(${ACL_INC_DIR})
include_directories(${OPENSOURCE_DIR}/include)
include_directories(${OPENSOURCE_DIR}/include/opencv4)

include_directories(${MXBASE_INC})
include_directories(${MXBASE_POST_PROCESS_DIR})

link_directories(${ACL_LIB_DIR})
link_directories(${OPENSOURCE_DIR}/lib)
link_directories(${MXBASE_LIB_DIR})
link_directories(${MXBASE_POST_LIB_DIR})


include_directories(${CMAKE_CURRENT_BINARY_DIR})

include_directories($ENV{MX_SDK_HOME}/include)
include_directories($ENV{MX_SDK_HOME}/opensource/include)
include_directories($ENV{MX_SDK_HOME}/opensource/include/opencv4)
include_directories($ENV{MX_SDK_HOME}/opensource/include/gstreamer-1.0)
include_directories($ENV{MX_SDK_HOME}/opensource/include/opencv4)

include_directories($ENV{MX_SDK_HOME}/opensource/include)
include_directories("/usr/local/Ascend/nnrt/5.0.1/x86_64-linux/acllib/include/")
include_directories($ENV{MX_SDK_HOME}/include/MxBase/postprocess/include)
link_directories($ENV{MX_SDK_HOME}/lib/)
link_directories($ENV{MX_SDK_HOME}/opensource/lib/)
link_directories($ENV{MX_SDK_HOME}/lib/modelpostprocessors)

add_compile_options(-std=c++11 -fPIC -fstack-protector-all -pie -Wno-deprecated-declarations)
add_compile_options("-DPLUGIN_NAME=${PLUGIN_NAME}")
add_compile_options("-Dgoogle=mindxsdk_private")

add_definitions(-DENABLE_DVPP_INTERFACE)

include_directories(${ACL_LIB_PATH}/include)
link_directories(${ACL_LIB_PATH}/lib64/)

message("TARGET_LIBRARY:${TARGET_LIBRARY}.")


add_executable(${TARGET} main_opencv.cpp FaceAttribute.cpp)
target_link_libraries(${TARGET} glog  cpprest mxbase libascendcl.so resnet50postprocess opencv_world)

