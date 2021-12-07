cmake_minimum_required(VERSION 3.10.0)
project(ssd_ms)

set(TARGET ssd_resnet50)

SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

add_definitions(-DENABLE_DVPP_INTERFACE)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
add_definitions(-Dgoogle=mindxsdk_private)
add_compile_options(-std=c++11 -fPIE -fstack-protector-all -fPIC -Wall)
add_link_options(-Wl,-z,relro,-z,now,-z,noexecstack -pie)

# Check environment variable
if(NOT DEFINED ENV{ASCEND_HOME})
    message(FATAL_ERROR "please define environment variable:ASCEND_HOME")
endif()
if(NOT DEFINED ENV{ASCEND_VERSION})
    message(WARNING "please define environment variable:ASCEND_VERSION")
endif()
if(NOT DEFINED ENV{ARCH_PATTERN})
    message(WARNING "please define environment variable:ARCH_PATTERN")
endif()

# 设置acllib的头文件和动态链接库
set(ACL_INC_DIR $ENV{ASCEND_HOME}/${ASCEND_VERSION}/${ARCH_PATTERN}/acllib/include)
set(ACL_LIB_DIR $ENV{ASCEND_HOME}/${ASCEND_VERSION}/${ARCH_PATTERN}/acllib/lib64)

# 设置MxBase的头文件和动态链接库
set(MXBASE_ROOT_DIR $ENV{MX_SDK_HOME})
set(MXBASE_INC ${MXBASE_ROOT_DIR}/include)
set(MXBASE_LIB_DIR ${MXBASE_ROOT_DIR}/lib)
set(MXBASE_POST_LIB_DIR ${MXBASE_ROOT_DIR}/lib/modelpostprocessors)
set(MXBASE_POST_PROCESS_DIR ${MXBASE_ROOT_DIR}/include/MxBase/postprocess/include)

# 设置opensource的头文件和动态链接库
# 主要包含OpenCV、Google log等开源库
set(OPENSOURCE_DIR ${MXBASE_ROOT_DIR}/opensource)

include_directories(${ACL_INC_DIR})
include_directories(${OPENSOURCE_DIR}/include)
include_directories(${OPENSOURCE_DIR}/include/opencv4)

include_directories(${MXBASE_INC})
include_directories(${MXBASE_POST_PROCESS_DIR})
link_directories(${ACL_LIB_DIR})
link_directories(${OPENSOURCE_DIR}/lib)
link_directories(${MXBASE_LIB_DIR})
link_directories(${MXBASE_POST_LIB_DIR})

# 本地编译链接文件，根据mxbase目录下文件添加修改
add_executable(${TARGET} ResNet50_main.cpp ResNet50Base.cpp)

# 链接动态链接库，后处理lib:yolov3postprocess根据实际情况修改
target_link_libraries(${TARGET} glog cpprest mxbase SsdMobilenetFpn_MindsporePost opencv_world stdc++fs)

install(TARGETS ${TARGET} RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/)