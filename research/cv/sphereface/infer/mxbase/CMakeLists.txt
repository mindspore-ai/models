cmake_minimum_required(VERSION 3.14.0)
project(sphereface)
set(TARGET sphereface)

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
set(ACL_INC_DIR $ENV{ASCEND_HOME}/$ENV{ASCEND_VERSION}/$ENV{ARCH_PATTERN}/acllib/include)
set(ACL_LIB_DIR $ENV{ASCEND_HOME}/$ENV{ASCEND_VERSION}/$ENV{ARCH_PATTERN}/acllib/lib64)

set(MXBASE_INC $ENV{MX_SDK_HOME}/include)
set(MXBASE_LIB_DIR $ENV{MX_SDK_HOME}/lib)
set(MXBASE_POST_LIB_DIR $ENV{MX_SDK_HOME}/lib/modelpostprocessors)
set(MXBASE_POST_PROCESS_DIR $ENV{MX_SDK_HOME}//include/MxBase/postprocess/include/)

if(DEFINED ENV{MXSDK_OPENSOURCE_DIR})
    set(OPENSOURCE_DIR $ENV{MXSDK_OPENSOURCE_DIR})
else()
    set(OPENSOURCE_DIR $ENV{MX_SDK_HOME}/opensource)
endif()

include_directories(${ACL_INC_DIR})
include_directories(${MXBASE_INC})
include_directories(${MXBASE_POST_PROCESS_DIR})
include_directories(${OPENSOURCE_DIR}/include)
include_directories(${OPENSOURCE_DIR}/include/opencv4)
message(WARNING "${OPENSOURCE_DIR}/include/opencv4/opencv2")


link_directories(${ACL_LIB_DIR})
link_directories(${MXBASE_LIB_DIR})
link_directories(${MXBASE_POST_LIB_DIR})
link_directories(${OPENSOURCE_DIR}/lib)

set(ACL_LIB_PATH $ENV{ASCEND_HOME}/nnrt/latest/acllib)
message(WARNING "ACL_LIB_PATH:${ACL_LIB_PATH}/lib64/.")
include_directories(${ACL_LIB_PATH}/include)
link_directories(${ACL_LIB_PATH}/lib64/)


add_executable(${TARGET} src/main_opencv.cpp src/SphereFaceOpencv.cpp)

target_link_libraries(${TARGET} glog cpprest mxbase resnet50postprocess opencv_world)

install(TARGETS ${TARGET} RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/)
