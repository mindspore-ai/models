cmake_minimum_required(VERSION 3.14.0)
project(pix2pix)
set(TARGET_MAIN Pix2Pix)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
add_definitions(-Dgoogle=mindxsdk_private)
add_compile_options(-std=c++11 -fPIE -fstack-protector-all -fPIC -Wall)
add_link_options(-Wl,-z,relro,-z,now,-z,noexecstack -s -pie)
# Check environment variable
if(NOT DEFINED ENV{ASCEND_HOME})
    message(FATAL_ERROR "please define environment variable:ASCEND_HOME")
endif()
# check env
if(NOT DEFINED ENV{ARCH_PATTERN})
    message(WARNING "please define environment variable:ARCH_PATTERN")
endif()

set(ACL_LIB_PATH $ENV{ASCEND_HOME}/nnrt/latest/acllib)
set(MXBASE_ROOT_DIR $ENV{MX_SDK_HOME})
set(MXBASE_INC ${MXBASE_ROOT_DIR}/include)
set(MXBASE_LIB_DIR ${MXBASE_ROOT_DIR}/lib)
set(MXBASE_POST_LIB_DIR ${MXBASE_ROOT_DIR}/lib/modelpostprocessors)
set(MXBASE_POST_PROCESS_DIR ${MXBASE_ROOT_DIR}/include/MxBase/postprocess/include)

if(NOT DEFINED ENV{MXSDK_OPENSOURCE_DIR})
    message(WARNING "please define environment variable:MXSDK_OPENSOURCE_DIR")
endif()

set(OPENSOURCE_DIR $ENV{MXSDK_OPENSOURCE_DIR})

include_directories(src)
include_directories(${ACL_INC_DIR})
include_directories(${OPENSOURCE_DIR}/include)
include_directories(${OPENSOURCE_DIR}/include/opencv4)


include_directories(${MXBASE_INC})
include_directories(${MXBASE_POST_PROCESS_DIR})

link_directories(${ACL_LIB_DIR})
link_directories(${OPENSOURCE_DIR}/lib)
link_directories(${MXBASE_LIB_DIR})
link_directories(${MXBASE_POST_LIB_DIR})


add_executable(${TARGET_MAIN} src/main.cpp src/Pix2Pix.cpp)

target_link_libraries(${TARGET_MAIN} glog cpprest mxbase opencv_world)
install(TARGETS ${TARGET_MAIN} RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/)
