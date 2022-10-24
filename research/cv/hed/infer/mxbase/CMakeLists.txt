cmake_minimum_required(VERSION 3.14.0)
project(hed)

set(TARGET hed)

add_definitions(-DENABLE_DVPP_INTERFACE)
add_link_options(-Wl,-z,relro,-z,now,-z,noexecstack -s -pie)

add_compile_options(-std=c++14 -fPIE -fstack-protector-all -fPIC -Wall
-Dgoogle=mindxsdk_private -D_GLIBCXX_USE_CXX11_ABI=0)


#Check environment variable
if(NOT DEFINED ENV{ASCEND_HOME})
    message(FATAL_ERROR "please define environment variable:ASCEND_HOME")
endif()
if(NOT DEFINED ENV{ASCEND_VERSION})
    message(WARNING "please define environment variable:ASCEND_VERSION")
endif()
if(NOT DEFINED ENV{ARCH_PATTERN})
    message(WARNING "please define environment variable:ARCH_PATTERN")
endif()
set(ACL_INC_DIR $ENV{ASCEND_HOME}/$ENV{ASCEND_VERSION}/acllib/include)
set(ACL_LIB_DIR $ENV{ASCEND_HOME}/$ENV{ASCEND_VERSION}/acllib/lib64)

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

add_executable(${TARGET} main.cpp HedEdgeDetection.cpp)
target_link_libraries(${TARGET} glog cpprest mxbase opencv_world stdc++fs)
install(TARGETS ${TARGET} RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/)
