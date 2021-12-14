cmake_minimum_required(VERSION 3.5.2)
project(ssdghostpost)

set(TARGET ssd_ghost)

#set(PLUGIN_NAME "ssdghost_mindspore_post")
set(TARGET_LIBRARY ssdghost_mindspore_post)
set(TARGET_MAIN ssdghost_mindspore)

SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -std=c++0x")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

add_definitions(-DENABLE_DVPP_INTERFACE)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
add_definitions(-Dgoogle=mindxsdk_private)
add_compile_options(-std=c++11 -fPIE -fstack-protector-all -fPIC -Wall)
add_compile_options("-DPLUGIN_NAME=${PLUGIN_NAME}")
add_compile_options("-Dgoogle=mindxsdk_private")
# set(CMAKE_SHARED_LINKER_FLAGS "Wl,-z,relro,-z,now,-z,noexecstack -pie")

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

# Set up ACLLIB header files and dynamic link libraries
set(ACL_INC_DIR $ENV{ASCEND_HOME}/${ASCEND_VERSION}/${ARCH_PATTERN}/acllib/include)
set(ACL_LIB_DIR $ENV{ASCEND_HOME}/${ASCEND_VERSION}/${ARCH_PATTERN}/acllib/lib64)
set(ACL_LIB_PATH $ENV{ASCEND_HOME}/nnrt/latest/acllib)

# Set the header file and dynamic link library of MXBase
set(MXBASE_ROOT_DIR $ENV{MX_SDK_HOME})
set(MXBASE_INC ${MXBASE_ROOT_DIR}/include)
set(MXBASE_LIB_DIR ${MXBASE_ROOT_DIR}/lib)
set(MXBASE_POST_LIB_DIR ${MXBASE_ROOT_DIR}/lib/modelpostprocessors)
# set(MXBASE_POST_PROCESS_DIR ${MXBASE_ROOT_DIR}/include/MxBase/postprocess/include)

# Set the header file and dynamic link library of OpenSource
# Mainly includes open source libraries such as OpenCV, Google Log
set(OPENSOURCE_DIR ${MXBASE_ROOT_DIR}/opensource)

include_directories(${MXBASE_ROOT_DIR}/opensource/include)
include_directories(${MXBASE_ROOT_DIR}/opensource/include/opencv4)
include_directories(${MXBASE_ROOT_DIR}/opensource/include/gstreamer-1.0)
include_directories(${MXBASE_ROOT_DIR}/opensource/include/glib-2.0)
include_directories(${MXBASE_ROOT_DIR}/opensource/lib/glib-2.0/include)

include_directories(${ACL_LIB_PATH}/include)
link_directories(${ACL_LIB_PATH}/lib64/)

link_directories(${MXBASE_ROOT_DIR}/lib)
link_directories(${MXBASE_ROOT_DIR}/opensource/lib/)

include_directories(${ACL_INC_DIR})
include_directories(${OPENSOURCE_DIR}/include)
include_directories(${OPENSOURCE_DIR}/include/opencv4)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

include_directories(${MXBASE_INC})
# include_directories(${MXBASE_POST_PROCESS_DIR})
link_directories(${ACL_LIB_DIR})
link_directories(${OPENSOURCE_DIR}/lib)
link_directories(${MXBASE_LIB_DIR})
link_directories(${MXBASE_POST_LIB_DIR})

add_library(${TARGET_LIBRARY} SHARED PostProcess/SsdGhost_MindsporePost.cpp)
target_link_libraries(${TARGET_LIBRARY} glib-2.0 gstreamer-1.0 gobject-2.0 gstbase-1.0 gmodule-2.0)
target_link_libraries(${TARGET_LIBRARY} plugintoolkit mxpidatatype mxbase)
target_link_libraries(${TARGET_LIBRARY} -Wl,-z,relro,-z,now,-z,noexecstack)

# Locally compiled link files, add changes to files in the MXBase directory
add_executable(${TARGET} main_ssd.cpp SSDGhost.cpp)

target_link_libraries(${TARGET} ${TARGET_LIBRARY} glog cpprest mxbase opencv_world stdc++fs)

install(TARGETS ${TARGET} RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/)