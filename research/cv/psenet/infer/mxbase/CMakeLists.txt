cmake_minimum_required(VERSION 3.5.2)
project(yolov4tinypost)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

set(TARGET_MAIN psenet)
SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
set(ACL_LIB_PATH $ENV{ASCEND_HOME}/nnrt/latest/acllib)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

include_directories($ENV{MX_SDK_HOME}/include)
include_directories($ENV{MX_SDK_HOME}/opensource/include)
include_directories($ENV{MX_SDK_HOME}/opensource/include/opencv4)
include_directories($ENV{MX_SDK_HOME}/opensource/include/gstreamer-1.0)
include_directories($ENV{MX_SDK_HOME}/opensource/include/glib-2.0)
include_directories($ENV{MX_SDK_HOME}/opensource/lib/glib-2.0/include)
include_directories($ENV{MX_SDK_HOME}/include/MxBase/postprocess/include)
link_directories($ENV{MX_SDK_HOME}/lib)
link_directories($ENV{MX_SDK_HOME}/opensource/lib/)
link_directories($ENV{MX_SDK_HOME}/lib/modelpostprocessors)

add_compile_options(-std=c++11 -fPIC -fstack-protector-all -pie -Wno-deprecated-declarations)
add_compile_options("-DPLUGIN_NAME=${PLUGIN_NAME}")
add_compile_options("-Dgoogle=mindxsdk_private")

add_definitions(-DENABLE_DVPP_INTERFACE)

include_directories(${ACL_LIB_PATH}/include)
link_directories(${ACL_LIB_PATH}/lib64/)

add_executable(${TARGET_MAIN} src/main.cpp src/PsenetDetection.cpp)
target_link_libraries(${TARGET_MAIN} glib-2.0 gstreamer-1.0 gobject-2.0 gstbase-1.0 gmodule-2.0)
target_link_libraries(${TARGET_MAIN} plugintoolkit mxpidatatype mxbase)
target_link_libraries(${TARGET_MAIN} -Wl,-z,relro,-z,now,-z,noexecstack -s)
target_link_libraries(${TARGET_MAIN} glog cpprest mxbase psenetpostprocess stdc++fs)

target_link_libraries(${TARGET_MAIN}  glog  cpprest mxbase libascendcl.so  opencv_world)


message("TARGET_LIBRARY:${TARGET_LIBRARY}.")