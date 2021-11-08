cmake_minimum_required(VERSION 3.5.2)
project(mxplugins)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
add_definitions(-Dgoogle=mindxsdk_private)
set(PLUGIN_NAME "mxpi_transposeplugin_2")
set(TARGET_LIBRARY ${PLUGIN_NAME})

set(MXBASE_DIR $ENV{MX_SDK_HOME})
set(OPENSOURCE_DIR $ENV{MX_SDK_HOME}/opensource)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${OPENSOURCE_DIR}/../../include)
include_directories(${OPENSOURCE_DIR}/include)
include_directories(${OPENSOURCE_DIR}/include/gstreamer-1.0)
include_directories(${OPENSOURCE_DIR}/include/glib-2.0)
include_directories(${OPENSOURCE_DIR}/lib/glib-2.0/include)
include_directories(${MXBASE_DIR}/include)

link_directories(${MXBASE_DIR}/lib)
link_directories(${OPENSOURCE_DIR}/lib)

add_compile_options(-std=c++11 -fPIC -fstack-protector-all -pie -Wno-deprecated-declarations)
add_compile_options("-DPLUGIN_NAME=${PLUGIN_NAME}")

add_definitions(-DENABLE_DVPP_INTERFACE)

add_library(${TARGET_LIBRARY} SHARED MxpiTransposePlugin.cpp)

target_link_libraries(${TARGET_LIBRARY} glib-2.0 gstreamer-1.0 gobject-2.0 gstbase-1.0 gmodule-2.0)

target_link_libraries(${TARGET_LIBRARY} mxpidatatype plugintoolkit mxbase mindxsdk_protobuf)