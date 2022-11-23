cmake_minimum_required(VERSION 3.10.2)
project(FeatureExtraction)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -std=c++17 -Werror -Wall -fPIE -Wl,--allow-shlib-undefined -lm")

add_executable(lpcnet_demo
    src/celt_lpc.c
    src/ceps_codebooks.c
    src/common.c
    src/freq.c
    src/kiss_fft.c
    src/lpcnet_demo.c
    src/lpcnet_enc.c
    src/pitch.c)
add_executable(dump_data
    src/celt_lpc.c
    src/ceps_codebooks.c
    src/common.c
    src/dump_data.c
    src/freq.c
    src/kiss_fft.c
    src/lpcnet_enc.c
    src/lpcnet_dec.c
    src/pitch.c)
target_link_libraries(lpcnet_demo m)
target_link_libraries(dump_data m)
