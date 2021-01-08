cmake_minimum_required(VERSION 3.10)
project(webrtc_test)

set(CMAKE_CXX_STANDARD 17)
link_directories("/usr/local/lib")
include_directories("/usr/local/include")
include_directories("/home/pi/Projects/webrtc-audio-processing")

add_executable(webrtc_test webrtc_test.cpp)

SET(CMAKE_CXX_FLAGS "-DASIO_STANDALONE -Wall -O3")
target_link_libraries(webrtc_test /home/pi/Projects/webrtc-audio-processing/build/webrtc/modules/audio_processing/libwebrtc_audio_processing.so /home/pi/Projects/webrtc-audio-processing/build/webrtc/libwebrtc.a)