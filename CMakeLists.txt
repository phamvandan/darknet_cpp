cmake_minimum_required(VERSION 1.0.0)
project(test)
file(GLOB SOURCES "src/*.cpp")

add_executable(out ${SOURCES})

set(PROJECT_LINK_LIBS libdarkhelp.so libdarknet.so )
link_directories(/usr/lib)

find_package( OpenCV REQUIRED )

target_link_libraries( out ${OpenCV_LIBS} ${PROJECT_LINK_LIBS} )