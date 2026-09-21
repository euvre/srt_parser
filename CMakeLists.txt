cmake_minimum_required(VERSION 4.2.0)
set(CMAKE_CXX_COMPILER g++)
project(srt_parser LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(srt_texts srt_texts.cpp)
target_compile_options(srt_texts PRIVATE -Wall -Wextra -Wpedantic
                       -march=native -freflection)
