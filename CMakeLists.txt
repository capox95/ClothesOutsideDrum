cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set (CMAKE_CXX_STANDARD 11)

project(Outsize)

find_package(Boost REQUIRED)

find_package(PCL 1.9 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


add_executable (Align align.cpp findTarget.cpp entropy.cpp pointpose.cpp alignProcessing.cpp) 
target_link_libraries (Align ${PCL_LIBRARIES})
target_link_libraries(Align ${Boost_LIBRARIES})

add_executable (Model model.cpp) 
target_link_libraries (Model ${PCL_LIBRARIES})
target_link_libraries(Model ${Boost_LIBRARIES})





























