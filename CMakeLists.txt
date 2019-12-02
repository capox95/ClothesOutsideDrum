cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set (CMAKE_CXX_STANDARD 11)

project(Outsize)

find_package(Boost REQUIRED)

find_package(PCL 1.9 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})



add_executable (WashingMachine washing_machine.cpp findTarget.cpp entropyOut.cpp pointposeOut.cpp alignTarget.cpp) 
target_link_libraries (WashingMachine ${PCL_LIBRARIES})
target_link_libraries(WashingMachine ${Boost_LIBRARIES})

add_executable (Bin bin.cpp binsegmentation.cpp entropyOut.cpp pointposeOut.cpp) 
target_link_libraries (Bin ${PCL_LIBRARIES})
target_link_libraries(Bin ${Boost_LIBRARIES})

add_executable (Model model.cpp) 
target_link_libraries (Model ${PCL_LIBRARIES})
target_link_libraries(Model ${Boost_LIBRARIES})





























