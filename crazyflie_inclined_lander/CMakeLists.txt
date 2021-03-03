cmake_minimum_required(VERSION 2.8.3)
project(crazyflie_inclined_lander)


find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        std_msgs
        geometry_msgs
        nav_msgs
        tf
        message_generation
        bebop2_msgs
        )


find_package(Eigen3 REQUIRED)

catkin_package(
        CATKIN_DEPENDS
        roscpp rospy
        std_msgs
        geometry_msgs
        message_runtime
        bebop2_msgs
)


include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        include/${PROJECT_NAME}
)

# !---------------] find libs what wavehands need
set(CMAKE_PREFIX_PATH "/home/jacob/libtorch")
set(Torch_DIR $ENV{/home/jacob/libtorch/share/cmake/Torch})
message(STATUS "++++++ torch find path" ${CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)
include_directories(include/)
if (NOT Torch_FOUND)
    message(FATAL_ERROR "you should change CMAKE_PREFIX_PATH to libtorch dir")
else ()
    message(STATUS "found libtorch at " ${TORCH_LIBRARIES})
endif ()

add_executable(inclined_landing_node nodes/inclined_landing_node.cpp src/inclined_landing.cpp)
target_link_libraries(inclined_landing_node
        ${catkin_LIBRARIES}
        ${TORCH_LIBRARIES})
## For debugging
#set (CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -g ")
#set (CMAKE_VERBOSE_MAKEFILE ON)
