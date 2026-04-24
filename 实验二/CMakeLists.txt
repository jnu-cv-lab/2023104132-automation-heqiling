cmake_minimum_required(VERSION 3.10)
project(OpenCV_Experiment2)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找OpenCV库
find_package(OpenCV REQUIRED)

# 包含头文件
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加可执行文件（main.cpp在根目录）
add_executable(${PROJECT_NAME} main.cpp)

# 链接OpenCV库
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})