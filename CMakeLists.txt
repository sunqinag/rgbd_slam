cmake_minimum_required(VERSION 3.17)
project(RGBD_SLAM)

set(CMAKE_CXX_STANDARD 14)

include_directories(${PROJECT_SOURCE_DIR}/include)

# 增加PCL库的依赖
FIND_PACKAGE( PCL REQUIRED COMPONENTS common io visualization)

# 增加opencv的依赖
FIND_PACKAGE( OpenCV REQUIRED )

# 添加头文件和库文件
ADD_DEFINITIONS( ${PCL_DEFINITIONS} )
INCLUDE_DIRECTORIES( ${PCL_INCLUDE_DIRS}  )
LINK_LIBRARIES( ${PCL_LIBRARY_DIRS} )

add_library(${PROJECT_NAME} SHARED
#        src/generatePointCloud.cpp
        src/slamBase.cpp
        src/detectFeatures.cpp src/joinPointCloud.cpp)


target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES})

add_executable(generatePointCloud src/generatePointCloud.cpp)
target_link_libraries(generatePointCloud ${PROJECT_NAME})

add_executable(slamBase src/slamBase.cpp)
target_link_libraries(slamBase ${PROJECT_NAME})

 add_executable(detectFeatures src/detectFeatures.cpp)
 target_link_libraries(detectFeatures ${PROJECT_NAME})