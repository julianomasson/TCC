cmake_minimum_required(VERSION 3.14.0)
project (TCC)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_path(SIFTGPU_INCLUDE_DIR SiftGPU)
find_library(SIFTGPU_LIBRARY_RELEASE SiftGPU)
include_directories(${SIFTGPU_INCLUDE_DIR})

find_path(PBA_INCLUDE_DIR pba)
find_library(PBA_LIBRARY_RELEASE pba_x64)
include_directories(${PBA_INCLUDE_DIR})

add_executable(${PROJECT_NAME} src/main.cpp
				src/BundleAdjustment.cpp
				src/BundleAdjustment.h
				src/Camera.cpp
				src/Camera.h
				src/Graph.cpp
				src/Graph.h
				src/Keypoint.cpp
				src/Keypoint.h
				src/Pair.cpp
				src/Pair.h
				src/PairMatch.cpp
				src/PairMatch.h
				src/Seed.cpp
				src/Seed.h
				src/Track.cpp
				src/Track.h
				src/Utils.h)

# Find PCL
find_package( PCL 1.8 REQUIRED )
 
if( PCL_FOUND )
  # Additional Include Directories
  # [C/C++]>[General]>[Additional Include Directories]
  include_directories( ${PCL_INCLUDE_DIRS} )
 
  # Preprocessor Definitions
  # [C/C++]>[Preprocessor]>[Preprocessor Definitions]
  add_definitions( ${PCL_DEFINITIONS} )
  #add_definitions( -DPCL_NO_PRECOMPILE )
 
  # Additional Library Directories
  # [Linker]>[General]>[Additional Library Directories]
  link_directories( ${PCL_LIBRARY_DIRS} )
 
  # Additional Dependencies
  # [Linker]>[Input]>[Additional Dependencies]
  target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES} )
endif()


target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} )

target_link_libraries(${PROJECT_NAME} ${SIFTGPU_LIBRARY_RELEASE})

target_link_libraries(${PROJECT_NAME} ${PBA_LIBRARY_RELEASE})