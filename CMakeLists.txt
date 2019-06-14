cmake_minimum_required(VERSION 3.14.0)
project (TCC)

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
				src/Plane.cpp
				src/Plane.h
				src/Seed.cpp
				src/Seed.h
				src/Track.cpp
				src/Track.h
				src/Utils.h)