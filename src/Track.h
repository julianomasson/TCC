#pragma once

#include <iostream>
#include <fstream>
#include <opencv2\opencv.hpp>
#include <vector>
#include "Keypoint.h"

#define REPROJ_ERROR 10.0

class Track
{
public:
  Track();
  ~Track();

  void addKeypoint(Keypoint* k);
  bool mergeTrack(Track* k);
   
  std::vector<Keypoint*> keypoints;
  cv::Point3f pt3D = cv::Point3f(-1,-1,-1);
  cv::Point3f getPoint3D();
  void calculatePoint3D();
  void testPoint();

  double lastReprojError = VTK_DOUBLE_MAX;

};