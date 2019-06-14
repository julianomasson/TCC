#pragma once

#include <iostream>
#include <fstream>
#include <opencv2\opencv.hpp>
#include <vector>
#include "Camera.h"
#include "SiftGPU.h"

class Keypoint
{
public:
  Keypoint();
  Keypoint(cv::KeyPoint keypoint, Camera* cam);
  Keypoint(SiftGPU::SiftKeypoint keypoint, Camera* cam);
  ~Keypoint();

  bool isEqual(Keypoint* k);
  double getReprojectionError(cv::Mat point3D);
  cv::Point2f getPoint();

  Camera* cam = NULL;
  bool isActive = true;
  private:
    bool useSiftGPU = false;
    cv::KeyPoint keypoint;
    SiftGPU::SiftKeypoint keypointSiftGPU;
    cv::Point2f pt = cv::Point2f(-1, -1);

};