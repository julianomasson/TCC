#pragma once

#include <iostream>
#include <fstream>
#include <opencv2\opencv.hpp>
#include <vector>
#include "Keypoint.h"

class Seed
{
public:
  Seed();
  Seed(cv::Point2f p1, cv::Point2f p2, Camera* cam1, Camera* cam2, double score = 0);
  ~Seed();

  cv::Point3f getPoint3D();
  void calculatePoint3D(cv::Mat newH1, cv::Mat newH2);
  double getReprojectionError(cv::Mat point3D, Camera* cam, cv::Point2f p);


  cv::Point2f p1 = cv::Point2f(-1, -1);
  cv::Point2f p2 = cv::Point2f(-1, -1);
  cv::Point2f p1Rect = cv::Point2f(-1, -1);
  cv::Point2f p2Rect = cv::Point2f(-1, -1);
  double reprojError = VTK_DOUBLE_MAX;
  double score;


  cv::Vec3f ptNormal = cv::Vec3f(0, 0, 0);
private:
  cv::Point3f pt3D = cv::Point3f(-1, -1, -1);
  Camera* cam1 = NULL;
  Camera* cam2 = NULL;
};