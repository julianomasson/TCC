#pragma once

#include <iostream>
#include <fstream>
#include <opencv2\opencv.hpp>
#include <vector>
#include "Camera.h"

#define LAMBDA_MAX 200
#define LAMBDA_MIN 10
#define WINDOW_SIZE 17

class Plane
{
public:
  Plane(Camera* cam_i, Camera* cam_j, int x, int y);
  ~Plane();

  double m();

private:
  cv::Point2f pt2D;
  cv::Mat X;
  cv::Mat p;
  cv::Mat n;
  Camera* cam_i = NULL;
  Camera* cam_j = NULL;
  double lambda;
  double theta;
  double phi;
  double score = -1;
};