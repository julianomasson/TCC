#pragma once

#include <iostream>
#include <fstream>
#include <opencv2\opencv.hpp>
#include <vector>
#include <opencv2\xfeatures2d.hpp>
#include <opencv2\features2d.hpp>
#include "Utils.h"
#include "SiftGPU.h"

class Camera
{
public:
  Camera();
  Camera(std::string pathImage);
  ~Camera();

  std::string pathImage;
  cv::Mat img;
  cv::Mat K;
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  cv::Matx34f P = cv::Matx34f::eye();

  std::vector<float> descriptorsSiftGPU;
  std::vector<SiftGPU::SiftKeypoint> keypointsSiftGPU;

  std::string getStringSFM();
  std::string getStringNVM();

  void updateOrigin(cv::Matx34f PtoOrigin);
  void computeSiftGPUKeypoints();
  void computeOpenCVKeypoints(bool useAKAZE);
  void drawKeypoints(); 
  void getViewVector(double* point);
  double getAngleBetweenCameras(Camera* cam);
  double distanceBetweenOpticalCenter(Camera* cam);

};