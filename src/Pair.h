#pragma once

#include <iostream>
#include <fstream>
#include <opencv2\opencv.hpp>
#include <vector>
#include <opencv2\xfeatures2d.hpp>
#include <opencv2\features2d.hpp>
#include "Camera.h"
#include "SiftGPU.h"
#include "Track.h"
#include <mutex>
#include "Seed.h"
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/io/ply_io.h>
#include <pcl/surface/mls.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <Windows.h>



class Pair
{
public:
  Pair();
  Pair(Camera* camLeft, Camera* camRight, cv::Mat K, float distanceNNDR = 0.75);
  ~Pair();

  void matchOpenCV();
  void matchSiftGPU();
  void computePose();
  void createTracks();

  //Dense
  void computeRectify();
  void computeDenseSIFT();
  void matchDenseSIFT();
  void computeDenseDAISY();
  void matchDenseDAISY();
  void createInitialSeeds();
  std::string computeNewSeeds();
  void compute3DPoints();
  void createDenseCloud(std::string path);
  void clearSeeds();
  void saveMasks(std::string path);
  void computeDepthMap();
  void saveCloudNormals(std::string path);

  //PCL
  void createPCLPointCloud();
  void filterPCLCloud();
  void computePCLNormal();
  void savePCLResult(std::string path);

  Camera* camLeft = NULL;
  Camera* camRight = NULL;
  bool pairMerged = false;
  int numMaxMatches = 4096;
  std::vector<Track*> tracks;

  std::vector<cv::KeyPoint> goodKeypointsLeft;
  std::vector<cv::KeyPoint> goodKeypointsRight;
  std::vector<SiftGPU::SiftKeypoint> goodKeypointsLeftSiftGPU;
  std::vector<SiftGPU::SiftKeypoint> goodKeypointsRightSiftGPU;
  std::vector<cv::Point2f> imgLeftPts;
  std::vector<cv::Point2f> imgRightPts;

  cv::Matx34f Pright;

  size_t drawMatches();
  std::vector<Seed*> seeds;
  cv::Mat newH1;
  cv::Mat newH2;

  //PCL
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPCL = NULL;
  pcl::PointCloud<pcl::Normal>::Ptr cloudPCLNormals = NULL;
  pcl::PointCloud<pcl::PointNormal>::Ptr cloudPlusNormalsPCL = NULL;
private:
  std::vector< cv::DMatch > good_matches;
  cv::Mat K, E;
  float distanceNNDR;

  //Dense
  cv::Mat img1Warp;
  cv::Mat img1Mask;
  
  std::vector<cv::KeyPoint> keypoints1;
  cv::Mat descriptors1;

  cv::Mat img2Warp;
  cv::Mat img2Mask;
  
  std::vector<cv::KeyPoint> keypoints2;
  cv::Mat descriptors2;
  
};