#pragma once
#include <iostream>
#include <fstream>
#include <opencv2\opencv.hpp>
#include <vector>
#include <DataInterface.h>
#include "Graph.h"
#include <pba.h>


class BundleAdjustment
{
public:
  BundleAdjustment();
  BundleAdjustment(Graph* g);
  ~BundleAdjustment();

  Graph* graph;

  std::vector<CameraT>        camera_data;    //camera (input/ouput)
  std::vector<Point3D>        point_data;     //3D point(iput/output)
  std::vector<Point2D>        measurements;   //measurment/projection vector
  std::vector<int>            camidx, ptidx;  //index of camera/point for each projection
  std::vector<std::string>    photo_names;        //from NVM file, not used in bundle adjustment
  std::vector<int>            point_color;        //from NVM file, not used in bundle adjustment


  void runBundle(bool fullBA);
  
  Graph* getResult();

  double error;
  double getError();

  void saveNVM(std::string filename);

};