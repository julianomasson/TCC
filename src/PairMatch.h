#pragma once

#include <iostream>
#include <fstream>
#include <opencv2\opencv.hpp>
#include <vector>
#include "Camera.h"
#include "Pair.h"


class PairMatch
{
public:
  PairMatch();
  ~PairMatch();

  Camera* cam;
  std::vector<Pair*> pairs;
  
};