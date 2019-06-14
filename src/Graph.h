#pragma once
#include <iostream>
#include <fstream>
#include <opencv2\opencv.hpp>
#include <vector>
#include "Keypoint.h"
#include "Pair.h"
#include "Track.h"
#include "PairMatch.h"

class Graph
{
public:
  Graph();
  Graph(PairMatch* p);
  Graph(Pair* p);
  ~Graph();

  void mergeGraph(Graph* g);
  void addPair(Pair* p);
  void addPairMatch(PairMatch* p);
  void calculate3DPoints();
  void saveSFM(std::string path);
  void saveNVM(std::string path);
  void savePointCloud(std::string path);
  size_t getSizeModel();
  void filterPoints();
  void mergeTracks(std::vector<Track*> newTracks);
  void addTracks(std::vector<Track*> newTracks);

  std::vector<Track*> tracks;
  std::vector<Camera*> cameras;
  std::vector<PairMatch*> pairMatches;
};