#include "stdafx.h"
#include "Graph.h"

Graph::Graph()
{
}

Graph::Graph(PairMatch * p)
{
  pairMatches.push_back(p);
  Graph(p->pairs.at(0));
}

Graph::Graph(Pair * p)
{
  tracks = p->tracks;
  cameras.push_back(p->camLeft);
  cameras.push_back(p->camRight);
}

Graph::~Graph()
{
}

void Graph::addPairMatch(PairMatch * p)
{
  
}

void Graph::mergeGraph(Graph * g)
{
  //Update the camera + 1 with the last camera of the graph
  g->cameras.back()->updateOrigin(cameras.back()->P);
  cameras.push_back(g->cameras.back());
  std::cout << "Adding graph camera" << cameras.back()->pathImage << std::endl;
  mergeTracks(g->tracks);
  /*unsigned int numTracks = this->tracks.size();
  unsigned int numTracks2 = g->tracks.size();
  //std::vector<int> tracksMerged;
  for (size_t i = 0; i < numTracks; i++)
  {
    for (size_t j = 0; j < numTracks2; j++)
    {
      if (g->tracks.at(j) != NULL)
      {
        if (this->tracks.at(i)->keypoints.back()->isEqual(g->tracks.at(j)->keypoints.front()))
        {
          this->tracks.at(i)->addKeypoint(g->tracks.at(j)->keypoints.back());
          g->tracks.at(j) = NULL;
          //tracksMerged.push_back(j);
          break;//we already found the match
        }
      }
    }
  }
  for (size_t i = 0; i < numTracks2; i++)
  {
    if (g->tracks.at(i) != NULL)
    {
      this->tracks.push_back(g->tracks.at(i));
    }
  }*/
  /*size_t indexAux = 0;
  for (size_t i = 0; i < numTracks2; i++)
  {
    if (i == tracksMerged.at(indexAux))
    {
      if (indexAux + 1 < tracksMerged.size())
      {
        indexAux++;
      }
    }
    else
    {
      this->tracks.push_back(g->tracks.at(i));
    }
  }*/
  //std::cout << "Tracks camera 0 " << numTracks << " Tracks camera 1 " << numTracks2 << " Tracks relacionadas" << tracks.size() << "\n";
}

void Graph::addPair(Pair* p)
{
  bool hasCam1 = false;
  bool hasCam2 = false;
  Camera* cam0 = NULL, *cam1 = NULL;
  for (size_t i = 0; i < cameras.size(); i++)
  {
    if (cameras.at(i) == p->camLeft)
    {
      hasCam1 = true;
      cam0 = p->camLeft;
    }
    if (cameras.at(i) == p->camRight)
    {
      hasCam2 = true;
      cam1 = p->camRight;
    }
  }
  if (!hasCam1 || !hasCam2 || p->pairMerged)
  {
    return;
  }
  std::cout << "Adding pair Camera " << cam0->pathImage << " Camera " << cam1->pathImage << std::endl;
  //mergeTracks(p->tracks);
  addTracks(p->tracks);
  /*for (size_t i = 0; i < tracks.size(); i++)
  {
    for (size_t j = 0; j < p->tracks.size(); j++)
    {
      if (p->tracks.at(j) != NULL)
      {
        if (tracks.at(i)->mergeTrack(p->tracks.at(j)))
        {
          p->tracks.at(j) = NULL;
        }
      }
    }
  }
  for (size_t i = 0; i < p->tracks.size(); i++)
  {
    if (p->tracks.at(i) != NULL)
    {
      tracks.push_back(p->tracks.at(i));
    }
  }*/
  p->pairMerged = true;
}

void Graph::calculate3DPoints()
{
  for (size_t i = 0; i < tracks.size(); i++)
  {
    tracks.at(i)->calculatePoint3D();
  }
}

void Graph::saveSFM(std::string path)
{
  std::ofstream myfile;
  myfile.open(path);
  myfile << cameras.size() <<"\n\n";
  for (size_t i = 0; i < cameras.size(); i++)
  {
    myfile << cameras.at(i)->getStringSFM();
  }
  myfile.close();
}

void Graph::saveNVM(std::string path)
{
  std::ofstream myfile;
  myfile.open(path);
  myfile << "NVM_V3" << "\n\n";
  myfile << cameras.size() << "\n";
  for (size_t i = 0; i < cameras.size(); i++)
  {
    myfile << cameras.at(i)->getStringNVM();
  }
  myfile.close();
}

void Graph::savePointCloud(std::string path)
{
  std::ofstream myfile;
  myfile.open(path);
  cv::Point3f pt;
  for (size_t i = 0; i < tracks.size(); i++)
  {
    pt = tracks.at(i)->getPoint3D();
    if (pt.x != -1 && pt.y != -1 && pt.z != -1)
    {
      myfile << "v " << pt.x << " " << pt.y << " " << pt.z << " 1.0 \n";
    }
  }
  myfile.close();
  /*vtkSmartPointer<vtkPoints> pts = vtkSmartPointer<vtkPoints>::New();
  cv::Point3f pt;
  for (size_t i = 0; i < tracks.size(); i++)
  {
    pt = tracks.at(i)->getPoint3D();
    pts->InsertNextPoint(pt.x, pt.y, pt.z);
  }
  vtkSmartPointer<vtkPolyData> pointsPolydata = vtkSmartPointer<vtkPolyData>::New();
  pointsPolydata->SetPoints(pts);
  vtkSmartPointer<vtkPLYWriter> plyWriter = vtkSmartPointer<vtkPLYWriter>::New();
  plyWriter->SetFileName(path.c_str());
  plyWriter->SetInputData(pointsPolydata);
  plyWriter->Write();*/
}

size_t Graph::getSizeModel()
{
  cv::Point3f pt;
  size_t size = 0;
  for (size_t i = 0; i < tracks.size(); i++)
  {
    pt = tracks.at(i)->getPoint3D();
    if (pt.x != -1 && pt.y != -1 && pt.z != -1)
    {
      size++;
    }
  }
  return size;
}

void Graph::filterPoints()
{
  for (size_t i = 0; i < tracks.size(); i++)
  {
    tracks.at(i)->testPoint();
  }
}

void Graph::mergeTracks(std::vector<Track*> newTracks)
{
  size_t numTracksMerged = 0;
  size_t newTracksMerged = 0;
  size_t numTracks = tracks.size();
  size_t numTracks2 = newTracks.size();
  for (size_t i = 0; i < numTracks; i++)
  {
    for (size_t j = 0; j < numTracks2; j++)
    {
      if (newTracks.at(j) != NULL)
      {
        if (tracks.at(i)->mergeTrack(newTracks.at(j)))
        {
          numTracksMerged++;
          newTracks.at(j) = NULL;
        }
      }
    }
  }
  for (size_t i = 0; i < numTracks2; i++)
  {
    if (newTracks.at(i) != NULL)
    {
      newTracksMerged++;
      tracks.push_back(newTracks.at(i));
    }
  }
  std::cout << "Tracks iniciais " << numTracks << " Tracks novas " << numTracks2 << " Tracks merged " << numTracksMerged << " newTracks " << newTracksMerged << std::endl;
}

void Graph::addTracks(std::vector<Track*> newTracks)
{
  size_t numTracks = tracks.size();
  size_t numTracks2 = newTracks.size();
  for (size_t i = 0; i < numTracks2; i++)
  {
    tracks.push_back(newTracks.at(i));
  }
  std::cout << "Tracks iniciais " << numTracks << " Tracks novas " << numTracks2 << std::endl;
}
