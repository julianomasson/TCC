// SIFT_OpenCV.cpp : Defines the entry point for the console application.
//

#include <iostream>
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\xfeatures2d.hpp>
#include <opencv2\features2d.hpp>
#include <opencv2\calib3d.hpp>
#include <opencv2\stitching.hpp>
#include <algorithm>
#include "PairMatch.h"

#include "Draw.h"
#include <windows.h>
#include "Pair.h"
#include "Graph.h"
#include "BundleAdjustment.h"


using namespace cv;
using namespace std;

cv::Mat skew(cv::InputArray _x);
template<typename T> cv::Mat skewMat(const cv::Mat_<T> &x)
{
  cv::Mat_<T> skew(3, 3);
  skew << 0, -x(2), x(1),
    x(2), 0, -x(0),
    -x(1), x(0), 0;

  return skew;
}


void transformEpipolarLine(cv::Mat H1, std::vector<cv::Vec3f>* lines2)
{
  cv::Vec3f tempLine;
  cv::Mat H1T = H1.inv().t();
  for (int i = 0; i < lines2->size(); i++)
  {
    tempLine = lines2->at(i);
    lines2->at(i)[0] = H1T.at<double>(0, 0)*tempLine[0] + H1T.at<double>(0, 1)*tempLine[1] + H1T.at<double>(0, 2)*tempLine[2];
    lines2->at(i)[1] = H1T.at<double>(1, 0)*tempLine[0] + H1T.at<double>(1, 1)*tempLine[1] + H1T.at<double>(1, 2)*tempLine[2];
    lines2->at(i)[2] = H1T.at<double>(2, 0)*tempLine[0] + H1T.at<double>(2, 1)*tempLine[1] + H1T.at<double>(2, 2)*tempLine[2];
  }
}

cv::Mat skew(cv::InputArray _x)
{
  const cv::Mat x = _x.getMat();
  const int depth = x.depth();
  //CV_Assert(x.size() == Size(3, 1) || x.size() == Size(1, 3));
  //CV_Assert(depth == CV_32F || depth == CV_64F);

  cv::Mat skewMatrix;
  if (depth == CV_32F)
  {
    skewMatrix = skewMat<float>(x);
  }
  else if (depth == CV_64F)
  {
    skewMatrix = skewMat<double>(x);
  }
  else
  {
    //CV_Error(CV_StsBadArg, "The DataType must be CV_32F or CV_64F");
  }

  return skewMatrix;
}

template<typename T> void projectionsFromFundamental(const cv::Mat_<T> &F, cv::Mat_<T> P1, cv::Mat_<T> P2)
{
  P1 << 1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0;

  cv::Vec<T, 3> e2;
  cv::SVD::solveZ(F.t(), e2);

  cv::Mat_<T> P2cols = skew(e2) * F;
  for (char j = 0; j<3; ++j) {
    for (char i = 0; i<3; ++i)
      P2(j, i) = P2cols(j, i);
    P2(j, 3) = e2(j);
  }
}


void computeSFM(std::string pathImages, int qtdImages, bool useOpenCV, bool useAKAZE, bool showMatch, std::string output)
{
  std::vector<std::string> imageFileNames;
  WIN32_FIND_DATA data;
  HANDLE hFind = FindFirstFile((pathImages + "\\*.*").c_str(), &data);//("trilho\\*.JPG", &data);
  if (hFind != INVALID_HANDLE_VALUE) {
    do {
      std::stringstream s;
      s << pathImages << "\\" << data.cFileName;
      if (s.str().back() == 'g' || s.str().back() == 'G')
      {
        imageFileNames.push_back(s.str());
      }
    } while (FindNextFile(hFind, &data));
    FindClose(hFind);
  }

  cv::Mat img = cv::imread(imageFileNames.at(0), CV_LOAD_IMAGE_GRAYSCALE);
  double f = 1.2*std::max(img.cols, img.rows);
  cv::Mat K = (cv::Mat_<double>(3, 3) << f, 0, img.cols / 2, 0, f, img.rows / 2, 0, 0, 1);

  if (qtdImages > imageFileNames.size())
  {
    qtdImages = imageFileNames.size();
  }

  //Load cameras
  std::vector<Camera*> cameras;
  for (size_t i = 0; i < qtdImages; i++)
  {
    cameras.push_back(new Camera(imageFileNames.at(i)));
    if (useOpenCV)
    {
      cameras.back()->computeOpenCVKeypoints(useAKAZE);
      std::cout << "Camera " << i << " Keypoints: " << cameras.back()->keypoints.size() << std::endl;
    }
    else
    {
      cameras.back()->computeSiftGPUKeypoints();
      std::cout << "Camera " << i << " Keypoints: " << cameras.back()->keypointsSiftGPU.size() << std::endl;
    }
    //cameras.back()->drawKeypoints();
  }

  //Create two-view match
  std::vector<Pair*> pairs;
  std::vector<Pair*> additionalPairs;
  bool createAdditionalPairs = true;
  int numMinMatchs = 100;
  for (size_t i = 0; i < cameras.size(); i++)
  {
    for (size_t j = i + 1; j < cameras.size(); j++)
    {
      if (j == i + 1)
      {
        pairs.push_back(new Pair(cameras.at(i), cameras.at(j), K));
        if (useOpenCV)
        {
          pairs.back()->matchOpenCV();
        }
        else
        {
          pairs.back()->matchSiftGPU();
        }
        pairs.back()->createTracks();
        pairs.back()->computePose();
        std::cout << "Camera " << i << " - " << j << " matches: " << pairs.back()->imgLeftPts.size() << std::endl;
      }
      else if (createAdditionalPairs)
      {
        Pair* p = new Pair(cameras.at(i), cameras.at(j), K);
        if (useOpenCV)
        {
          p->matchOpenCV();
        }
        else
        {
          p->matchSiftGPU();
        }
        if (p->imgLeftPts.size() > numMinMatchs)
        {
          p->createTracks();
          additionalPairs.push_back(p);
        }
        else
        {
          break;
        }
        std::cout << "Camera " << i << " - " << j << " matches: " << p->imgLeftPts.size() << std::endl;
      }
    }
    for (size_t j = cameras.size() - 1; j >= i + 1; j--)
    {
      if (createAdditionalPairs)
      {
        Pair* p = new Pair(cameras.at(i), cameras.at(j), K);
        if (useOpenCV)
        {
          p->matchOpenCV();
        }
        else
        {
          p->matchSiftGPU();
        }
        if (p->imgLeftPts.size() > numMinMatchs)
        {
          p->createTracks();
          additionalPairs.push_back(p);
        }
        else
        {
          break;
        }
        std::cout << "Camera " << i << " - " << j << " matches: " << p->imgLeftPts.size() << std::endl;
      }
    }
  }

  //Create the graphs
  std::vector<Graph*> graphs;
  for (size_t i = 0; i < pairs.size(); i++)
  {
    graphs.push_back(new Graph(pairs.at(i)));
  }

  //Merge to a global graph
  graphs.at(0)->calculate3DPoints();
  size_t sizeModel = 0;
  for (size_t i = 1; i < graphs.size(); i++)
  {
    graphs.at(0)->mergeGraph(graphs.at(i));
    for (size_t j = 0; j < additionalPairs.size(); j++)
    {
      if (!additionalPairs.at(j)->pairMerged)
      {
        graphs.at(0)->addPair(additionalPairs.at(j));
      }
    }
    graphs.at(0)->calculate3DPoints();
    /*if (i == graphs.size() - 1)
    {
    break;
    }*/
    BundleAdjustment* pba = new BundleAdjustment(graphs.at(0));
    if (i == 1 || (graphs.at(0)->getSizeModel() > sizeModel*1.05 && graphs.at(0)->cameras.size() > 5))
    {
      pba->runBundle(true);
      graphs.at(0) = pba->getResult();
      sizeModel = graphs.at(0)->getSizeModel();
    }
    else
    {
      pba->runBundle(false);
      graphs.at(0) = pba->getResult();
    }
    graphs.at(0)->filterPoints();
  }
  BundleAdjustment* pba = new BundleAdjustment(graphs.at(0));
  pba->runBundle(true);
  graphs.at(0) = pba->getResult();
  graphs.at(0)->saveSFM(output + ".sfm");
  graphs.at(0)->saveNVM(output + ".nvm");
  graphs.at(0)->savePointCloud(output + ".obj");

  /*ofstream myfile;
  myfile.open(output + ".txt");
  Track* t;
  cv::Point3f pt3D;
  for (size_t i = 0; i < graphs.at(0)->tracks.size(); i++)
  {
    t = graphs.at(0)->tracks.at(i);
    pt3D = t->getPoint3D();
    myfile << "Point " << pt3D.x << " " << pt3D.y << " " << pt3D.z << " seen by " << t->keypoints.size() << "\n";
    for (size_t j = 0; j < t->keypoints.size(); j++)
    {
      float dummy_query_data[3] = { pt3D.x, pt3D.y, pt3D.z };
      cv::Mat dummy_query = cv::Mat(1, 3, CV_32F, dummy_query_data);
      myfile << j << ": " << t->keypoints.at(j)->getReprojectionError(dummy_query) << "\n";
    }
    myfile << "-----------------------------------------------------" << "\n";
  }
  myfile.close();*/


  if (!showMatch)
  {
    return;
  }
  bool matchBymatch = 0;
  //Draw
  cv::Mat imgConcat = graphs.at(0)->cameras.at(0)->img;
  for (size_t i = 1; i < graphs.at(0)->cameras.size(); i++)
  {
    hconcat(imgConcat, graphs.at(0)->cameras.at(i)->img, imgConcat);
  }
  cv::cvtColor(imgConcat, imgConcat, cv::COLOR_GRAY2BGR);
  cv::Point2d p;
  int color = 0;
  namedWindow("im1", CV_WINDOW_NORMAL);
  for (size_t i = 0; i < graphs.at(0)->tracks.size(); i++)
  {
    Track* t = graphs.at(0)->tracks.at(i);
    if (t->keypoints.size() > 2)
    {
      color = 255;
    }
    else
    {
      color = 0;
    }
    for (size_t j = 0; j < t->keypoints.size(); j++)
    {
      p = t->keypoints.at(j)->getPoint();
      for (size_t k = 0; k < graphs.at(0)->cameras.size(); k++)
      {
        if (t->keypoints.at(j)->cam == graphs.at(0)->cameras.at(k))
        {
          p.x += k*img.cols;
          break;
        }
      }
      if (color == 255)
      {
        cv::circle(imgConcat, p, 5, cv::Scalar(color, 0, 0), 3);
      }
      else
      {
        cv::circle(imgConcat, p, 5, cv::Scalar(color, 0, 0), 2);
      }
    }
    if (matchBymatch)
    {
      imshow("im1", imgConcat);
      waitKey(0);
    }
  }
  cv::imwrite(output + ".jpg", imgConcat);
  imshow("im1", imgConcat);
  waitKey(0);
}

std::vector<Camera*> loadSFM(std::string path)
{
  std::vector<Camera*> cameras;
  std::ifstream in(path.c_str());
  if (!in.good())
  {
    return cameras;
  }

  // TODO: Handle multiple models.

  /* Read number of views. */
  int num_views = 0;
  in >> num_views;
  /* Discard the next empty line */
  {
    std::string temp;
    std::getline(in, temp);
  }
  if (num_views < 0 || num_views > 10000)
  {
    return cameras;
  }

   std::string pathImage;
  for (int i = 0; i < num_views; ++i)
  {
    /* Filename*/
    in >> pathImage;

    Camera* cam = new Camera(pathImage);

    /* Camera rotation*/
    for (size_t j = 0; j < 3; j++)
    {
      for (size_t k = 0; k < 3; k++)
      {
        in >> cam->P(j, k);
      }
    }
    /* Camera translation*/
    for (size_t k = 0; k < 3; k++)
    {
      in >> cam->P(k, 3);
    }
    /* Focal length. */
    in >> cam->K.at<double>(0,0);
    cam->K.at<double>(1, 1) = cam->K.at<double>(0, 0);

    cameras.push_back(cam);

    /* Getting the extra information not used to go to the next line */
    float temp;
    in >> temp;
    in.eof();
  }
  in.close();

  return cameras;
}

int main(int argc, const char* argv[])
{
  //freopen("out.txt", "w", stdout);
  if (argc < 7)
  {
    std::cout << "path qtdImages showMatch useOpenCV useAKAZE computeDense pathSFMDense[É necessário criar uma pasta Results no diretório do exe]" << std::endl;
    //return 0;
  }
  std::string pathImages = "D:\\UFSC\\Graduacao\\10_Semestre\\TCC2\\SIFT_OpenCV\\SIFT_OpenCV\\gargoyle\\rotated";//argv[1];//"C:\\UFSC\\10_Semestre\\TCC2\\SIFT_OpenCV\\SIFT_OpenCV\\trilho";//
  int qtdImages = 100;//atoi(argv[2]);//100;
  int showMatch = 0;//atoi(argv[3]);
  bool useOpenCV = true;//atoi(argv[4]); //true;
  bool useAKAZE = false;//atoi(argv[5]); //true;
  bool computeDense = false;//atoi(argv[6]); //true;
  std::string pathDense = "mesh.sfm";//

  std::ofstream myfile;
  DWORD initial_time = GetTickCount();
  DWORD variableTime;
  DWORD intialPairTime, timeALL = 0, timeSequence = 0;
  if (!computeDense)
  {
    myfile.open("log_SFM.txt");
    myfile << "Imagens: " << pathImages << "\n";
    myfile << "useOpenCV: " << useOpenCV << "\n";
    myfile << "useAKAZE: " << useAKAZE << "\n";
    computeSFM(pathImages, qtdImages, useOpenCV, useAKAZE, showMatch, "Results\\gargoyle");
    myfile << "SFM Time: " << Utils::calcTime(initial_time, GetTickCount()) << "\n";
    myfile.close();
    return 0;
  }

  myfile.open("log_dense.txt");
  myfile << "Arquivo: " << pathDense << "\n";
  initial_time = GetTickCount();
  std::vector<Camera*> cameras = loadSFM(pathDense);

  //Graph* newGrap = new Graph();
  //newGrap->cameras = cameras;
  //newGrap->saveNVM("SIFT.nvm");

  cv::Mat img = cameras.at(0)->img;
  double f = 1.2*std::max(img.cols, img.rows);
  cv::Mat K = (cv::Mat_<double>(3, 3) << f, 0, img.cols / 2, 0, f, img.rows / 2, 0, 0, 1);

  std::vector<Pair*> pairs;
  std::vector<Pair*> pairsSequence;
  double angle = 0;
  double numberOfPoints = 0;
  //int j = 0;
  for (size_t i = 0; i < cameras.size(); i++)
  {
    //j = i + 1;
    //if (i + 1 < cameras.size())
    for (size_t j = i + 1; j < cameras.size(); j++)//cameras.size()
    {
      std::cout << i << "_" << j << std::endl;
      angle = cameras.at(i)->getAngleBetweenCameras(cameras.at(j));
      if (angle > 5 && angle < 30)
      {
        intialPairTime = GetTickCount();
        myfile << "*******************************************************************\n";
        myfile << i << " - " << j <<"\n";
        pairs.push_back(new Pair(cameras.at(i), cameras.at(j), K));

        variableTime = GetTickCount();
        pairs.back()->computeRectify();
        myfile << "computeRectify Time: " << Utils::calcTime(variableTime, GetTickCount()) << "\n";

        variableTime = GetTickCount();
        pairs.back()->computeDenseSIFT();
        myfile << "computeDenseSIFT Time: " << Utils::calcTime(variableTime, GetTickCount()) << "\n";

        variableTime = GetTickCount();
        pairs.back()->matchDenseSIFT();
        myfile << "matchDenseSIFT Time: " << Utils::calcTime(variableTime, GetTickCount()) << "\n";

        variableTime = GetTickCount();
        pairs.back()->createInitialSeeds();
        myfile << "createInitialSeeds Time: " << Utils::calcTime(variableTime, GetTickCount()) << "\n";

        myfile << "*******************************************************************\n";
        variableTime = GetTickCount();
        myfile << pairs.back()->computeNewSeeds();
        myfile << "computeNewSeeds Time: " << Utils::calcTime(variableTime, GetTickCount()) << "\n";
        myfile << "*******************************************************************\n";

        variableTime = GetTickCount();
        pairs.back()->compute3DPoints();
        myfile << "compute3DPoints Time: " << Utils::calcTime(variableTime, GetTickCount()) << "\n";

        variableTime = GetTickCount();
        std::stringstream s2;
        s2 << "testeGeral/" << i << "_" << j << "_" << angle << "_" << "dense.ply";
        pairs.back()->createPCLPointCloud();
        myfile << "createPCLPointCloud Time: " << Utils::calcTime(variableTime, GetTickCount()) << "\n";
        myfile << "Seeds/Seeds with reproj error < 1.0 diferrence " << pairs.back()->seeds.size() << "/" << pairs.back()->cloudPCL->size() << " " << pairs.back()->seeds.size() - pairs.back()->cloudPCL->size() << "\n";

        numberOfPoints = pairs.back()->cloudPCL->size();
        variableTime = GetTickCount();
        pairs.back()->filterPCLCloud();
        myfile << "filterPCLCloud Time: " << Utils::calcTime(variableTime, GetTickCount()) << "\n";
        myfile << "Points/PointsFiltered  erased " << numberOfPoints << "/" << pairs.back()->cloudPCL->size() << "  " << (numberOfPoints - pairs.back()->cloudPCL->size()) << "\n";

        variableTime = GetTickCount();
        pairs.back()->computePCLNormal();
        myfile << "computePCLNormal Time: " << Utils::calcTime(variableTime, GetTickCount()) << "\n";

        variableTime = GetTickCount();
        pairs.back()->savePCLResult(s2.str());
        myfile << "savePCLResult Time: " << Utils::calcTime(variableTime, GetTickCount()) << "\n";

        variableTime = GetTickCount();
        std::stringstream s;
        s << "testeGeral/" << i << "_" << j << "_";
        pairs.back()->saveMasks(s.str());
        myfile << "saveMasks Time: " << Utils::calcTime(variableTime, GetTickCount()) << "\n";

        pairs.back()->clearSeeds();
        myfile << "*******************************************************************\n";

        timeALL += GetTickCount() - intialPairTime;
        if (j == i + 1)
        {
          pairsSequence.push_back(pairs.back());
          timeSequence += GetTickCount() - intialPairTime;
        }
      }
    }
  }

  variableTime = GetTickCount();
  pcl::PointCloud<pcl::PointNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointNormal>);
  for (size_t i = 0; i < pairs.size(); i++)
  {
    for (size_t j = 0; j < pairs.at(i)->cloudPlusNormalsPCL->size(); j++)
    {
      cloud->push_back(pairs.at(i)->cloudPlusNormalsPCL->at(j));
    }
  }
  pcl::io::savePLYFileBinary("geralDense.ply", *cloud);
  pcl::PointCloud<pcl::PointNormal>::Ptr cloudSeq(new pcl::PointCloud<pcl::PointNormal>);
  for (size_t i = 0; i < pairsSequence.size(); i++)
  {
    for (size_t j = 0; j < pairsSequence.at(i)->cloudPlusNormalsPCL->size(); j++)
    {
      cloudSeq->push_back(pairsSequence.at(i)->cloudPlusNormalsPCL->at(j));
    }
  }
  pcl::io::savePLYFileBinary("geralDenseSequence.ply", *cloudSeq);
  myfile << "saveGeral Time: " << Utils::calcTime(variableTime, GetTickCount()) << "\n";
  //pcl::RadiusOutlierRemoval<pcl::PointNormal> outrem;
  //outrem.setInputCloud(cloud);
  //outrem.setRadiusSearch(0.01);
  //outrem.setMinNeighborsInRadius(200);
  //pcl::PointCloud<pcl::PointNormal>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointNormal>);
  //outrem.filter(*cloud_filtered);
  //pcl::io::savePLYFileBinary("geralDenseFiltered.ply", *cloud_filtered);

  myfile << "Total Time: " << Utils::calcTime(initial_time, GetTickCount()) << "\n";
  myfile << "Sequence pairs Time: " << Utils::getTimeString(timeSequence) << "\n";
  myfile << "ALL pairs Time: " << Utils::getTimeString(timeALL) << "\n";
  myfile.close();
  return 0;
}

