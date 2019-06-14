// SIFT_OpenCV.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"


//#include <vtkAutoInit.h>
//VTK_MODULE_INIT(vtkRenderingOpenGL2);
//
//#define vtkRenderingCore_AUTOINIT 3(vtkInteractionStyle,vtkRenderingFreeType,vtkRenderingOpenGL2)
//#define vtkRenderingVolume_AUTOINIT 1(vtkRenderingVolumeOpenGL2)

#include <iostream>
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\xfeatures2d.hpp>
#include <opencv2\features2d.hpp>
#include <opencv2\calib3d.hpp>
#include <opencv2\stitching.hpp>
#include <algorithm>
#include "PairMatch.h"

//#include <exiv2\exiv2.hpp>

#include "Draw.h"
#include <windows.h>
#include "Pair.h"
#include "Graph.h"
#include "BundleAdjustment.h"
#include "Plane.h"


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

/*bool setFocalLengthFromExif(Exiv2::ExifData exifData, int width, int height)
{
  static const float kMinFocalLength = 1e-2;

  if (!exifData["Exif.Photo.FocalLength"].getValue()->ok() ||
  !exifData["Exif.Photo.FocalPlaneXResolution"].getValue()->ok() ||
   !exifData["Exif.Photo.FocalPlaneYResolution"].getValue()->ok() ||
    !exifData["Exif.Photo.FocalPlaneResolutionUnit"].getValue()->ok())
  {
    return 0;
  }
  Exiv2::Rational exif_focal_length1 = exifData["Exif.Photo.FocalLength"].getValue()->toRational();
  float exif_focal_length = exif_focal_length1.first / exif_focal_length1.second;
  Exiv2::Rational focal_plane_x_resolution1 = exifData["Exif.Photo.FocalPlaneXResolution"].getValue()->toRational();
  float focal_plane_x_resolution = focal_plane_x_resolution1.first / focal_plane_x_resolution1.second;
  Exiv2::Rational focal_plane_y_resolution1 = exifData["Exif.Photo.FocalPlaneYResolution"].getValue()->toRational();
  float focal_plane_y_resolution = focal_plane_y_resolution1.first / focal_plane_y_resolution1.second;
  int focal_plane_resolution_unit = exifData["Exif.Photo.FocalPlaneResolutionUnit"].getValue()->toFloat();

  // Make sure the values are sane.
  if (exif_focal_length <= kMinFocalLength || focal_plane_x_resolution <= 0.0 ||
    focal_plane_y_resolution <= 0.0) {
    return false;
  }

  // CCD resolution is the pixels per unit resolution of the CCD.
  double ccd_resolution_units = 1.0;
  switch (focal_plane_resolution_unit) {
  case 2:
    // Convert inches to mm.
    ccd_resolution_units = 25.4;
    break;
  case 3:
    // Convert centimeters to mm.
    ccd_resolution_units = 10.0;
    break;
  case 4:
    // Already in mm.
    break;
  case 5:
    // Convert micrometers to mm.
    ccd_resolution_units = 1.0 / 1000.0;
    break;
  default:
    return false;
    break;
  }

  // Get the ccd dimensions in mm.
  const int exif_width = exifData["Exif.Photo.PixelXDimension"].getValue()->toLong();
  const int exif_height = exifData["Exif.Photo.PixelYDimension"].getValue()->toLong();
  const double ccd_width =
    exif_width / (focal_plane_x_resolution / ccd_resolution_units);
  const double ccd_height =
    exif_height / (focal_plane_y_resolution / ccd_resolution_units);

  const double focal_length_x =
    exif_focal_length * width / ccd_width;
  const double focal_length_y =
    exif_focal_length * height / ccd_height;

  // Normalize for the image size in case the original size is different
  // than the current size.
  const double focal_length = (focal_length_x + focal_length_y) / 2.0;
  return true;//IsValidFocalLength(focal_length);
}*/


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

void testeArtigoDENSE(std::vector<Camera*> cameras)
{
  /*std::vector<Camera*> nearCameras;
  std::vector<double> medianVector;
  double angle;
  double median;
  for (size_t i = 1; i < cameras.size(); i++)
  {
    angle = cameras.at(0)->getAngleBetweenCameras(cameras.at(i));
    if (angle > 5 && angle < 60)
    {
      nearCameras.push_back(cameras.at(i));
      medianVector.push_back(cameras.at(0)->distanceBetweenOpticalCenter(cameras.at(i)));
    }
  }
  size_t size = medianVector.size();
  sort(medianVector.begin(), medianVector.end());
  if (size % 2 == 0)
  {
    median = (medianVector[size / 2 - 1] + medianVector[size / 2]) / 2;
  }
  else
  {
    median = medianVector[size / 2];
  }*/
  cv::Mat img = cameras.at(0)->img;
  std::vector<Plane*> planes;
  for (size_t i = 0; i < img.cols; i++)
  {
    for (size_t j = 0; j < img.rows; j++)
    {
      planes.push_back(new Plane(cameras.at(0), cameras.at(1), i, j));
      planes.back()->m();
    }
  }
}

#include <pcl/TextureMesh.h>
#include <pcl/surface/texture_mapping.h>
#include <pcl/io/obj_io.h>

void camPoseTCC2PCL(Camera* camTCC, pcl::TextureMapping<pcl::PointXYZ>::Camera &cam)
{

  /*cv::Mat Pnew = cv::Mat_<double>(camTCC->P.rows, camTCC->P.cols);
  for (size_t i = 0; i < camTCC->P.rows; i++)
  {
    for (size_t j = 0; j < camTCC->P.cols; j++)
    {
      Pnew.at<double>(i, j) = camTCC->P(i, j);
    }
  }
  camTCC->K.at<double>(0, 0) = camTCC->K.at<double>(0, 0) / 10.0f;
  camTCC->K.at<double>(1, 1) = camTCC->K.at<double>(1, 1) / 10.0f;*/
  //Pnew = camTCC->K*Pnew;
  cam.texture_file = camTCC->pathImage;
  /*for (size_t i = 0; i < camTCC->P.rows; i++)
  {
    for (size_t j = 0; j < camTCC->P.cols; j++)
    {
      cam.pose(i, j) = camTCC->P(i, j);
    }
  }*/
  cv::Mat center = Utils::getCenterVector(camTCC->P);
  for (size_t i = 0; i < camTCC->P.rows; i++)
  {
    cam.pose(i, 3) = center.at<double>(i, 0);
  }
  cv::Mat R = cv::Mat_<double>(3, 3);
  Utils::getRotationMatrix(R, camTCC->P);
  R = R.inv();

  for (size_t i = 0; i < camTCC->P.rows; i++)
  {
    for (size_t j = 0; j < camTCC->P.cols - 1; j++)
    {
      cam.pose(i, j) = R.at<double>(i, j);
    }
  }

  cam.pose(3, 0) = 0.0;
  cam.pose(3, 1) = 0.0;
  cam.pose(3, 2) = 0.0;
  cam.pose(3, 3) = 1.0; //Scale
  
  cam.focal_length = camTCC->K.at<double>(0, 0);
  cam.height = camTCC->img.rows;
  cam.width = camTCC->img.cols;

  cam.center_h = cam.height / 2.0f;
  cam.center_w = cam.width / 2.0f;
}
#include <pcl/visualization/pcl_visualizer.h>
#include "vtkAutoInit.h" 
VTK_MODULE_INIT(vtkRenderingOpenGL);
/** \brief Display a 3D representation showing the a cloud and a list of camera with their 6DOf poses */
void showCameras(pcl::texture_mapping::CameraVector cams, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{

  // visualization object
  pcl::visualization::PCLVisualizer visu("cameras");

  // add a visual for each camera at the correct pose
  for (int i = 0; i < cams.size(); ++i)
  {
    // read current camera
    pcl::TextureMapping<pcl::PointXYZ>::Camera cam = cams[i];
    double focal = cam.focal_length;
    double height = cam.height;
    double width = cam.width;

    // create a 5-point visual for each camera
    pcl::PointXYZ p1, p2, p3, p4, p5;
    p1.x = 0; p1.y = 0; p1.z = 0;
    double angleX = RAD2DEG(2.0 * atan(width / (2.0*focal)));
    double angleY = RAD2DEG(2.0 * atan(height / (2.0*focal)));
    double dist = 0.75;
    double minX, minY, maxX, maxY;
    maxX = dist*tan(atan(width / (2.0*focal)));
    minX = -maxX;
    maxY = dist*tan(atan(height / (2.0*focal)));
    minY = -maxY;
    p2.x = minX; p2.y = minY; p2.z = dist;
    p3.x = maxX; p3.y = minY; p3.z = dist;
    p4.x = maxX; p4.y = maxY; p4.z = dist;
    p5.x = minX; p5.y = maxY; p5.z = dist;
    p1 = pcl::transformPoint(p1, cam.pose);
    p2 = pcl::transformPoint(p2, cam.pose);
    p3 = pcl::transformPoint(p3, cam.pose);
    p4 = pcl::transformPoint(p4, cam.pose);
    p5 = pcl::transformPoint(p5, cam.pose);
    std::stringstream ss;
    ss << "Cam #" << i + 1;
    visu.addText3D(ss.str(), p1, 0.1, 1.0, 1.0, 1.0, ss.str());

    ss.str("");
    ss << "camera_" << i << "line1";
    visu.addLine(p1, p2, ss.str());
    ss.str("");
    ss << "camera_" << i << "line2";
    visu.addLine(p1, p3, ss.str());
    ss.str("");
    ss << "camera_" << i << "line3";
    visu.addLine(p1, p4, ss.str());
    ss.str("");
    ss << "camera_" << i << "line4";
    visu.addLine(p1, p5, ss.str());
    ss.str("");
    ss << "camera_" << i << "line5";
    visu.addLine(p2, p5, ss.str());
    ss.str("");
    ss << "camera_" << i << "line6";
    visu.addLine(p5, p4, ss.str());
    ss.str("");
    ss << "camera_" << i << "line7";
    visu.addLine(p4, p3, ss.str());
    ss.str("");
    ss << "camera_" << i << "line8";
    visu.addLine(p3, p2, ss.str());
  }

  // add a coordinate system
  visu.addCoordinateSystem(1.0, "global");

  // add the mesh's cloud (colored on Z axis)
  pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ> color_handler(cloud, "z");
  visu.addPointCloud(cloud, color_handler, "cloud");

  // reset camera
  visu.resetCamera();

  // wait for user input
  visu.spin();
}

std::string textureMesh(std::vector<Camera*> cameras, pcl::PolygonMesh triangles, pcl::PointCloud<pcl::Normal>::Ptr normals)
{
  pcl::TextureMesh mesh;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromPCLPointCloud2(triangles.cloud, *cloud);
  // Create the texturemesh object that will contain our UV-mapped mesh
  mesh.cloud = triangles.cloud;
  mesh.tex_polygons.push_back(triangles.polygons);
  // Load textures and cameras poses and intrinsics
  PCL_INFO("\nLoading textures and camera poses...\n");
  pcl::texture_mapping::CameraVector my_cams;

  for (size_t i = 0; i < cameras.size(); i++)
  {
    pcl::TextureMapping<pcl::PointXYZ>::Camera cam;
    camPoseTCC2PCL(cameras.at(i), cam);
    my_cams.push_back(cam);
  }
  PCL_INFO("\tLoaded %d textures.\n", my_cams.size());
  PCL_INFO("...Done.\n");

  // Create materials for each texture (and one extra for occluded faces)
  mesh.tex_materials.resize(my_cams.size() + 1);
  for (int i = 0; i <= my_cams.size(); ++i)
  {
    pcl::TexMaterial mesh_material;
    mesh_material.tex_Ka.r = 0.2f;
    mesh_material.tex_Ka.g = 0.2f;
    mesh_material.tex_Ka.b = 0.2f;

    mesh_material.tex_Kd.r = 0.8f;
    mesh_material.tex_Kd.g = 0.8f;
    mesh_material.tex_Kd.b = 0.8f;

    mesh_material.tex_Ks.r = 1.0f;
    mesh_material.tex_Ks.g = 1.0f;
    mesh_material.tex_Ks.b = 1.0f;

    mesh_material.tex_d = 1.0f;
    mesh_material.tex_Ns = 75.0f;
    mesh_material.tex_illum = 2;

    std::stringstream tex_name;
    tex_name << "material_" << i;
    tex_name >> mesh_material.tex_name;

    if (i < my_cams.size())
      mesh_material.tex_file = my_cams[i].texture_file;
    else
      mesh_material.tex_file = "occluded.jpg";

    mesh.tex_materials[i] = mesh_material;
  }
  // Sort faces
  PCL_INFO("\nSorting faces by cameras...\n");
  pcl::TextureMapping<pcl::PointXYZ> tm; // TextureMapping object that will perform the sort
  DWORD begin_time = clock();
  tm.textureMeshwithMultipleCameras(mesh, my_cams);
  DWORD end_time = clock();
  DWORD process_time = (end_time - begin_time) / (double)CLOCKS_PER_SEC;
  std::string parametros = "Tempo de Execucao = " + Utils::getTimeString(process_time) + " seg\n";

  PCL_INFO("Sorting faces by cameras done.\n");
  for (int i = 0; i <= my_cams.size(); ++i)
  {
    PCL_INFO("\tSub mesh %d contains %d faces and %d UV coordinates.\n", i, mesh.tex_polygons[i].size(), mesh.tex_coordinates[i].size());
    parametros += "Camera " + std::to_string(i) + " faces = " + std::to_string(mesh.tex_polygons[i].size()) + " UV coordinates = " + std::to_string(mesh.tex_coordinates[i].size()) + "\n";
  }
  // Concatenate XYZ and normal fields
  pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normalsNew(new pcl::PointCloud<pcl::PointNormal>);
  pcl::concatenateFields(*cloud, *normals, *cloud_with_normalsNew);
  pcl::toPCLPointCloud2(*cloud_with_normalsNew, mesh.cloud);
  PCL_INFO("...Done.\n");
  pcl::io::saveOBJFile("PCLTEXTURE.obj", mesh, 5);
  return parametros;
}

int main(int argc, const char* argv[])
{
  //freopen("out.txt", "w", stdout);
  if (argc < 7)
  {
    std::cout << "path qtdImages showMatch useOpenCV useAKAZE computeDense pathSFMDense[É necessário criar uma pasta Results no diretório do exe]" << std::endl;
    //return 0;
  }
  std::string pathImages = "C:\\UFSC\\10_Semestre\\TCC2\\SIFT_OpenCV\\SIFT_OpenCV\\gargoyle\\rotated";//argv[1];//"C:\\UFSC\\10_Semestre\\TCC2\\SIFT_OpenCV\\SIFT_OpenCV\\trilho";//
  int qtdImages = 100;//atoi(argv[2]);//100;
  int showMatch = 0;//atoi(argv[3]);
  bool useOpenCV = true;//atoi(argv[4]); //true;
  bool useAKAZE = false;//atoi(argv[5]); //true;
  bool computeDense = true;//atoi(argv[6]); //true;
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

  //TexturePCL
  pcl::PolygonMesh triangles;
  pcl::io::loadPLYFile("poissonrecon.ply", triangles);
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  pcl::io::loadPLYFile<pcl::Normal>("poissonrecon.ply", *normals);
  myfile << textureMesh(cameras, triangles, normals);
  myfile << "Tempo" << Utils::calcTime(initial_time, GetTickCount()) << "\n";

  myfile.close();
  return 0;
  //Graph* newGrap = new Graph();
  //newGrap->cameras = cameras;
  //newGrap->saveNVM("SIFT.nvm");

  cv::Mat img = cameras.at(0)->img;
  double f = 1.2*std::max(img.cols, img.rows);
  cv::Mat K = (cv::Mat_<double>(3, 3) << f, 0, img.cols / 2, 0, f, img.rows / 2, 0, 0, 1);

  //testeArtigoDENSE(cameras);

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
  


  //teste ZNCC
  //cv::Mat img = cv::imread("imgT.png", CV_LOAD_IMAGE_GRAYSCALE);
  //cv::Mat temp = cv::imread("templateInv.png", CV_LOAD_IMAGE_GRAYSCALE);
  //ZNCC(img, temp, 15, 1, 15, Vec3f(0,1,-1));
  //ZNCC(img, temp, 1, 1, 15, Vec3f(0, 1, -1));
  // 
  // now, you can no more create an instance on the 'stack', like in the tutorial
  // (yea, noticed for a fix/pr).
  // you will have to use cv::Ptr all the way down:
  //


  //cv::namedWindow("im1", CV_WINDOW_NORMAL);
  //cv::namedWindow("im2", CV_WINDOW_NORMAL);
  //cv::namedWindow("im1wrap", CV_WINDOW_NORMAL);
  //cv::namedWindow("im2wrap", CV_WINDOW_NORMAL);
  //if (argc < 6)
  //{
  //  std::cout << "img1 img2 use_AKAZE(bool) windowSize maxDist drawSomething 0 sift 1 dense 2 none" << std::endl;
  //  //return 0;
  //}
  //else
  //{
  //  std::cout << "use_AKAZE(bool) " << atoi(argv[3]) << " windowSize " << atoi(argv[4]) << " maxDist " << atoi(argv[5]) << std::endl;
  //}
  //int draw = 2;//atoi(argv[6]);
  //bool drawSIFTepipolar = 0;
  //bool drawDENSEepipolar = 0;
  //if (draw == 0)
  //{
  //  drawSIFTepipolar = true;
  //}
  //else if (draw == 1)
  //{
  //  drawDENSEepipolar = true;
  //}
  //string pathLeft = "img2.jpg";//argv[1];
  //string pathRight = "img3.jpg";//argv[2];
  //cv::Mat img_1 = cv::imread(pathLeft, CV_LOAD_IMAGE_GRAYSCALE);//cv::imread(argv[1], 0);//
  //cv::Mat img_2 = cv::imread(pathRight, CV_LOAD_IMAGE_GRAYSCALE);//cv::imread(argv[2], 0);//
  //cv::Mat img_3 = cv::imread("img4.jpg", CV_LOAD_IMAGE_GRAYSCALE);//cv::imread(argv[1], 0);
  ////cv::Mat img_4 = cv::imread("img4.jpg", 0);//cv::imread(argv[2], 0);

  //double f = 2650;//1.2*std::max(img_1.cols, img_1.rows);//2650
  //cv::Mat K = (cv::Mat_<double>(3, 3) << f, 0, img_1.cols / 2, 0, f, img_1.rows / 2, 0, 0, 1);

  //Camera* camLeft = new Camera(pathLeft, K);//, atoi(argv[3]));
  //Camera* camRight = new Camera(pathRight, K);//, atoi(argv[3]));
  //Camera* camRight2 = new Camera("img4.jpg", K);

  //PairMatch* p = new PairMatch(camLeft, camRight,K);//, 0.75f, atoi(argv[3]));
  //p->computePose();
  //p->computePointCloud();
  //PairMatch* p2 = new PairMatch(camRight, camRight2, K);//, 0.75f, atoi(argv[3]));
  //p2->computePose();
  ////p->saveSFM(".\coruja23.sfm");
  //p2->computePointCloud();
  ////camRight2->updateOrigin(p->Pright);

  //


  //vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
  //vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
  //renderWindow->AddRenderer(renderer);
  //vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
  //vtkSmartPointer<vtkInteractorStyleTrackballCamera> style = vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New(); //like paraview
  //renderWindowInteractor->SetInteractorStyle(style);
  //renderWindowInteractor->SetRenderWindow(renderWindow);

  ////Draw::createPointCloud(renderer, p->pointCloud);
  //Draw::createPointCloud(renderer, p2->pointCloud);
  //Draw::createCamera(renderer, img_1.cols, img_1.rows, K.at<double>(0, 0), (cv::Mat)p2->Pleft, img_2);
  //Draw::createCamera(renderer, img_1.cols, img_1.rows, K.at<double>(0, 0), (cv::Mat)p2->Pright, img_3);
  ////Draw::createCamera(renderer, img_1.cols, img_1.rows, K.at<double>(0, 0), (cv::Mat)camRight2->P, camRight2->img);

  //renderWindow->Render();
  //renderWindowInteractor->Start();



  //cv::Mat F = cv::findFundamentalMat(p->imgLeftPts, p->imgRightPts);

  //std::vector<cv::Vec3f> lines1Left, lines2Right;
  //cv::computeCorrespondEpilines(cv::Mat(p->imgLeftPts), 1, F, lines1Left);
  //cv::computeCorrespondEpilines(cv::Mat(p->imgRightPts), 2, F, lines2Right);



  //std::vector<cv::Point2f> imgPts;
  //for (size_t i = 0; i < img_1.cols; i++)
  //{
  //  for (size_t j = 0; j < img_1.rows; j++)
  //  {
  //    imgPts.push_back(Point2f(j, i));
  //  }
  //}

  //std::vector<cv::Vec3f> lines1, lines2;
  //cv::computeCorrespondEpilines(cv::Mat(imgPts), 1, F, lines1);
  //cv::computeCorrespondEpilines(cv::Mat(imgPts), 2, F, lines2);

  //cv::Mat H1, H2;
  //cv::stereoRectifyUncalibrated(p->imgLeftPts, p->imgRightPts, F, img_1.size(), H1, H2);

  //cv::Mat S = shearingTransform(img_1.cols, img_1.rows, H1, H2);
  //H1 = S * H1;
  //H2 = S * H2;

  //cv::Mat img1_warp, img2_warp;
  //cv::warpPerspective(img_1, img1_warp, H1, img_1.size());
  //cv::warpPerspective(img_2, img2_warp, H2, img_2.size());


  ///*Ptr<StereoSGBM> sgbm = StereoSGBM::create(-64,192,5,600,2400,10,4,1,150,2, 0);

  //cv::Mat disp,disp8;
  //sgbm->compute(img1_remap,img2_remap,disp);
  //normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U);
  //cv::imshow("im1", disp8);
  //cv::waitKey(0);*/

  //std::vector<cv::Point2f> imgLeftPtsTransf, imgRightPtsTransf;
  //perspectiveTransform(p->imgLeftPts, imgLeftPtsTransf, H1);
  //perspectiveTransform(p->imgRightPts, imgRightPtsTransf, H2);

  //std::vector<cv::Point2f> imgPtsTransf;
  //perspectiveTransform(imgPts, imgPtsTransf, H1);

  //transformEpipolarLine(H1, &lines2);
  //transformEpipolarLine(H2, &lines1);
  //transformEpipolarLine(H1, &lines2Right);
  //transformEpipolarLine(H2, &lines1Left);

  //if (drawSIFTepipolar)
  //{
  //  cv::Mat imgConcat;
  //  hconcat(img1_warp, img2_warp, imgConcat);
  //  cv::Mat imgConcatDraw;
  //  for (size_t i = 0; i < imgLeftPtsTransf.size(); i++)
  //  {
  //    imgConcat.copyTo(imgConcatDraw);
  //    cv::circle(imgConcatDraw, imgLeftPtsTransf.at(i), 5, cv::Scalar(0, 0, 0), 2);
  //    cv::circle(imgConcatDraw, cv::Point(imgRightPtsTransf.at(i).x + img_1.cols, imgRightPtsTransf.at(i).y), 5, cv::Scalar(0, 0, 0), 2);
  //    cv::line(imgConcatDraw, cv::Point(img_1.cols, -(lines1Left.at(i)[2] + lines1Left.at(i)[0] * img_1.cols) / lines1Left.at(i)[1]), cv::Point(imgConcat.cols, -(lines1Left.at(i)[2] + lines1Left.at(i)[0] * imgConcat.cols) / lines1Left.at(i)[1]), cv::Scalar(0, 0, 0), 2);
  //    cv::line(imgConcatDraw, cv::Point(0, -lines2Right.at(i)[2] / lines2Right.at(i)[1]), cv::Point(img_1.cols, -(lines2Right.at(i)[2] + lines2Right.at(i)[0] * img_1.cols) / lines2Right.at(i)[1]), cv::Scalar(0, 0, 0), 2);
  //    cv::imshow("im1", imgConcatDraw);
  //    cv::waitKey(0);
  //  }
  //}

  ///*cv::Mat error = cv::Mat(img1_warp.size(), CV_8UC1);
  //uchar erro;
  //for (double i = 0; i < imgLeftPtsTransf.size(); i++)
  //{
  //  if (imgLeftPtsTransf.at(i).y > 0 && imgLeftPtsTransf.at(i).y < error.rows && imgLeftPtsTransf.at(i).x > 0 && imgLeftPtsTransf.at(i).x < error.cols)
  //  {
  //    erro = ceil(abs(imgLeftPtsTransf.at(i).y - imgRightPtsTransf.at(i).y));
  //    error.at<uchar>(imgLeftPtsTransf.at(i).y, imgLeftPtsTransf.at(i).x) = erro;//(erro > 255) ? 255 : erro;
  //  }
  //  
  //}
  //cv::imshow("im1", error);
  //cv::waitKey(0);*/



  ///*cv::Mat imgaaaaa;
  //cv::Mat imgaaaa;
  //for (int i = 0; i < imgLeftPtsTransf.size(); i++)
  //{
  //  img1_warp.copyTo(imgaaaaa);
  //  img2_warp.copyTo(imgaaaa);
  //  cv::circle(imgaaaaa, imgLeftPtsTransf.at(i), 5, cv::Scalar(0, 0, 0));
  //  cv::circle(imgaaaa, imgRightPtsTransf.at(i), 5, cv::Scalar(0, 0, 0));
  //  cv::line(imgaaaa, cv::Point(0, -lines1.at(i)[2] / lines1.at(i)[1]), cv::Point(img_2.cols, -(lines1.at(i)[2] + lines1.at(i)[0] * img_2.cols) / lines1.at(i)[1]), cv::Scalar(0, 0, 0),2);
  //  cv::line(imgaaaaa, cv::Point(0, -lines2.at(i)[2] / lines2.at(i)[1]), cv::Point(img_2.cols, -(lines2.at(i)[2] + lines2.at(i)[0] * img_2.cols) / lines2.at(i)[1]), cv::Scalar(0, 0, 0),2);
  //  cv::imshow("im1", imgaaaaa);
  //  cv::imshow("im2", imgaaaa);
  //  cv::waitKey(0);
  //}*/

  //int windowSize = 5;//atoi(argv[4]);//2 5;//
  //int maxDist = 50;//atoi(argv[5]);//25 50;//
  //int x,y;
  //Point2f res;
  //std::vector<cv::Point2f> imgPtsDenseLeft, imgPtsDenseRight;
  //std::vector<cv::Vec3f> linesDense, linesDense2;
  //for (size_t i = 0; i < lines1.size(); i++)
  //{
  //  x = imgPtsTransf.at(i).x;
  //  y = imgPtsTransf.at(i).y;
  //  if ((x - windowSize) >= 0 && (x + windowSize) < img1_warp.cols && (y - windowSize) >= 0 && (y + windowSize) < img1_warp.rows)
  //  {
  //    res = ZNCC(img2_warp, img1_warp(cv::Range(y - windowSize, y + windowSize + 1), cv::Range(x - windowSize, x + windowSize + 1)), x, windowSize, maxDist, lines1.at(i));
  //    if (res.x != -1)
  //    {
  //      imgPtsDenseLeft.push_back(imgPtsTransf.at(i));
  //      imgPtsDenseRight.push_back(res);
  //      linesDense.push_back(lines1.at(i));
  //      linesDense2.push_back(lines2.at(i));
  //    }
  //  }
  //}

  //std::vector<cv::Point2f> imgPtsDenseLeftOriginal, imgPtsDenseRightOriginal;
  //perspectiveTransform(imgPtsDenseLeft, imgPtsDenseLeftOriginal, H1.inv());
  //perspectiveTransform(imgPtsDenseRight, imgPtsDenseRightOriginal, H2.inv());


  //if (drawDENSEepipolar)
  //{
  //  cv::Mat dst;
  //  hconcat(img1_warp, img2_warp, dst);
  //  cv::Mat imgConcatDraw;
  //  for (size_t i = 0; i < imgPtsDenseLeftOriginal.size(); i++)
  //  {
  //    dst.copyTo(imgConcatDraw);
  //    cv::circle(imgConcatDraw, imgPtsDenseLeft.at(i), 5, cv::Scalar(0, 0, 0), 2);
  //    cv::circle(imgConcatDraw, cv::Point(imgPtsDenseRight.at(i).x + img_1.cols, imgPtsDenseRight.at(i).y), 5, cv::Scalar(0, 0, 0), 2);
  //    //cv::line(imgConcatDraw, cv::Point(0, -linesDense.at(i)[2] / linesDense.at(i)[1]), cv::Point(dst.cols, -(linesDense.at(i)[2] + linesDense.at(i)[0] * dst.cols) / linesDense.at(i)[1]), cv::Scalar(0, 0, 0), 2);
  //    cv::line(imgConcatDraw, cv::Point(img1_warp.cols, -(linesDense.at(i)[2] + linesDense.at(i)[0] * img1_warp.cols) / linesDense.at(i)[1]), cv::Point(dst.cols, -(linesDense.at(i)[2] + linesDense.at(i)[0] * dst.cols) / linesDense.at(i)[1]), cv::Scalar(0, 0, 0), 2);
  //    cv::line(imgConcatDraw, cv::Point(0, -linesDense2.at(i)[2] / linesDense2.at(i)[1]), cv::Point(img1_warp.cols, -(linesDense2.at(i)[2] + linesDense2.at(i)[0] * img1_warp.cols) / linesDense2.at(i)[1]), cv::Scalar(0, 0, 0), 2);
  //    cv::imshow("im1", imgConcatDraw);
  //    cv::waitKey(0);
  //  }
  //}


  //for (size_t i = 0; i < imgPtsDenseLeftOriginal.size(); i++)
  //{
  //  cv::circle(img_1, imgPtsDenseLeftOriginal.at(i), 5, cv::Scalar(0, 0, 0));
  //  cv::circle(img_2, imgPtsDenseRightOriginal.at(i), 5, cv::Scalar(0, 0, 0));
  //  cv::circle(img1_warp, imgPtsDenseLeft.at(i), 5, cv::Scalar(0, 0, 0));
  //  cv::circle(img2_warp, imgPtsDenseRight.at(i), 5, cv::Scalar(0, 0, 0));
  //  //cv::line(img2_warp, cv::Point(0, -linesDense.at(i)[2] / linesDense.at(i)[1]), cv::Point(img_2.cols, -(linesDense.at(i)[2] + linesDense.at(i)[0] * img_2.cols) / linesDense.at(i)[1]), cv::Scalar(0, 0, 0));
  //}
  //cv::imshow("im1", img_1);
  //cv::imshow("im2", img_2);
  //cv::imshow("im1wrap", img1_warp);
  //cv::imshow("im2wrap", img2_warp);
  //cv::waitKey(0);

  //p->computeDensePointCloud(imgPtsDenseLeftOriginal, imgPtsDenseRightOriginal);
  //p->savePointCloud(".\corujaPoint.ply", 1);

  /*vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
  vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
  renderWindow->AddRenderer(renderer);
  vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
  vtkSmartPointer<vtkInteractorStyleTrackballCamera> style = vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New(); //like paraview
  renderWindowInteractor->SetInteractorStyle(style);
  renderWindowInteractor->SetRenderWindow(renderWindow);

  //Draw::createPointCloud(renderer, p->pointCloud);
  Draw::createPointCloud(renderer, p->densePointCloud);
  Draw::createCamera(renderer, img_1.cols, img_1.rows, K.at<double>(0, 0), (cv::Mat)p->Pleft, img_1);
  Draw::createCamera(renderer, img_1.cols, img_1.rows, K.at<double>(0, 0), (cv::Mat)p->Pright, img_2);
  //Draw::createCamera(renderer, img_1.cols, img_1.rows, K.at<double>(0, 0), (cv::Mat)Pright, img_3);

  renderWindow->Render();
  renderWindowInteractor->Start();*/

  return 0;
}

