#include "Camera.h"

Camera::Camera()
{
}

Camera::Camera(std::string pathImage)
{
  this->pathImage = pathImage;
  img = cv::imread(pathImage, CV_LOAD_IMAGE_GRAYSCALE);
  double f = 1.2*std::max(img.cols, img.rows);
  this->K = (cv::Mat_<double>(3, 3) << f, 0, img.cols / 2, 0, f, img.rows / 2, 0, 0, 1);
}

Camera::~Camera()
{
}

std::string Camera::getStringSFM()
{
  std::stringstream out;
  out << pathImage << " ";
  for (size_t j = 0; j < P.rows; j++)
  {
    for (size_t k = 0; k < (P.cols - 1); k++)
    {
      out << P(j, k) << " ";
    }
  }
  for (size_t i = 0; i < P.rows; i++)
  {
    out << P(i, 3) << " ";
  }
  out << K.at<double>(0,0) << " " << K.at<double>(1, 1) << "\n";
  return out.str();
}

std::string Camera::getStringNVM()
{
  std::stringstream out;
  out << pathImage << " ";
  out << K.at<double>(0, 0) << " ";
  cv::Mat quat = Utils::mRot2Quat(P);
  for (size_t j = 0; j < quat.rows; j++)
  {
      out << quat.at<float>(j, 0) << " ";
  }
  cv::Mat center = Utils::getCenterVector(P);
  for (size_t i = 0; i < P.rows; i++)
  {
    out << center.at<double>(i, 0) << " ";
  }
  out << "0 0\n";
  return out.str();
}

void Camera::updateOrigin(cv::Matx34f PtoOrigin)
{
  cv::Mat R01 = cv::Mat_<double>(3, 3);
  Utils::getRotationMatrix(R01, PtoOrigin);
  cv::Mat R12 = cv::Mat_<double>(3, 3);
  Utils::getRotationMatrix(R12, P);
  cv::Mat t01 = cv::Mat_<double>(3, 1);
  Utils::getTranslationVector(t01, PtoOrigin);
  cv::Mat t12 = cv::Mat_<double>(3, 1);
  Utils::getTranslationVector(t12, P);

  for (size_t i = 0; i < 3; i++)
  {
    for (size_t j = 0; j < 3; j++)
    {
      P(i,j) = 0;
      for (size_t k = 0; k < 3; k++)
      {
        P(i, j) += R01.at<double>(i, k) * R12.at<double>(k, j);
      }
    }
  }
  for (size_t i = 0; i < 3; i++)
  {
    P(i, 3) = 0;
    for (size_t j = 0; j < 3; j++)
    {
      P(i, 3) += R01.at<double>(i, j) * t12.at<double>(j, 0);
    }
  }
  for (size_t i = 0; i < 3; i++)
  {
    P(i, 3) = t01.at<double>(i, 0) + P(i, 3);
  }
}


void Camera::computeSiftGPUKeypoints()
{
  SiftGPU sift;
  char * argv[] = {"-fo","-1","-v","1"};
  sift.ParseParam(4,argv);

  int support = sift.CreateContextGL();

  if (support != SiftGPU::SIFTGPU_FULL_SUPPORTED)
  {
    return;
  }

  sift.RunSIFT(pathImage.c_str());
  int num = sift.GetFeatureNum();

  descriptorsSiftGPU.resize(128*num);
  keypointsSiftGPU.resize(num);

  sift.GetFeatureVector(&keypointsSiftGPU[0], &descriptorsSiftGPU[0]);
}

void Camera::computeOpenCVKeypoints(bool useAKAZE)
{
  if (useAKAZE)
  {
    cv::Ptr<cv::AKAZE> f2dAKAZE = cv::AKAZE::create();
    f2dAKAZE->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
  }
  else
  {
    cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SIFT::create();
    f2d->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
  }
}

void Camera::drawKeypoints()
{
  if (keypointsSiftGPU.size() > 0)
  {
    cv::namedWindow("SiftGPU", CV_WINDOW_NORMAL);
    cv::Mat siftGPU;
    this->img.copyTo(siftGPU);
    for (size_t i = 0; i < keypointsSiftGPU.size(); i++)
    {
      cv::circle(siftGPU, cv::Point2f(keypointsSiftGPU.at(i).x, keypointsSiftGPU.at(i).y), 5, cv::Scalar(0, 0, 0), 2);
    }
    cv::imshow("SiftGPU", siftGPU);
  }
  if (keypoints.size() > 0)
  {
    cv::namedWindow("SIFT", CV_WINDOW_NORMAL);
    cv::Mat sift;
    this->img.copyTo(sift);
    for (size_t i = 0; i < keypoints.size(); i++)
    {
      cv::circle(sift, keypoints.at(i).pt, 5, cv::Scalar(0, 0, 0), 2);
    }
    cv::imshow("SIFT", sift);
  }
  cv::waitKey(0);
}

void Camera::getViewVector(double * point)
{
  double* p1 = Utils::createDoubleVector(0, 0, 0); Utils::transformPoint(p1, this->P);
  double* p2 = Utils::createDoubleVector(0, 0, 0.5); Utils::transformPoint(p2, this->P);
  vtkMath::Subtract(p1, p2, point);
  delete p1, p2;
}

double Camera::getAngleBetweenCameras(Camera * cam)
{
  double* cam1 = new double[3];
  double* cam2 = new double[3];
  this->getViewVector(cam1);
  cam->getViewVector(cam2);
  double angle = vtkMath::DegreesFromRadians(vtkMath::AngleBetweenVectors(cam1, cam2));
  delete cam1, cam2;
  return angle;
}

double Camera::distanceBetweenOpticalCenter(Camera * cam)
{
  double* pcam1 = Utils::createDoubleVector(0, 0, 0); Utils::transformPoint(pcam1, this->P);
  double* pcam2 = Utils::createDoubleVector(0, 0, 0); Utils::transformPoint(pcam2, cam->P);
  double dist = sqrt(vtkMath::Distance2BetweenPoints(pcam1, pcam2));
  delete pcam1, pcam2;
  return dist;
}
