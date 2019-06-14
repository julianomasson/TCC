#include "stdafx.h"
#include "Seed.h"

Seed::Seed()
{
}

Seed::Seed(cv::Point2f p1, cv::Point2f p2, Camera * cam1, Camera * cam2, double score)
{
  this->p1 = p1;
  this->p2 = p2;
  this->cam1 = cam1;
  this->cam2 = cam2;
  this->score = score;
}

Seed::~Seed()
{
}

cv::Point3f Seed::getPoint3D()
{
  return pt3D;
}

void Seed::calculatePoint3D(cv::Mat newH1, cv::Mat newH2)
{
  std::vector<cv::Point2f> imgPtsDenseLeft, imgPtsDenseRight;


  imgPtsDenseLeft.push_back(p1);
  imgPtsDenseRight.push_back(p2);


  std::vector<cv::Point2f> imgLeftPtsTransf, imgRightPtsTransf;
  cv::perspectiveTransform(imgPtsDenseLeft, imgLeftPtsTransf, newH1);
  cv::perspectiveTransform(imgPtsDenseRight, imgRightPtsTransf, newH2);

  p1Rect = imgLeftPtsTransf.at(0);
  if (p1Rect.x < 0)
  {
    p1Rect.x = 0;
  }
  if (p1Rect.x >= cam1->img.cols)
  {
    p1Rect.x = cam1->img.cols - 1;
  }
  if (p1Rect.y < 0)
  {
    p1Rect.y = 0;
  }
  if (p1Rect.y >= cam1->img.rows)
  {
    p1Rect.y = cam1->img.rows - 1;
  }
  p2Rect = imgRightPtsTransf.at(0);
  if (p2Rect.x < 0)
  {
    p2Rect.x = 0;
  }
  if (p2Rect.x >= cam2->img.cols)
  {
    p2Rect.x = cam2->img.cols - 1;
  }
  if (p2Rect.y < 0)
  {
    p2Rect.y = 0;
  }
  if (p2Rect.y >= cam2->img.rows)
  {
    p2Rect.y = cam2->img.rows - 1;
  }

  cv::Mat normalizedLeftPts;
  undistortPoints(imgLeftPtsTransf, normalizedLeftPts, cam1->K, cv::Mat());
  cv::Mat normalizedRightPts;
  undistortPoints(imgRightPtsTransf, normalizedRightPts, cam2->K, cv::Mat());

  cv::Mat points3dHomogeneous;
  triangulatePoints(cam1->P, cam2->P, normalizedLeftPts, normalizedRightPts, points3dHomogeneous);
  cv::Mat points3d;
  convertPointsFromHomogeneous(points3dHomogeneous.t(), points3d);

  reprojError = (getReprojectionError(points3d, cam1, imgLeftPtsTransf.at(0)) + getReprojectionError(points3d, cam2, imgRightPtsTransf.at(0))) / 2.0;
  pt3D = cv::Point3f(points3d.at<float>(0, 0), points3d.at<float>(0, 1), points3d.at<float>(0, 2));
}

double Seed::getReprojectionError(cv::Mat point3D, Camera* cam, cv::Point2f p)
{
  cv::Mat rvecLeft;
  Rodrigues(cam->P.get_minor<3, 3>(0, 0), rvecLeft);
  cv::Mat tvecLeft(cam->P.get_minor<3, 1>(0, 3).t());

  std::vector<cv::Point2f> projectedOnLeft(1);
  projectPoints(point3D, rvecLeft, tvecLeft, cam->K, cv::Mat(), projectedOnLeft);

  return cv::norm(projectedOnLeft[0] - p);
}