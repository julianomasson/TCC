#include "stdafx.h"
#include "Keypoint.h"

Keypoint::Keypoint(cv::KeyPoint keypoint, Camera * cam)
{
  this->keypoint = keypoint;
  this->cam = cam;
}

Keypoint::Keypoint(SiftGPU::SiftKeypoint keypoint, Camera * cam)
{
  useSiftGPU = true;
  this->keypointSiftGPU = keypoint;
  this->cam = cam;
}

bool Keypoint::isEqual(Keypoint * k)
{ 
  if (this->cam != k->cam)
  {
    return false;
  }
  float dist = 1e-10;
  cv::Vec2f dx = this->getPoint() - k->getPoint();
  return (sqrtf(dx.dot(dx)) < dist);
}

double Keypoint::getReprojectionError(cv::Mat point3D)
{
  cv::Mat rvecLeft;
  Rodrigues(cam->P.get_minor<3, 3>(0, 0), rvecLeft);
  cv::Mat tvecLeft(cam->P.get_minor<3, 1>(0, 3).t());

  std::vector<cv::Point2f> projectedOnLeft(1);
  projectPoints(point3D, rvecLeft, tvecLeft, cam->K, cv::Mat(), projectedOnLeft);

  return cv::norm(projectedOnLeft[0] - this->getPoint());
}

cv::Point2f Keypoint::getPoint()
{
  if (pt.x == -1 && pt.y == -1)
  {
    if (useSiftGPU)
    {
      pt = cv::Point2f(keypointSiftGPU.x, keypointSiftGPU.y);
    }
    else
    {
      pt = keypoint.pt;
    }
  }
  return pt;
}
