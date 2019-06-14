#include "stdafx.h"
#include "Track.h"

Track::Track()
{
}

void Track::addKeypoint(Keypoint * k)
{
  keypoints.push_back(k);
}

bool Track::mergeTrack(Track * k)
{
  std::vector<Keypoint*> newKeypoints;
  bool isNew = true;
  bool change = false;
  for (size_t i = 0; i < k->keypoints.size(); i++)
  {
    isNew = true;
    for (size_t j = 0; j < keypoints.size(); j++)
    {
      if (k->keypoints.at(i)->isEqual(keypoints.at(j)))
      {
        isNew = false;
        change = true;
        break;
      }
    }
    if (isNew)
    {
      newKeypoints.push_back(k->keypoints.at(i));
    }
  }
  if (change)
  {
    for (size_t i = 0; i < newKeypoints.size(); i++)
    {
      keypoints.push_back(newKeypoints.at(i));
    }
  }
  return change;
}

cv::Point3f Track::getPoint3D()
{
  return pt3D;
}

void Track::calculatePoint3D()
{
  std::vector<cv::Mat> points3D;
  cv::Mat KLeft, KRight;
  Camera* camLeft, *camRight;
  for (size_t i = 0; i < keypoints.size(); i++)
  {
    std::vector<cv::Point2f> imgLeftPts;
    imgLeftPts.push_back(keypoints.at(i)->getPoint());
    KLeft = keypoints.at(i)->cam->K;
    cv::Mat normalizedLeftPts;
    undistortPoints(imgLeftPts, normalizedLeftPts, KLeft, cv::Mat());
    camLeft = keypoints.at(i)->cam;
    for (size_t j = i + 1; j < keypoints.size(); j++)
    {
      if (camLeft->getAngleBetweenCameras(keypoints.at(j)->cam) > 10)
      {
        std::vector<cv::Point2f> imgRightPts;
        imgRightPts.push_back(keypoints.at(j)->getPoint());
        KRight = keypoints.at(j)->cam->K;
        cv::Mat normalizedRightPts;
        undistortPoints(imgRightPts, normalizedRightPts, KRight, cv::Mat());
        camRight = keypoints.at(j)->cam;
        cv::Mat points3dHomogeneous;
        triangulatePoints(camLeft->P, camRight->P, normalizedLeftPts, normalizedRightPts, points3dHomogeneous);
        cv::Mat points3d;
        convertPointsFromHomogeneous(points3dHomogeneous.t(), points3d);
        points3D.push_back(points3d);
      }
    }
  }
  double sumReproj;
  size_t idxMinReproj = -1;
  double minReproj = VTK_DOUBLE_MAX;
  for (size_t i = 0; i < points3D.size(); i++)
  {
    sumReproj = 0;
    for (size_t j = 0; j < keypoints.size(); j++)
    {
      sumReproj += keypoints.at(j)->getReprojectionError(points3D.at(i));
    }
    sumReproj /= keypoints.size();
    if (sumReproj < minReproj)
    {
      minReproj = sumReproj;
      idxMinReproj = i;
    }
  }
  if (minReproj < lastReprojError && minReproj < 100)
  {
    float x = abs(points3D.at(idxMinReproj).at<float>(0, 0));
    float y = abs(points3D.at(idxMinReproj).at<float>(0, 1));
    float z = abs(points3D.at(idxMinReproj).at<float>(0, 2));
    if (x < 10 && y < 10 && z < 10)
    {
      pt3D = cv::Point3f(points3D.at(idxMinReproj).at<float>(0, 0), points3D.at(idxMinReproj).at<float>(0, 1), points3D.at(idxMinReproj).at<float>(0, 2));
      lastReprojError = minReproj;
    }
  }
  /*std::vector<cv::Point2f> imgLeftPts;
  std::vector<cv::Point2f> imgRightPts;

  imgLeftPts.push_back(keypoints.at(0)->keypoint.pt);
  imgRightPts.push_back(keypoints.at(1)->keypoint.pt);

  cv::Mat KLeft = keypoints.at(0)->cam->K;
  cv::Mat KRight = keypoints.at(1)->cam->K;


  cv::Mat normalizedLeftPts;
  cv::Mat normalizedRightPts;
  undistortPoints(imgLeftPts, normalizedLeftPts, KLeft, cv::Mat());
  undistortPoints(imgRightPts, normalizedRightPts, KRight, cv::Mat());

  Camera* camLeft = keypoints.at(0)->cam;
  Camera* camRight = keypoints.at(1)->cam;

  cv::Mat points3dHomogeneous;
  triangulatePoints(camLeft->P, camRight->P, normalizedLeftPts, normalizedRightPts, points3dHomogeneous);

  cv::Mat points3d;
  convertPointsFromHomogeneous(points3dHomogeneous.t(), points3d);*/

  //Reprojection

  /*cv::Mat rvecLeft;
  Rodrigues(camLeft->P.get_minor<3, 3>(0, 0), rvecLeft);
  cv::Mat tvecLeft(camLeft->P.get_minor<3, 1>(0, 3).t());

  std::vector<cv::Point2f> projectedOnLeft(imgLeftPts.size());
  projectPoints(points3d, rvecLeft, tvecLeft, KLeft, cv::Mat(), projectedOnLeft);

  cv::Mat rvecRight;
  Rodrigues(camRight->P.get_minor<3, 3>(0, 0), rvecRight);
  cv::Mat tvecRight(camRight->P.get_minor<3, 1>(0, 3).t());

  std::vector<cv::Point2f> projectedOnRight(imgRightPts.size());
  projectPoints(points3d, rvecRight, tvecRight, KRight, cv::Mat(), projectedOnRight);

  pt3D.x = -1;
  pt3D.y = -1;
  pt3D.z = -1;
  for (size_t i = 0; i < points3d.rows; i++) {
    //check if point reprojection error is small enough
    if (norm(projectedOnLeft[i] - imgLeftPts[i])  > REPROJ_ERROR || norm(projectedOnRight[i] - imgRightPts[i]) > REPROJ_ERROR)
    {
      continue;
    }
    if (points3d.at<float>(i, 2) > 0)//z>0
    {
      pt3D = cv::Point3f(points3d.at<float>(i, 0), points3d.at<float>(i, 1), points3d.at<float>(i, 2));
    }
  }*/
}

void Track::testPoint()
{
  if (abs(pt3D.x) > 10 || abs(pt3D.y) > 10 || abs(pt3D.z) > 10)
  {
    pt3D = cv::Point3f(-1, -1, -1);
    return;
  }
  float dummy_query_data[3] = { pt3D.x, pt3D.y, pt3D.z };
  cv::Mat pt3DMat = cv::Mat(1, 3, CV_32F, dummy_query_data);
  double minReproj = 0;
  for (size_t j = 0; j < keypoints.size(); j++)
  {
    minReproj += keypoints.at(j)->getReprojectionError(pt3DMat);
  }
  minReproj /= keypoints.size();
  if (minReproj > 100)
  {
    pt3D = cv::Point3f(-1, -1, -1);
  }
}
