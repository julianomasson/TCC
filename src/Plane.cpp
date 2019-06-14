#include "stdafx.h"
#include "Plane.h"

Plane::Plane(Camera* cam_i, Camera* cam_j, int x, int y)
{
  this->cam_i = cam_i;
  this->cam_j = cam_j;

  p = (cv::Mat_<double>(3, 1) << x, y, 1);
  
  pt2D = cv::Point2f(x, y);

  lambda = LAMBDA_MIN + (std::rand() % (LAMBDA_MAX - LAMBDA_MIN + 1));

  X = lambda*this->cam_i->K.inv()*p;

  theta = (std::rand() % (60 + 1));

  phi = (std::rand() % (360 + 1));

  n = (cv::Mat_<double>(3, 1) << cos(theta)*sin(phi), sin(theta)*sin(phi), sin(phi));
}

Plane::~Plane()
{
}

double Plane::m()
{
  if (pt2D.x - WINDOW_SIZE < 0 || pt2D.x + WINDOW_SIZE + 1 >= cam_i->img.cols || pt2D.y - WINDOW_SIZE < 0 || pt2D.y + WINDOW_SIZE + 1 >= cam_i->img.rows)
  {
    return -1;
  }
  cv::Mat R_i = cv::Mat_<double>(3, 3);
  cv::Mat R_j = cv::Mat_<double>(3, 3);
  Utils::getRotationMatrix(R_i, cam_i->P);
  Utils::getRotationMatrix(R_j, cam_j->P);

  cv::Mat C_i = cv::Mat_<double>(3, 1);
  cv::Mat C_j = cv::Mat_<double>(3, 1);
  Utils::getTranslationVector(C_i, cam_i->P);
  Utils::getTranslationVector(C_j, cam_j->P);

  cv::Mat R_ij = R_j*R_i.inv();
  cv::Mat C_ij = R_ij*R_i*C_i - R_j*C_j;

  cv::Mat H_ij = this->cam_j->K*(R_ij + ( (C_ij*n.t()) / (n.t()*X) ) )*this->cam_i->K.inv();

  cv::Mat bb = Utils::findBB(H_ij, cam_i->img.size());

  cv::Mat A =cv:: Mat::eye(3, 3, CV_64F); A.at<double>(0, 2) = -bb.at<double>(0, 0); A.at<double>(1, 2) = -bb.at<double>(1, 0);
  cv::Mat F = A*H_ij;

  std::vector<cv::Point2f> pts;
  for (int i = -WINDOW_SIZE; i <= WINDOW_SIZE; i++)
  {
    for (int j = -WINDOW_SIZE; j <= WINDOW_SIZE; j++)
    {
      pts.push_back(cv::Point2f(pt2D.x + i, pt2D.y + j));
    }
  }
  cv::Mat imgwarp = Utils::getTransformedImage(cam_j->img, pts, F, WINDOW_SIZE * 2 + 1);


  //cv::Mat imgwarp;
  //cv::warpPerspective(cam_i->img, imgwarp, F, cv::Size(bb.at<double>(2, 0) - bb.at<double>(0, 0), bb.at<double>(3, 0) - bb.at<double>(1, 0)));

  //std::vector<cv::Point2f> ptsTransf;
  //std::vector<cv::Point2f> pts;
  //pts.push_back(pt2D);
  //cv::perspectiveTransform(pts, ptsTransf, F);


  //cv::circle(imgwarp, ptsTransf.at(0), 5, cv::Scalar(255,0,0),5);

  //cv::namedWindow("Pair", CV_WINDOW_NORMAL);
  //cv::namedWindow("trans", CV_WINDOW_NORMAL);
  //cv::imshow("Pair", imgwarp);
  //cv::waitKey(0);

  score = Utils::NCC(cam_i->img(cv::Range(pt2D.y - WINDOW_SIZE, pt2D.y + WINDOW_SIZE + 1), cv::Range(pt2D.x - WINDOW_SIZE, pt2D.x + WINDOW_SIZE + 1)), imgwarp);

  return 1 - score;
}
