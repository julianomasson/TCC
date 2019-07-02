#include "Pair.h"

Pair::Pair()
{
}

Pair::Pair(Camera * camLeft, Camera * camRight, cv::Mat K, float distanceNNDR)
{
  this->camLeft = camLeft;
  this->camRight = camRight;
  this->K = K;
  this->distanceNNDR = distanceNNDR;
}

Pair::~Pair()
{
  seeds.clear();
  cloudPCL = NULL;
  cloudPCLNormals = NULL;
  cloudPlusNormalsPCL = NULL;
}

void Pair::matchOpenCV()
{
  cv::Ptr<cv::DescriptorMatcher> matcher;
  if (camLeft->descriptors.type() != CV_32F) //AKAZE
  {
    matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
  }
  else//SIFT
  {
    matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
  }
  std::vector<std::vector< cv::DMatch > > matches;

  matcher->knnMatch(camLeft->descriptors, camRight->descriptors, matches, 2);

  for (size_t i = 0; i < matches.size(); i++)
  {
    if (matches.at(i).size() >= 2)
    {
      if (matches.at(i).at(0).distance < distanceNNDR*matches.at(i).at(1).distance)
      {
        good_matches.push_back(matches.at(i).at(0));
      }
    }
  }
  for (size_t i = 0; i < good_matches.size(); i++)
  {
    //-- Get the keypoints from the good matches
    goodKeypointsLeft.push_back(camLeft->keypoints[good_matches[i].queryIdx]);
    goodKeypointsRight.push_back(camRight->keypoints[good_matches[i].trainIdx]);
    imgLeftPts.push_back(camLeft->keypoints[good_matches[i].queryIdx].pt);
    imgRightPts.push_back(camRight->keypoints[good_matches[i].trainIdx].pt);
  }
  cv::Mat inliers;
  E = findEssentialMat(imgLeftPts, imgRightPts, K, CV_RANSAC, 0.99999, 1.0, inliers);

  std::vector<cv::Point2f> imgLeftPtsTemp;
  std::vector<cv::Point2f> imgRightPtsTemp;
  std::vector<cv::KeyPoint> goodKeypointsLeftTemp;
  std::vector<cv::KeyPoint> goodKeypointsRightTemp;
  for (size_t i = 0; i < inliers.rows; i++)
  {
    if (inliers.at<uchar>(i, 0) == 1)
    {
      goodKeypointsLeftTemp.push_back(goodKeypointsLeft.at(i));
      goodKeypointsRightTemp.push_back(goodKeypointsRight.at(i));
      imgLeftPtsTemp.push_back(imgLeftPts.at(i));
      imgRightPtsTemp.push_back(imgRightPts.at(i));
    }
  }
  goodKeypointsLeft = goodKeypointsLeftTemp;
  goodKeypointsRight = goodKeypointsRightTemp;
  imgLeftPts = imgLeftPtsTemp;
  imgRightPts = imgRightPtsTemp;
}

void Pair::matchSiftGPU()
{
  int maxMatches = std::min(camLeft->keypointsSiftGPU.size(), camRight->keypointsSiftGPU.size());
  if (maxMatches > numMaxMatches)
  {
    maxMatches = numMaxMatches;
  }
  SiftMatchGPU matcher(maxMatches);

  if (matcher.VerifyContextGL() == 0)
  {
    return;
  }

  matcher.SetDescriptors(0, camLeft->keypointsSiftGPU.size(), &camLeft->descriptorsSiftGPU[0]);
  matcher.SetDescriptors(1, camRight->keypointsSiftGPU.size(), &camRight->descriptorsSiftGPU[0]);


  int(*match_buf)[2] = new int[maxMatches][2];
  //use the default thresholds. Check the declaration in SiftGPU.h
  int num_match = matcher.GetSiftMatch(maxMatches, match_buf);
  std::cout << num_match << " sift matches were found;\n";

  //enumerate all the feature matches
  for (int i = 0; i < num_match; ++i)
  {
    //How to get the feature matches: 
    goodKeypointsLeftSiftGPU.push_back(camLeft->keypointsSiftGPU[match_buf[i][0]]);
    goodKeypointsRightSiftGPU.push_back(camRight->keypointsSiftGPU[match_buf[i][1]]);
    imgLeftPts.push_back(cv::Point2f(goodKeypointsLeftSiftGPU.back().x, goodKeypointsLeftSiftGPU.back().y));
    imgRightPts.push_back(cv::Point2f(goodKeypointsRightSiftGPU.back().x, goodKeypointsRightSiftGPU.back().y));
  }

  cv::Mat inliers;
  E = findEssentialMat(imgLeftPts, imgRightPts, K, CV_RANSAC, 0.99999, 1.0, inliers);

  std::vector<cv::Point2f> imgLeftPtsTemp;
  std::vector<cv::Point2f> imgRightPtsTemp;
  std::vector<SiftGPU::SiftKeypoint> goodKeypointsLeftTemp;
  std::vector<SiftGPU::SiftKeypoint> goodKeypointsRightTemp;
  for (size_t i = 0; i < inliers.rows; i++)
  {
    if (inliers.at<uchar>(i, 0) == 1)
    {
      goodKeypointsLeftTemp.push_back(goodKeypointsLeftSiftGPU.at(i));
      goodKeypointsRightTemp.push_back(goodKeypointsRightSiftGPU.at(i));
      imgLeftPtsTemp.push_back(imgLeftPts.at(i));
      imgRightPtsTemp.push_back(imgRightPts.at(i));
    }
  }
  goodKeypointsLeftSiftGPU = goodKeypointsLeftTemp;
  goodKeypointsRightSiftGPU = goodKeypointsRightTemp;
  imgLeftPts = imgLeftPtsTemp;
  imgRightPts = imgRightPtsTemp;
  //std::cout << "Matches " << imgLeftPts.size() << std::endl;
}

void Pair::computePose()
{
  cv::Mat R, t;
  recoverPose(E, imgLeftPts, imgRightPts, K, R, t);

  //TODO: stratify over Pleft
  Pright = cv::Matx34f(R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0),
    R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1),
    R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2));
  camRight->P = Pright;
}

void Pair::createTracks()
{
  if (goodKeypointsLeft.size() != 0)
  {
    size_t numMatches = goodKeypointsLeft.size();
    for (size_t i = 0; i < numMatches; i++)
    {
      tracks.push_back(new Track());
      tracks.back()->addKeypoint(new Keypoint(goodKeypointsLeft.at(i), camLeft));
      tracks.back()->addKeypoint(new Keypoint(goodKeypointsRight.at(i), camRight));
    }
  }
  else if (goodKeypointsLeftSiftGPU.size() != 0)
  {
    size_t numMatches = goodKeypointsLeftSiftGPU.size();
    for (size_t i = 0; i < numMatches; i++)
    {
      tracks.push_back(new Track());
      tracks.back()->addKeypoint(new Keypoint(goodKeypointsLeftSiftGPU.at(i), camLeft));
      tracks.back()->addKeypoint(new Keypoint(goodKeypointsRightSiftGPU.at(i), camRight));
    }
  }
}

template<typename T>
void
fundamentalFromProjections(const cv::Mat_<T> &P1,
  const cv::Mat_<T> &P2,
  cv::Mat_<T> F)
{
  cv::Mat_<T> X[3];
  cv::vconcat(P1.row(1), P1.row(2), X[0]);
  cv::vconcat(P1.row(2), P1.row(0), X[1]);
  cv::vconcat(P1.row(0), P1.row(1), X[2]);

  cv::Mat_<T> Y[3];
  cv::vconcat(P2.row(1), P2.row(2), Y[0]);
  cv::vconcat(P2.row(2), P2.row(0), Y[1]);
  cv::vconcat(P2.row(0), P2.row(1), Y[2]);

  cv::Mat_<T> XY;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
    {
      cv::vconcat(X[j], Y[i], XY);
      F(i, j) = cv::determinant(XY);
    }
}

void
fundamentalFromProjections(cv::InputArray _P1,
  cv::InputArray _P2,
  cv::OutputArray _F)
{
  const cv::Mat P1 = _P1.getMat(), P2 = _P2.getMat();
  const int depth = P1.depth();
  CV_Assert((P1.cols == 4 && P1.rows == 3) && P1.rows == P2.rows && P1.cols == P2.cols);
  CV_Assert((depth == CV_32F || depth == CV_64F) && depth == P2.depth());

  _F.create(3, 3, depth);

  cv::Mat F = _F.getMat();

  // type
  if (depth == CV_32F)
  {
    fundamentalFromProjections<float>(P1, P2, F);
  }
  else
  {
    fundamentalFromProjections<double>(P1, P2, F);
  }

}

cv::Point3f multH(cv::Mat H, cv::Point3f p)
{
  cv::Point3f pR;
  pR.x = H.at<double>(0, 0)*p.x + H.at<double>(0, 1)*p.y + H.at<double>(0, 2)*p.z;
  pR.y = H.at<double>(1, 0)*p.x + H.at<double>(1, 1)*p.y + H.at<double>(1, 2)*p.z;
  pR.z = H.at<double>(2, 0)*p.x + H.at<double>(2, 1)*p.y + H.at<double>(2, 2)*p.z;
  pR.x = pR.x / pR.z;
  pR.y = pR.y / pR.z;
  pR.z = pR.z / pR.z;
  return pR;
}

cv::Mat shearingTransform(double w, double h, cv::Mat H1, cv::Mat H2)
{
  cv::Point3f a = cv::Point3f((w - 1) / 2.0, 0, 1);
  cv::Point3f b = cv::Point3f(w - 1, (h - 1) / 2.0, 1);
  cv::Point3f c = cv::Point3f((w - 1) / 2.0, h - 1, 1);
  cv::Point3f d = cv::Point3f(0, (h - 1) / 2.0, 1);

  cv::Point3f a2 = multH(H1, a);
  cv::Point3f b2 = multH(H1, b);
  cv::Point3f c2 = multH(H1, c);
  cv::Point3f d2 = multH(H1, d);

  cv::Point3f xS = b2 - d2;
  cv::Point3f yS = c2 - a2;

  double k1 = (h*h*xS.y*xS.y + w*w*yS.y*yS.y) / (h*w*(xS.y*yS.x - xS.x*yS.y));
  double k2 = (h*h*xS.x*xS.y + w*w*yS.x*yS.y) / (h*w*(xS.x*yS.y - xS.y*yS.x));

  if (k1 < 0)
  {
    k1 *= -1;
    k2 *= -1;
  }
  return (cv::Mat_<double>(3, 3) << k1, k2, 0, 0, 1, 0, 0, 0, 1);
}

void Pair::computeRectify()
{
  cv::Mat H1, H2, Pn1, Pn2;
  Utils::rectifyP(camLeft->P, camRight->P, camLeft->K, H1, H2, Pn1, Pn2);

  /*cv::Mat bb1 = Utils::findBB(H1, camLeft->img.size());
  cv::Mat A = cv::Mat::eye(3, 3, CV_64F); A.at<double>(0, 2) = -bb1.at<double>(0, 0); A.at<double>(1, 2) = -bb1.at<double>(1, 0);
  newH1 = A*H1;
  cv::warpPerspective(camLeft->img, img1Warp, newH1, cv::Size(bb1.at<double>(2, 0) - bb1.at<double>(0, 0), bb1.at<double>(3, 0) - bb1.at<double>(1, 0)));

  cv::Mat bb2 = Utils::findBB(H2, camRight->img.size());
  A.at<double>(0, 2) = -bb2.at<double>(0, 0); A.at<double>(1, 2) = -bb2.at<double>(1, 0);
  newH2 = A*H2;
  cv::warpPerspective(camRight->img, img2Warp, newH2, cv::Size(bb2.at<double>(2, 0) - bb2.at<double>(0, 0), bb2.at<double>(3, 0) - bb2.at<double>(1, 0)));*/

  /*cv::Mat S = shearingTransform(camLeft->img.cols, camLeft->img.rows, H1, H2);
  H1 = S * H1;
  H2 = S * H2;*/

  Utils::imRectify(camLeft->img, camRight->img, H1, H2, img1Warp, img2Warp, img1Mask, img2Mask, newH1, newH2);

  //cv::Mat imgConcat2;
  //hconcat(img1Warp, img2Warp, imgConcat2);
  //cv::namedWindow("Pair", CV_WINDOW_NORMAL);
  //cv::imwrite("img1Warp.jpg", img1Warp);
  //cv::imwrite("img2Warp.jpg", img2Warp);
  //cv::waitKey(0);

}

void Pair::computeDenseSIFT()
{
  cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SIFT::create();
  f2d->detectAndCompute(img1Warp, cv::noArray(), keypoints1, descriptors1);
  f2d->detectAndCompute(img2Warp, cv::noArray(), keypoints2, descriptors2);
}

void Pair::matchDenseSIFT()
{
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
  std::vector<std::vector< cv::DMatch > > matches;

  matcher->knnMatch(descriptors1, descriptors2, matches, 2);

  good_matches.clear();
  for (size_t i = 0; i < matches.size(); i++)
  {
    if (matches.at(i).size() >= 2)
    {
      if (matches.at(i).at(0).distance < distanceNNDR*matches.at(i).at(1).distance)
      {
        good_matches.push_back(matches.at(i).at(0));
      }
    }
  }
  goodKeypointsLeft.clear();
  goodKeypointsRight.clear();
  imgLeftPts.clear();
  imgRightPts.clear();
  //std::vector< cv::DMatch > aaamatches;
  for (size_t i = 0; i < good_matches.size(); i++)
  {
    //-- Get the keypoints from the good matches
    if (abs(keypoints1[good_matches[i].queryIdx].pt.y - keypoints2[good_matches[i].trainIdx].pt.y) < 10)//Epipolar constraint
    {
      goodKeypointsLeft.push_back(keypoints1[good_matches[i].queryIdx]);
      goodKeypointsRight.push_back(keypoints2[good_matches[i].trainIdx]);
      //aaamatches.push_back(good_matches[i]);
      imgLeftPts.push_back(keypoints1[good_matches[i].queryIdx].pt);
      imgRightPts.push_back(keypoints2[good_matches[i].trainIdx].pt);
      img1Mask.at<uchar>(imgLeftPts.back()) = 0;
      img2Mask.at<uchar>(imgRightPts.back()) = 0;
    }
  }

  //cv::Mat outimg;
  //cv::drawMatches(img1Warp, keypoints1, img2Warp, keypoints2, aaamatches, outimg);

  //cv::namedWindow("Pair", CV_WINDOW_NORMAL);
  //cv::namedWindow("mask", CV_WINDOW_NORMAL);
  ////cv::Mat imgConcat;
  ////hconcat(img1Warp, img2Warp, imgConcat);
  ////cv::Mat imgConcat2;
  ////hconcat(img1Mask, img2Mask, imgConcat2);
  ////cv::Mat imgConcatDraw;
  ////imgConcat.copyTo(imgConcatDraw);
  ////cv::cvtColor(imgConcatDraw, imgConcatDraw, cv::COLOR_GRAY2BGR);
  //cv::Mat imgConcatDraw1, imgConcatDraw2;
  //cv::cvtColor(img1Warp, imgConcatDraw1, cv::COLOR_GRAY2BGR);
  //cv::cvtColor(img2Warp, imgConcatDraw2, cv::COLOR_GRAY2BGR);
  //int cont = 0;
  //for (size_t i = 0; i < imgLeftPts.size(); i++)
  //{
  //  cv::circle(imgConcatDraw1, imgLeftPts.at(i), 2, cv::Scalar(255, 0, 0), 2);
  //  cv::circle(imgConcatDraw2, imgRightPts.at(i), 2, cv::Scalar(255, 0, 0), 2);
  //  //cv::circle(imgConcatDraw, cv::Point(imgRightPts.at(i).x + img1Warp.cols, imgRightPts.at(i).y), 2, cv::Scalar(255, 0, 0), 2);
  //  //if (cont >= 10)
  //  //{
  //  //  cv::line(imgConcatDraw, imgLeftPts.at(i), cv::Point(imgRightPts.at(i).x + img1Warp.cols, imgRightPts.at(i).y), cv::Scalar(255,0,0), 2);
  //  //  cont = 0;
  //  //}
  //  //cont++;
  //}
  //cv::imshow("Pair", imgConcatDraw1);
  //cv::imwrite("matches1.jpg", imgConcatDraw1);
  //cv::imwrite("matches2.jpg", imgConcatDraw2);
  ////cv::imshow("mask", imgConcat2);
  //cv::waitKey(0);
}

void Pair::computeDenseDAISY()
{
  std::vector<cv::Point2f> pts;
  const float keypoint_diameter = 15.0f;
  // Add every pixel to the list of keypoints for each image
  for (float xx = keypoint_diameter; xx < 500 - keypoint_diameter; xx++) {
    for (float yy = keypoint_diameter; yy < 500 - keypoint_diameter; yy++) {
      //keypoints1.push_back(cv::KeyPoint(xx, yy, keypoint_diameter));
      //keypoints2.push_back(cv::KeyPoint(xx, yy, keypoint_diameter));
      pts.push_back(cv::Point2f(xx, yy));
    }
  }
  std::vector<cv::Point2f> ptsTrans1;
  std::vector<cv::Point2f> ptsTrans2;

  cv::perspectiveTransform(pts, ptsTrans1, newH1);
  cv::perspectiveTransform(pts, ptsTrans2, newH2);

  newH1 = newH1.inv();
  newH2 = newH2.inv();

  {
    keypoints1.push_back(cv::KeyPoint(ptsTrans1.at(i), keypoint_diameter));
    keypoints2.push_back(cv::KeyPoint(ptsTrans2.at(i), keypoint_diameter));
  }

  cv::Ptr<cv::xfeatures2d::DAISY> descriptor_extractor = cv::xfeatures2d::DAISY::create();
  // Compute DAISY descriptors for both images 
  descriptor_extractor->compute(img1Warp, keypoints1, descriptors1);
  descriptor_extractor->compute(img2Warp, keypoints2, descriptors2);
}

void Pair::matchDenseDAISY()
{
  std::vector <std::vector<cv::DMatch>> matches;

  // For each descriptor in image1, find 2 closest matched in image2 (note: couldn't get BF matcher to work here at all)
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
  matcher->add(descriptors1);
  matcher->train();
  matcher->knnMatch(descriptors2, matches, 2);


  // ignore matches with high ambiguity -- i.e. second closest match not much worse than first
  // push all remaining matches back into DMatch Vector "good_matches" so we can draw them using DrawMatches
  int                 num_good = 0;
  std::vector<cv::KeyPoint>    matched1, matched2;
  std::vector<cv::DMatch>      good_matches;

  for (int i = 0; i < matches.size(); i++) {
    cv::DMatch first = matches[i][0];
    cv::DMatch second = matches[i][1];

    if (first.distance < distanceNNDR * second.distance) {
      matched1.push_back(keypoints1[first.trainIdx]);
      matched2.push_back(keypoints2[first.queryIdx]); 
      good_matches.push_back(cv::DMatch(num_good, num_good, 0));
      num_good++;
    }
  }

  cv::Mat res;
  cv::drawMatches(img1Warp, matched1, img2Warp, matched2, good_matches, res);
  imwrite("_res.png", res);
}

void Pair::createInitialSeeds()
{
  for (size_t i = 0; i < imgLeftPts.size(); i++)
  {
    seeds.push_back(new Seed(imgLeftPts.at(i), imgRightPts.at(i), camLeft, camRight, 1.0));
  }
}

bool sortByScore(Seed* i, Seed* j)
{
  return (i->score<j->score);
}
std::string Pair::computeNewSeeds()
{
  std::stringstream log;
  int windowSize = 6;
  int windowSize2 = windowSize + 1;
  int searchSeed = 5;
  int maxDist = 5;
  log << "windowSize: " << windowSize << " searchSeed: " << searchSeed << " maxDist: " << maxDist << "\n";
  double score = 0;
  int sizeSeeds;
  std::mutex mut;
  std::vector<Seed*> newSeeds;
  std::vector<Seed*> usedSeeds;
  std::vector<Seed*> seedsTemp;
  DWORD initial_time;
  for (size_t i = 0; i < 20; i++)
  {
    sizeSeeds = seeds.size();
    std::cout << sizeSeeds << std::endl;
    log << i << " - " << sizeSeeds << "\n";
    initial_time = GetTickCount();
    cv::parallel_for_(cv::Range(0, sizeSeeds), [&](const cv::Range& range) {
      for (int r = range.start; r < range.end; r++)
      {
        int x_P1 = seeds.at(r)->p1.x;
        int y_P1 = seeds.at(r)->p1.y;
        int minX = x_P1 - searchSeed;
        int maxX = x_P1 + searchSeed;
        int minY = y_P1 - searchSeed;
        int maxY = y_P1 + searchSeed;
        int minX2 = seeds.at(r)->p2.x - searchSeed;
        int x2 = 0;
        for (int x = minX; x <= maxX; x++, x2++)
        {
          for (int y = minY; y <= maxY; y++)
          {
            if (y >= 0 && x >= 0 && y < img1Mask.rows && x < img1Mask.cols)
            {
              if (img1Mask.at<uchar>(y, x) != 0)
              {
                if ((x - windowSize) >= 0 && (x + windowSize2) < img1Warp.cols && (y - windowSize) >= 0 && (y + windowSize2) < img1Warp.rows)
                {
                  cv::Point2f pointFound = Utils::ZNCC(img2Warp, img1Warp(cv::Range(y - windowSize, y + windowSize2), cv::Range(x - windowSize, x + windowSize2)), minX2 + x2, windowSize, maxDist, y, img2Mask, score);
                  if (pointFound.x != -1)
                  {
                    mut.lock();
                    newSeeds.push_back(new Seed(cv::Point2f(x, y), pointFound, camLeft, camRight, score));
                    mut.unlock();
                  }
                }
              }
            }
          }
        }
      }
    });
    log << "parallel for time: " << Utils::calcTime(initial_time, GetTickCount()) << "\n";
    initial_time = GetTickCount();

    usedSeeds.insert(usedSeeds.end(), seeds.begin(), seeds.end());
    seeds.clear();
    std::sort(newSeeds.begin(), newSeeds.end(), sortByScore);

    for (size_t i = 0; i < newSeeds.size(); i++)
    {
      if (img2Mask.at<uchar>(newSeeds.at(i)->p2) != 0)
      {
        img1Mask.at<uchar>(newSeeds.at(i)->p1) = 0;
        img2Mask.at<uchar>(newSeeds.at(i)->p2) = 0;
        seedsTemp.push_back(newSeeds.at(i));
      }
      else
      {
        delete newSeeds.at(i);
        newSeeds.at(i) = NULL;
      }
    }
    log << "newSeed/usedSeeds " << newSeeds.size() << "/" << seedsTemp.size() << "\n";
    newSeeds.clear();
    seeds = seedsTemp;
    seedsTemp.clear();

    log << "cleaning vector seeds: " << Utils::calcTime(initial_time, GetTickCount()) << "\n";
  }
  seeds.clear();
  seeds = usedSeeds;
  usedSeeds.clear();
  std::vector<Seed*>(usedSeeds).swap(usedSeeds);
  newSeeds.clear();
  std::vector<Seed*>(newSeeds).swap(newSeeds);
  std::vector<Seed*>(seedsTemp).swap(seedsTemp);
  return log.str();
  //cv::namedWindow("Pair", CV_WINDOW_NORMAL);
  //cv::namedWindow("Mask", CV_WINDOW_NORMAL);
  //cv::Mat imgConcat;
  //hconcat(img1Mask, img2Mask, imgConcat);
  //cv::Mat imgConcat2;
  //hconcat(img1Warp, img2Warp, imgConcat2);
  //cv::imshow("Pair", imgConcat);
  //cv::imshow("Mask", imgConcat2);
  //cv::waitKey(0);
}

void Pair::compute3DPoints()
{
  cv::parallel_for_(cv::Range(0, seeds.size()), [&](const cv::Range& range) {
    for (int r = range.start; r < range.end; r++)
    {
      seeds.at(r)->calculatePoint3D(newH1, newH2);
    }
  });
}

void Pair::createDenseCloud(std::string path)
{
  
  std::ofstream myfile;
  myfile.open(path);
  cv::Point3f pTemp;
  for (size_t i = 0; i < seeds.size(); i++)
  {
    seeds.at(i)->calculatePoint3D(newH1, newH2);
    if (seeds.at(i)->reprojError < 1.0)
    {
      pTemp = seeds.at(i)->getPoint3D();
      myfile << "v " << pTemp.x << " " << pTemp.y << " " << pTemp.z << " 1.0 \n";
    }
  }
  myfile.close();

  //cv::namedWindow("Pair", CV_WINDOW_NORMAL);
  //cv::Mat imgConcat;
  //hconcat(camLeft->img, camRight->img, imgConcat);
  //cv::Mat imgConcatDraw;
  //imgConcat.copyTo(imgConcatDraw);
  //cv::cvtColor(imgConcatDraw, imgConcatDraw, cv::COLOR_GRAY2BGR);
  //for (size_t i = 0; i < seeds.size(); i++)
  //{
  //  cv::circle(imgConcatDraw, seeds.at(i)->p1Rect, 5, cv::Scalar(255, 0, 0), 5);
  //  cv::circle(imgConcatDraw, cv::Point(seeds.at(i)->p2Rect.x + camLeft->img.cols, seeds.at(i)->p2Rect.y), 5, cv::Scalar(255, 0, 0), 5);
  //}
  //cv::imshow("Pair", imgConcatDraw);
  //cv::waitKey(0);
}

void Pair::clearSeeds()
{
  cv::parallel_for_(cv::Range(0, seeds.size()), [&](const cv::Range& range) {
    for (int r = range.start; r < range.end; r++)
    {
      delete seeds.at(r);
      seeds.at(r) = NULL;
    }
  });
  seeds.clear();
  std::vector<Seed*>(seeds).swap(seeds);
}

void Pair::saveMasks(std::string path)
{
  cv::Mat imgConcat;
  hconcat(img1Mask, img2Mask, imgConcat);
  cv::Mat imgConcat2;
  hconcat(img1Warp, img2Warp, imgConcat2);
  cv::imwrite(path + "mask.jpg", imgConcat);
  cv::imwrite(path + "warp.jpg", imgConcat2);
}

void Pair::computeDepthMap()
{
  cv::Mat imgTeste(camLeft->img.size(), CV_32FC1, cv::Scalar(0, 0, 0));

  cv::parallel_for_(cv::Range(0, seeds.size()), [&](const cv::Range& range) {
    for (int r = range.start; r < range.end; r++)
    {
      if (seeds.at(r)->reprojError < 1.0)
      {
        imgTeste.at<float>(seeds.at(r)->p2Rect) = seeds.at(r)->getPoint3D().z;
      }
    }
  });
  //for (size_t i = 0; i < seeds.size(); i++)
  //{
  //  if (seeds.at(i)->reprojError < 1.0)
  //  {
  //    imgTeste.at<float>(seeds.at(i)->p2Rect) = seeds.at(i)->getPoint3D().z;
  //  }
  //}
  cv::parallel_for_(cv::Range(0, seeds.size()), [&](const cv::Range& range) {
    for (int r = range.start; r < range.end; r++)
    {
      if (seeds.at(r)->reprojError < 1.0)
      {
        int x = seeds.at(r)->p2Rect.x;
        int y = seeds.at(r)->p2Rect.y;
        if (x + 1 < imgTeste.cols && y + 1 < imgTeste.rows && x - 1 >= 0 && y - 1 >= 0)
        {
          if (imgTeste.at<float>(y, x + 1) == 0 || imgTeste.at<float>(y, x - 1) == 0 || imgTeste.at<float>(y + 1, x) == 0 || imgTeste.at<float>(y - 1, x) == 0)
          {
            seeds.at(r)->reprojError = 1000;
          }
          else
          {
            float dzdx = (imgTeste.at<float>(y, x + 1) - imgTeste.at<float>(y, x - 1)) / 2.0f;
            float dzdy = (imgTeste.at<float>(y + 1, x) - imgTeste.at<float>(y - 1, x)) / 2.0f;
            cv::Vec3f d(dzdx, dzdy, 0);
            seeds.at(r)->ptNormal = cv::normalize(d);
          }
        }
      }
    }
  });
  /*int x, y;
  float dzdx, dzdy;
  for (size_t i = 0; i < seeds.size(); i++)
  {
    if (seeds.at(i)->reprojError < 1.0)
    {
      x = seeds.at(i)->p2Rect.x;
      y = seeds.at(i)->p2Rect.y;
      if (x + 1 < imgTeste.cols && y + 1 < imgTeste.rows && x - 1 >= 0 && y - 1 >= 0)
      {
        if (imgTeste.at<float>(y, x + 1) == 0 || imgTeste.at<float>(y, x - 1) == 0 || imgTeste.at<float>(y + 1, x) == 0 || imgTeste.at<float>(y - 1, x) == 0)
        {
          seeds.at(i)->reprojError = 1000;
        }
        else
        {
          dzdx = (imgTeste.at<float>(y, x + 1) - imgTeste.at<float>(y, x - 1)) / 2.0f;
          dzdy = (imgTeste.at<float>(y + 1, x) - imgTeste.at<float>(y - 1, x)) / 2.0f;
          cv::Vec3f d(dzdx, dzdy, 0);
          seeds.at(i)->ptNormal = cv::normalize(d);
        }
      }
    }
  }*/
  /*cv::namedWindow("Pair", CV_WINDOW_NORMAL);
  cv::imshow("Pair", imgTeste);
  cv::waitKey(0);*/
  //cv::imwrite("disp.jpg", imgTeste);
}

void Pair::saveCloudNormals(std::string path)
{
  pcl::PointCloud<pcl::PointNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointNormal>);
  for (size_t i = 0; i < seeds.size(); i++)
  {
    if (seeds.at(i)->reprojError < 1.0)
    {
      pcl::PointNormal p;
      p.x = seeds.at(i)->getPoint3D().x;
      p.y = seeds.at(i)->getPoint3D().y;
      p.z = seeds.at(i)->getPoint3D().z;
      p.normal_x = seeds.at(i)->ptNormal[0];
      p.normal_y = seeds.at(i)->ptNormal[1];
      p.normal_z = seeds.at(i)->ptNormal[2];
      cloud->push_back(p);
    }
  }
  cloudPlusNormalsPCL = cloud;
  pcl::io::savePLYFileBinary(path.c_str(), *cloud);
}

void Pair::createPCLPointCloud()
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  cv::Point3f pTemp;
  for (size_t i = 0; i < seeds.size(); i++)
  {
    if (seeds.at(i)->reprojError < 1.0)
    {
      pTemp = seeds.at(i)->getPoint3D();
      cloud->push_back(pcl::PointXYZ(pTemp.x, pTemp.y, pTemp.z));
    }
  }
  cloudPCL = cloud;
}

void Pair::filterPCLCloud()
{
  pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
  outrem.setInputCloud(cloudPCL);
  outrem.setRadiusSearch(0.01);
  outrem.setMinNeighborsInRadius(50);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
  outrem.filter(*cloud_filtered);
  cloudPCL = cloud_filtered;
}

void Pair::computePCLNormal()
{
  // Create the normal estimation class, and pass the input dataset to it
  pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
  ne.setInputCloud(cloudPCL);

  // Create an empty kdtree representation, and pass it to the normal estimation object.
  // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
  ne.setSearchMethod(tree);

  // Output datasets
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);

  // Use all neighbors in a sphere of radius 3cm
  ne.setRadiusSearch(0.05);
  double* pView = Utils::createDoubleVector(0, 0, 0);
  Utils::transformPoint(pView, camRight->P);
  ne.setViewPoint(pView[0], pView[1], pView[2]);
  // Compute the features
  ne.compute(*cloud_normals);
  cloudPCLNormals = cloud_normals;
  delete pView;
}

void Pair::savePCLResult(std::string path)
{
  if (cloudPCLNormals)
  {
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointNormal>);
    for (size_t i = 0; i < cloudPCL->size(); i++)
    {
      pcl::PointNormal p;
      p.x = cloudPCL->at(i).x;
      p.y = cloudPCL->at(i).y;
      p.z = cloudPCL->at(i).z;
      p.normal_x = cloudPCLNormals->at(i).normal_x;
      p.normal_y = cloudPCLNormals->at(i).normal_y;
      p.normal_z = cloudPCLNormals->at(i).normal_z;
      cloud->push_back(p);
    }
    cloudPlusNormalsPCL = cloud;
    pcl::io::savePLYFileBinary(path.c_str(), *cloud);
  }
  else if (cloudPCL)
  {
    pcl::io::savePLYFileBinary(path.c_str(), *cloudPCL);
  }
}

size_t Pair::drawMatches()
{
  cv::Mat imgConcat;
  hconcat(camLeft->img, camRight->img, imgConcat);
  cv::cvtColor(imgConcat, imgConcat, cv::COLOR_GRAY2BGR);
  cv::namedWindow("Matches", CV_WINDOW_NORMAL);
  cv::Scalar color;
  for (size_t i = 0; i < imgLeftPts.size(); i++)
  {
    color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
    cv::circle(imgConcat, imgLeftPts.at(i), 5, color, 2);
    cv::circle(imgConcat, cv::Point(imgRightPts.at(i).x + camLeft->img.cols, imgRightPts.at(i).y), 5, color, 2);
  }
  cv::imshow("Matches", imgConcat);
  cv::waitKey(0);
  return imgLeftPts.size();
}
