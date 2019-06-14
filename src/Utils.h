#pragma once

#include <vtkMath.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkCamera.h>
#include <vtkWindowToImageFilter.h>
#include <vtkPNGWriter.h>
#include <vtkRenderWindow.h>
#include <vtkMatrix4x4.h>
#include <vtkImageImport.h>
#include <opencv2\opencv.hpp>
#include <vtkMatrix3x3.h>
#include <Windows.h>

class Utils {
public:
  Utils();
  ~Utils();

  /*
  B
  |
  |
  |
  A------C
  static PointXYZ* Utils::getNormal(PointXYZ* a, PointXYZ* b, PointXYZ* c)
  {
  PointXYZ* ab = getVector(a, b);
  PointXYZ* ac = getVector(a, c);
  PointXYZ* n = new PointXYZ((ab->y*ac->z) - (ab->z*ac->y), (ab->z*ac->x) - (ab->x*ac->z), (ab->x*ac->y) - (ab->y*ac->x));
  return n;
  }*/

  /*
  True if the file exists, false otherwise
  */
  static bool exists(const std::string& name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
  }

  static std::string getMTLFilenameFromOBJ(std::string filename)
  {
    ifstream myfile(filename);
    std::string line;
    std::string mtlFile = "";
    if (myfile.is_open())
    {
      while (getline(myfile, line, '\n'))
      {
        if (line.find("mtllib ", 0) != std::string::npos)
        {
          mtlFile = line.substr(line.find(' ', 0) + 1);
          if (mtlFile.find("/", 0) != std::string::npos)
          {
            mtlFile = mtlFile.substr(mtlFile.find("/", 0) + 1);
          }
          break;
        }
        else if (line[0] == 'v')//the vertex list started, there is no mtllib
        {
          break;
        }
      }
    }
    return mtlFile;
  }

  /*
  P1---Midpoint---P2
  */
  static double* getMidpoint(double* p1, double* p2)
  {
    double* result = new double[3];
    for (int i = 0; i < 3; i++)
    {
      result[i] = (p1[i] + p2[i]) / 2;
    }
    return result;
  }
  static double* getMidpoint(double* p1, double* p2, double* p3)
  {
    double* result = new double[3];
    for (int i = 0; i < 3; i++)
    {
      result[i] = (p1[i] + p2[i] + p3[i]) / 3;
    }
    return result;
  }

  static void transformPoint(double* point, vtkSmartPointer<vtkMatrix4x4> matrixRT)
  {
    double x = (matrixRT->Element[0][0] * point[0] + matrixRT->Element[0][1] * point[1] + matrixRT->Element[0][2] * point[2] + matrixRT->Element[0][3]);
    double y = (matrixRT->Element[1][0] * point[0] + matrixRT->Element[1][1] * point[1] + matrixRT->Element[1][2] * point[2] + matrixRT->Element[1][3]);
    double z = (matrixRT->Element[2][0] * point[0] + matrixRT->Element[2][1] * point[1] + matrixRT->Element[2][2] * point[2] + matrixRT->Element[2][3]);
    point[0] = x; point[1] = y; point[2] = z;
  }
  static void transformPoint(double* point, cv::Matx34f matrixRT)
  {
    vtkSmartPointer<vtkMatrix4x4> matrixRT4x4 = vtkSmartPointer<vtkMatrix4x4>::New();
    for (size_t i = 0; i < 3; i++)
    {
      for (size_t j = 0; j < 4; j++)
      {
        matrixRT4x4->SetElement(i,j, matrixRT(i, j));

      }
    }
    matrixRT4x4->SetElement(3, 0, 0); matrixRT4x4->SetElement(3, 1, 0); matrixRT4x4->SetElement(3, 2, 0); matrixRT4x4->SetElement(3, 3, 1);
    matrixRT4x4->Invert();
    transformPoint(point, matrixRT4x4);
  }
  /*
  p[0] = x; p[1] = y; p[2] = z;
  */
  static double* createDoubleVector(double x, double y, double z)
  {
    double* p = new double[3];
    p[0] = x; p[1] = y; p[2] = z;
    return p;
  }
  static double* createDoubleVector(double* xyz)
  {
    double* p = new double[3];
    memcpy(p, xyz, sizeof(double) * 3);
    return p;
  }

  /*
  A
  *
  *
  *
  B * * * C, compute the normal and return the (normal+pointB) nearest to pointTest
  */
  static double* getNormal(double* pointA, double* pointB, double* pointC, double* pointTest)
  {
    double v1[3];
    double v2[3];
    double n[3];
    vtkMath::Subtract(pointB, pointA, v1);
    vtkMath::Subtract(pointB, pointC, v2);
    vtkMath::Normalize(v1);
    vtkMath::Normalize(v2);
    vtkMath::Cross(v1, v2, n);
    vtkMath::Normalize(n);
    double* n2 = Utils::createDoubleVector(n[0], n[1], n[2]);
    vtkMath::MultiplyScalar(n2, -1);
    vtkMath::Add(n, pointB, v1);
    vtkMath::Add(n2, pointB, v2);
    if (vtkMath::Distance2BetweenPoints(v1, pointTest) > vtkMath::Distance2BetweenPoints(v2, pointTest))
    {
      return n2;
    }
    else
    {
      return n;
    }
  }

  static void fromMat2Vtk(cv::Mat _src, vtkImageData* _dest)
  {
    vtkImageImport *importer = vtkImageImport::New();
    if (_dest)
    {
      importer->SetOutput(_dest);
    }
    importer->SetDataSpacing(1, 1, 1);
    importer->SetDataOrigin(0, 0, 0);
    importer->SetWholeExtent(0, _src.size().width - 1, 0,
      _src.size().height - 1, 0, 0);
    importer->SetDataExtentToWholeExtent();
    importer->SetDataScalarTypeToUnsignedChar();
    importer->SetNumberOfScalarComponents(_src.channels());
    importer->SetImportVoidPointer(_src.data);
    importer->Update();
  }

  static void getRotationMatrix(cv::Mat &R, cv::Matx34f P)
  {
    for (size_t i = 0; i < 3; i++)
    {
      for (size_t j = 0; j < 3; j++)
      {
        R.at<double>(i,j) = P(i,j);
      }
    }
  }

  static void getRotationMatrix(cv::Mat &R, cv::Mat P)
  {
    for (size_t i = 0; i < 3; i++)
    {
      for (size_t j = 0; j < 3; j++)
      {
        R.at<double>(i, j) = P.at<double>(i, j);
      }
    }
  }

  static void getRotationMatrix(cv::Mat &R, vtkSmartPointer<vtkMatrix4x4> P)
  {
    for (size_t i = 0; i < 3; i++)
    {
      for (size_t j = 0; j < 3; j++)
      {
        R.at<float>(i, j) = P->GetElement(i, j);
      }
    }
  }

  static void getTranslationVector(cv::Mat &t, cv::Matx34f P)
  {
    for (size_t i = 0; i < 3; i++)
    {
      t.at<double>(i, 0) = P(i, 3);
    }
  }

  static void getTranslationVector(cv::Mat &t, vtkSmartPointer<vtkMatrix4x4> P)
  {
    for (size_t i = 0; i < 3; i++)
    {
      t.at<float>(i, 0) = (float)P->GetElement(i, 3);
    }
  }

  static void printR(cv::Mat R)
  {
    std::cout << "[" << R.at<double>(0, 0) << ", " << R.at<double>(0, 1) << ", " << R.at<double>(0, 2) << "\n" <<
      R.at<double>(1, 0) << ", " << R.at<double>(1, 1) << ", " << R.at<double>(1, 2)  << "\n" <<
      R.at<double>(2, 0) << ", " << R.at<double>(2, 1) << ", " << R.at<double>(2, 2)  << "]\n" << std::endl;
  }

  static void printMat(cv::Mat m)
  {
    for (size_t i = 0; i < m.rows; i++)
    {
      for (size_t j = 0; j < m.cols; j++)
      {
        std::cout << m.at<double>(i, j) << " ";
      }
      std::cout << "\n";
    }
    std::cout << "\n" << std::endl;
  }
  static void printMat(cv::Matx34f m)
  {
    for (size_t i = 0; i < m.rows; i++)
    {
      for (size_t j = 0; j < m.cols; j++)
      {
        std::cout << m(i, j) << " ";
      }
      std::cout << "\n";
    }
    std::cout << "\n" << std::endl;
  }

  static void saveMat(cv::Mat m, std::string path)
  {
    std::ofstream myfile;
    myfile.open(path);
    for (size_t i = 0; i < m.rows; i++)
    {
      for (size_t j = 0; j < m.cols; j++)
      {
        myfile << m.at<double>(i, j) << " ";
      }
      myfile << "\n";
    }
    myfile.close();
  }

  static void printProjectionMatrix(cv::Mat R, cv::Mat t)
  {
      std::cout << "[" << R.at<float>(0, 0) << ", " << R.at<float>(0, 1) << ", " << R.at<float>(0, 2) << ", " << t.at<float>(0, 0) << "\n" <<
      R.at<float>(1, 0)<< ", " << R.at<float>(1, 1)<< ", " << R.at<float>(1, 2)<< ", " << t.at<float>(1, 0) << "\n" <<
      R.at<float>(2, 0)<< ", " << R.at<float>(2, 1)<< ", " << R.at<float>(2, 2)<< ", " << t.at<float>(2, 0) << "]\n" << std::endl;
  }
  static void printProjectionMatrix(cv::Mat Rt)
  {
    std::cout << "[" << Rt.at<float>(0, 0) << ", " << Rt.at<float>(0, 1) << ", " << Rt.at<float>(0, 2) << ", " << Rt.at<float>(0, 3) << "\n" <<
      Rt.at<float>(1, 0) << ", " << Rt.at<float>(1, 1) << ", " << Rt.at<float>(1, 2) << ", " << Rt.at<float>(1, 3) << "\n" <<
      Rt.at<float>(2, 0) << ", " << Rt.at<float>(2, 1) << ", " << Rt.at<float>(2, 2) << ", " << Rt.at<float>(2, 3) << "]\n" << std::endl;
  }
  static void printProjectionMatrix(cv::Matx34f P)
  {
    std::cout << "[" << P(0, 0) << ", " << P(0, 1) << ", " << P(0, 2) << ", " << P(0, 3) << "\n" <<
                        P(1, 0) << ", " << P(1, 1) << ", " << P(1, 2) << ", " << P(1, 3) << "\n" <<
                        P(2, 0) << ", " << P(2, 1) << ", " << P(2, 2) << ", " << P(2, 3) << "]\n" << std::endl;
  }

  static vtkSmartPointer<vtkMatrix4x4> cv2vtkMatrix(cv::Matx34f P)
  {
    vtkSmartPointer<vtkMatrix4x4> mat = vtkSmartPointer<vtkMatrix4x4>::New();
    for (size_t i = 0; i < 3; i++)
    {
      for (size_t j = 0; j < 4; j++)
      {
        mat->SetElement(i,j, P(i,j));
      }
    }
    mat->SetElement(3, 0, 0);
    mat->SetElement(3, 1, 0);
    mat->SetElement(3, 2, 0);
    mat->SetElement(3, 3, 1);
    return mat;
  }

  static cv::Mat multiplyPose(cv::Mat mat1, cv::Matx34f P)
  {
    if (mat1.cols != P.rows)
    {
      return cv::Mat();
    }
    cv::Mat mat = cv::Mat_<double>(mat1.rows, P.cols);
    double sum = 0;
    for (size_t i = 0; i < mat1.rows; i++)
    {
      for (size_t j = 0; j < P.cols; j++)
      {
        for (size_t k = 0; k < P.rows; k++)
        {
          sum += mat1.at<double>(i, k)*P(k, j);
        }
        mat.at<double>(i, j) = sum;
        sum = 0;
      }
    }
    return mat;
  }

  static cv::Mat createRT(cv::Mat R, cv::Mat t)
  {
    cv::Mat RT = cv::Mat_<double>(3, 4);
    for (size_t i = 0; i < R.rows; i++)
    {
      for (size_t j = 0; j < R.cols; j++)
      {
        RT.at<double>(i, j) = R.at<double>(i, j);
      }
    }
    for (size_t i = 0; i < 3; i++)
    {
      RT.at<double>(i, 3) = t.at<double>(i, 0);
    }
    return RT;
  }
  static cv::Matx34f createRT4(cv::Mat R, cv::Mat t)
  {
    cv::Matx34f RT;
    for (size_t i = 0; i < R.rows; i++)
    {
      for (size_t j = 0; j < R.cols; j++)
      {
        RT(i, j) = R.at<double>(i, j);
      }
    }
    for (size_t i = 0; i < 3; i++)
    {
      RT(i, 3) = t.at<double>(i, 0) / t.at<double>(3, 0);
    }
    return RT;
  }

  static void rectifyP(cv::Matx34f P1, cv::Matx34f P2, cv::Mat K, cv::Mat &H1, cv::Mat &H2, cv::Mat &Pn1, cv::Mat &Pn2)
  {
    cv::Mat R1 = cv::Mat_<double>(3, 3);
    cv::Mat R2 = cv::Mat_<double>(3, 3);
    cv::Mat t1 = cv::Mat_<double>(3, 1);
    cv::Mat t2 = cv::Mat_<double>(3, 1);
    
    Utils::getRotationMatrix(R1, P1);
    Utils::getRotationMatrix(R2, P2);
    Utils::getTranslationVector(t1, P1);
    Utils::getTranslationVector(t2, P2);

    cv::Mat c1 = -1 * R1.t() * t1;
    cv::Mat c2 = -1 * R2.t() * t2;

    cv::Mat v1 = (c2 - c1);
    cv::Mat aux3 = R1.rowRange(1, 2).cross(R1.rowRange(2, 3)) * c2;
    double res = aux3.at<double>(0, 0);
    if (res > 0)
    {
      res = 1;
    }
    else if (res < 0)
    {
      res = -1;
    }
    v1 *= res;

    cv::Mat v2 = R1.rowRange(2, 3).t().cross(v1);

    cv::Mat v3 = v1.cross(v2);

    double normV1 = cv::norm(v1, cv::NORM_L2);
    double normV2 = cv::norm(v2, cv::NORM_L2);
    double normV3 = cv::norm(v3, cv::NORM_L2);
    cv::Mat R = cv::Mat_<double>(3, 3);
    for (size_t i = 0; i < R.rows; i++)
    {
      for (size_t j = 0; j < R.cols; j++)
      {
        if (i == 0)
        {
          R.at<double>(i, j) = v1.at<double>(j, 0) / normV1;
        }
        else if (i == 1)
        {
          R.at<double>(i, j) = v2.at<double>(j, 0) / normV2;
        }
        else if (i == 2)
        {
          R.at<double>(i, j) = v3.at<double>(j, 0) / normV3;
        }
      }
    }

    Pn1 = K * createRT(R, -1 * R * c1);
    Pn2 = K * createRT(R, -1 * R * c2);


    cv::Mat P1org = multiplyPose(K, (cv::Mat)P1);
    cv::Mat P2org = multiplyPose(K, (cv::Mat)P2);

    cv::Mat R_P1org = cv::Mat_<double>(3, 3);
    cv::Mat R_P2org = cv::Mat_<double>(3, 3);
    cv::Mat R_Pn1 = cv::Mat_<double>(3, 3);
    cv::Mat R_Pn2 = cv::Mat_<double>(3, 3);

    Utils::getRotationMatrix(R_P1org, P1org);
    Utils::getRotationMatrix(R_P2org, P2org);
    Utils::getRotationMatrix(R_Pn1, Pn1);
    Utils::getRotationMatrix(R_Pn2, Pn2);


    H1 = R_Pn1 * R_P1org.inv();
    H2 = R_Pn2 * R_P2org.inv();
  }

  static cv::Mat htx(cv::Mat H, cv::Mat X)
  {
    cv::Mat newX;
    X.copyTo(newX);

    newX.push_back(cv::Mat_<double>::ones(1, X.cols));


    cv::Mat Y = H * newX;

    cv::Mat newY;
    cv::divide(Y, cv::repeat(Y.rowRange(Y.rows - 1, Y.rows), Y.rows, 1), newY);

    return newY.rowRange(0, newY.rows - 1);

  }
  
  static cv::Mat findBB(cv::Mat H, cv::Size imageSize)
  {
    cv::Mat corners = (cv::Mat_<double>(2, 4) << 0, 0, imageSize.width, imageSize.width, 0, imageSize.height, 0, imageSize.height);
    corners = htx(H, corners);
    
    double minX, maxX, minY, maxY;

    cv::minMaxLoc(corners.rowRange(0, 1), &minX, &maxX);
    cv::minMaxLoc(corners.rowRange(1, 2), &minY, &maxY);

    cv::Mat bb = (cv::Mat_<double>(4, 1) << floor(minX), floor(minY), ceil(maxX), ceil(maxY));

    return bb;
  }

  static float euclideanDist(cv::Point2f& p, cv::Point2f& q) {
    cv::Point2f diff = p - q;
    return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
  }

  static cv::Mat imWarp(cv::Mat img, cv::Mat H, cv::Mat bb, cv::Mat &imgMask, cv::Mat &newH)
  {
    //Points to compute new H
    std::vector<cv::Point2f> srcPoints;
    //10 10 because some pixels are out of bounds
    srcPoints.push_back(cv::Point2f(10, 10));
    srcPoints.push_back(cv::Point2f(img.cols - 10, 10));
    srcPoints.push_back(cv::Point2f(img.cols - 10, img.rows - 10));
    srcPoints.push_back(cv::Point2f(10, img.rows - 10));
    std::vector<cv::Point2f> dstPoints;
    dstPoints.resize(4);
    //meshgrid
    cv::Mat x = cv::Mat_<double>(1, abs(bb.at<double>(0, 0)) + abs(bb.at<double>(2, 0)) + 1);
    double startValue = bb.at<double>(0, 0);
    cv::parallel_for_(cv::Range(0, x.cols), [&](const cv::Range& range) {
      for (int r = range.start; r < range.end; r++)
      {
        x.at<double>(0, r) = startValue + r;
      }
    });
    cv::Mat y = cv::Mat_<double>(abs(bb.at<double>(1, 0)) + abs(bb.at<double>(3, 0)) + 1, 1);
    startValue = bb.at<double>(1, 0);
    cv::parallel_for_(cv::Range(0, y.rows), [&](const cv::Range& range) {
      for (int r = range.start; r < range.end; r++)
      {
        y.at<double>(r, 0) = startValue + r;
      }
    });
    cv::Mat xx = cv::repeat(x, y.rows, 1);
    cv::Mat yy = cv::repeat(y, 1, x.cols);
    //
    cv::Mat pp = cv::Mat_<double>(2, xx.rows*xx.cols);


    cv::parallel_for_(cv::Range(0, xx.rows*xx.cols), [&](const cv::Range& range) {
      for (int r = range.start; r < range.end; r++)
      {
        int i = r / xx.cols;
        int j = r % xx.cols;
        pp.at<double>(0, r) = xx.at<double>(i, j);
        pp.at<double>(1, r) = yy.at<double>(i, j);
      }
    });
    
    cv::Mat Hinv = H.inv();
    cv::Mat newPP = htx(Hinv, pp);

    cv::Mat xxFloor = cv::Mat_<double>(xx.rows, xx.cols);
    cv::Mat yyFloor = cv::Mat_<double>(xx.rows, xx.cols);
    cv::parallel_for_(cv::Range(0, xx.rows*xx.cols), [&](const cv::Range& range) {
      for (int r = range.start; r < range.end; r++)
      {
        int i = r / xx.cols;
        int j = r % xx.cols;
        xx.at<double>(i, j) = newPP.at<double>(0, r);
        yy.at<double>(i, j) = newPP.at<double>(1, r);
        xxFloor.at<double>(i, j) = floor(xx.at<double>(i, j));
        yyFloor.at<double>(i, j) = floor(yy.at<double>(i, j));
      }
    });
    //
    cv::Mat imgWarp(xx.rows, xx.cols, CV_8UC1);
    imgMask = cv::Mat(xx.rows, xx.cols, CV_8UC1);
    cv::parallel_for_(cv::Range(0, xx.rows*xx.cols), [&](const cv::Range& range) {
      for (int r = range.start; r < range.end; r++)
      {
        int i = r / xx.cols;
        int j = r % xx.cols;

        int coordY = xxFloor.at<double>(i, j);
        int coordX = yyFloor.at<double>(i, j);
        for (size_t k = 0; k < srcPoints.size(); k++)
        {
          if (abs(srcPoints.at(k).x - coordY) < 2 && abs(srcPoints.at(k).y - coordX) < 2)
          {
            dstPoints.at(k) = cv::Point2f(j, i);
            break;
          }
        }
        if (coordX < 0 || coordX >(img.rows - 1) || coordY < 0 || coordY >(img.cols - 1) ||
          coordX + 1 >(img.rows - 1) || coordY + 1 > (img.cols - 1))
        {
          imgWarp.at<uchar>(i, j) = 0;
          imgMask.at<uchar>(i, j) = 0;
        }
        else
        {
          double deltaX = yy.at<double>(i, j) - coordX;
          double deltaMX = 1 - deltaX;
          double deltaY = xx.at<double>(i, j) - coordY;
          double deltaMY = 1 - deltaY;

          double F00 = img.at<uchar>(coordX, coordY);
          double F01 = img.at<uchar>(coordX, coordY + 1);
          double F10 = img.at<uchar>(coordX + 1, coordY);
          double F11 = img.at<uchar>(coordX + 1, coordY + 1);

          double FA = F00*deltaMX + F10*deltaX;
          double FB = F01*deltaMX + F11*deltaX;
          imgWarp.at<uchar>(i, j) = FA*deltaMY + FB*deltaY;
          imgMask.at<uchar>(i, j) = 255;
        }
      }
    });
    /*double deltaX = 0;
    double deltaMX = 0;
    double deltaY = 0;
    double deltaMY = 0;
    double F00 = 0;
    double F01 = 0;
    double F10 = 0;
    double F11 = 0;
    double FA = 0;
    double FB = 0;
    int coordX = 0;
    int coordY = 0;
    cv::Mat imgWarp(xx.rows, xx.cols, CV_8UC1);
    imgMask = cv::Mat(xx.rows, xx.cols, CV_8UC1);
    for (size_t i = 0; i < xx.rows; i++)
    {
      for (size_t j = 0; j < xx.cols; j++)
      {
        coordY = xxFloor.at<double>(i, j);
        coordX = yyFloor.at<double>(i, j);
        for (size_t k = 0; k < srcPoints.size(); k++)
        {
          if (srcPoints.at(k).x == coordY && srcPoints.at(k).y == coordX)
          {
            dstPoints.at(k) = cv::Point2f(j, i);
            break;
          }
        }
        if (coordX < 0 || coordX > (img.rows-1) || coordY < 0 || coordY > (img.cols-1) ||
            coordX + 1 > (img.rows-1) || coordY + 1 > (img.cols-1))
        {
          imgWarp.at<uchar>(i, j) = 0;
          imgMask.at<uchar>(i, j) = 0;
        }
        else
        {
          deltaX = yy.at<double>(i, j) - coordX;
          deltaMX = 1 - deltaX;
          deltaY = xx.at<double>(i, j) - coordY;
          deltaMY = 1 - deltaY;

          F00 = img.at<uchar>(coordX, coordY);
          F01 = img.at<uchar>(coordX, coordY + 1);
          F10 = img.at<uchar>(coordX + 1, coordY);
          F11 = img.at<uchar>(coordX + 1, coordY + 1);

          FA = F00*deltaMX + F10*deltaX;
          FB = F01*deltaMX + F11*deltaX;
          imgWarp.at<uchar>(i, j) = FA*deltaMY + FB*deltaY;
          imgMask.at<uchar>(i, j) = 255;
        }
      }
    }*/
    newH = cv::findHomography(dstPoints, srcPoints, CV_RANSAC);

    //cv::namedWindow("Pair", CV_WINDOW_NORMAL);
    //cv::Mat imgConcatDraw;
    //imgWarp.copyTo(imgConcatDraw);
    //cv::cvtColor(imgConcatDraw, imgConcatDraw, cv::COLOR_GRAY2BGR);
    //cv::circle(imgConcatDraw, dstPoints.at(0), 5, cv::Scalar(255, 0, 0), 5);
    //cv::circle(imgConcatDraw, dstPoints.at(1), 5, cv::Scalar(0, 255, 0), 5);
    //cv::circle(imgConcatDraw, dstPoints.at(2), 5, cv::Scalar(0, 0, 255), 5);
    //cv::circle(imgConcatDraw, dstPoints.at(3), 5, cv::Scalar(255, 0, 255), 5);
    //cv::imshow("Pair", imgConcatDraw);
    //cv::waitKey(0);

    //printMat(newH);
    return imgWarp;
  }

  static void imRectify(cv::Mat img1, cv::Mat img2, cv::Mat H1, cv::Mat H2, cv::Mat &img1Warp, cv::Mat &img2Warp, cv::Mat &img1Mask, cv::Mat &img2Mask, cv::Mat &newH1, cv::Mat &newH2)
  {
    cv::Mat bb1 = findBB(H1, img1.size());
    cv::Mat bb2 = findBB(H2, img2.size());

    //Define minY e maxY
    bb1.at<double>(1, 0) = bb1.at<double>(1, 0) <  bb2.at<double>(1, 0) ? bb1.at<double>(1, 0) : bb2.at<double>(1, 0);
    bb1.at<double>(3, 0) = bb1.at<double>(3, 0) >  bb2.at<double>(3, 0) ? bb1.at<double>(3, 0) : bb2.at<double>(3, 0);
    bb2.at<double>(1, 0) = bb1.at<double>(1, 0);
    bb2.at<double>(3, 0) = bb1.at<double>(3, 0);

    //Define maior widht
    double w1 = bb1.at<double>(2, 0) - bb1.at<double>(0, 0);
    double w2 = bb2.at<double>(2, 0) - bb2.at<double>(0, 0);

    double w = w1 >  w2 ? w1 : w2;

    double c1 = floor( (bb1.at<double>(2, 0) + bb1.at<double>(0, 0)) / 2.0);
    double c2 = floor( (bb2.at<double>(2, 0) + bb2.at<double>(0, 0)) / 2.0);


    bb1.at<double>(0, 0) = c1 - floor(w / 2.0);
    bb1.at<double>(2, 0) = c1 + floor(w / 2.0);

    bb2.at<double>(0, 0) = c2 - floor(w / 2.0);
    bb2.at<double>(2, 0) = c2 + floor(w / 2.0);


    //
    cv::Mat A = cv::Mat::eye(3, 3, CV_64F); A.at<double>(0, 2) = -bb1.at<double>(0, 0); A.at<double>(1, 2) = -bb1.at<double>(1, 0);
    newH1 = A*H1;
    cv::Mat img1MaskInit = cv::Mat(img1.size(), CV_8UC1, cv::Scalar(255, 255, 255));
    cv::warpPerspective(img1, img1Warp, newH1, cv::Size(bb1.at<double>(2, 0) - bb1.at<double>(0, 0), bb1.at<double>(3, 0) - bb1.at<double>(1, 0)));
    cv::warpPerspective(img1MaskInit, img1Mask, newH1, img1Warp.size());

    A.at<double>(0, 2) = -bb2.at<double>(0, 0); A.at<double>(1, 2) = -bb2.at<double>(1, 0);
    newH2 = A*H2;
    cv::Mat img2MaskInit = cv::Mat(img2.size(), CV_8UC1, cv::Scalar(255, 255, 255));
    cv::warpPerspective(img2, img2Warp, newH2, cv::Size(bb2.at<double>(2, 0) - bb2.at<double>(0, 0), bb2.at<double>(3, 0) - bb2.at<double>(1, 0)));
    cv::warpPerspective(img2MaskInit, img2Mask, newH2, img2Warp.size());

    newH1 = newH1.inv();
    newH2 = newH2.inv();

    //cv::Mat imgConcat2;
    //hconcat(img1Warp, img2Warp, imgConcat2);
    //cv::namedWindow("Pair", CV_WINDOW_NORMAL);
    //cv::imshow("Pair", imgConcat2);
    //cv::Mat imgConcat3;
    //hconcat(img1Mask, img2Mask, imgConcat3);
    //cv::namedWindow("Pair2", CV_WINDOW_NORMAL);
    //cv::imshow("Pair2", imgConcat3);
    //cv::waitKey(0);
    //



    //img1Warp = imWarp(img1, H1, bb1, img1Mask, newH1);
    //img2Warp = imWarp(img2, H2, bb2, img2Mask, newH2);
}

  static double ZNCC(cv::Mat img1, cv::Mat img2)
  {
    double mean1 = 0;
    double mean2 = 0;
    double windowSize = img1.rows;
    for (int k = 0; k < windowSize; k++)
    {
      for (int j = 0; j < windowSize; j++)
      {
        mean1 += img1.at<uchar>(k, j);
        mean2 += img2.at<uchar>(k, j);
      }
    }
    mean1 = mean1 / (windowSize*windowSize);
    mean2 = mean2 / (windowSize*windowSize);
    if (mean1 > 240 || mean1 < 15 || mean2 > 240 || mean2 < 15)
    {
      return -1;
    }
    double sum = 0;
    double sumSq1 = 0;
    double sumSq2 = 0;
    double res1;
    double res2;
    for (size_t k = 0; k < windowSize; k++)
    {
      for (size_t j = 0; j < windowSize; j++)
      {
        res1 = img1.at<uchar>(k, j) - mean1;
        res2 = img2.at<uchar>(k, j) - mean2;
        sum += res1 * res2;
        sumSq1 += res1*res1;
        sumSq2 += res2*res2;
      }
    }
    return (sum / sqrt(sumSq1*sumSq2));
  }

  static cv::Point2f ZNCC(cv::Mat image, cv::Mat temp, int xFeature, int windowSize, int maxDist, int yBusca)
  {
    int u, v;
    cv::Point2f best;
    double sum = 0;
    double sumSq1 = 0;
    double sumSq2 = 0;
    double med1 = 0;
    double med2 = 0;
    double ZNCC = 0;
    double bestMatch = -1;
    int y = yBusca;//-(epipolarLine[2]) / epipolarLine[1];
    if ((y - windowSize) < 0 && (y + windowSize) >= image.rows)
    {
      return cv::Point2f(-1, -1);
    }
    int xStart = xFeature - maxDist;
    int xEnd = xFeature + maxDist;
    if ((xStart - windowSize) < 0)
    {
      xStart = windowSize;
    }
    if ((xEnd + windowSize) >= image.cols)
    {
      xEnd = image.cols - 1 - windowSize;
    }
    int windowSize2 = windowSize * 2 + 1;
    double res1;
    double res2;
    v = y - windowSize;
    for (int k = 0; k < windowSize2; k++)
    {
      for (int j = 0; j < windowSize2; j++)
      {
        med2 += temp.at<uchar>(k, j);
      }
    }
    med2 = med2 / (windowSize2*windowSize2);
    if (med2 > 240 || med2 < 15)
    {
      return cv::Point2f(-1, -1);
    }
    for (int x = xStart; x <= xEnd; x++)
    {
      u = x - windowSize;
      for (size_t k = 0; k < windowSize2; k++)
      {
        for (size_t j = 0; j < windowSize2; j++)
        {
          med1 += image.at<uchar>(v + k, u + j);
        }
      }
      med1 = med1 / (windowSize2*windowSize2);
      if (med1 < 240 && med1 > 15)
      {
        for (size_t k = 0; k < windowSize2; k++)
        {
          for (size_t j = 0; j < windowSize2; j++)
          {
            res1 = image.at<uchar>(v + k, u + j) - med1;
            res2 = temp.at<uchar>(k, j) - med2;
            sum += res1 * res2;
            sumSq1 += res1*res1;
            sumSq2 += res2*res2;
          }
        }
        ZNCC = sum / sqrt(sumSq1*sumSq2);
        if (ZNCC > bestMatch)
        {
          bestMatch = ZNCC;
          best = cv::Point2f(x, y);
        }
        sum = 0;
        sumSq1 = 0;
        sumSq2 = 0;
      }
      med1 = 0;
    }
    if (bestMatch >= 0.9)
    {
      /*cv::namedWindow("im1", CV_WINDOW_NORMAL);
      cv::namedWindow("im2", CV_WINDOW_NORMAL);
      cv::imshow("im1", image(cv::Range(y - windowSize, y + windowSize + 1), cv::Range(best.x - windowSize, best.x + windowSize + 1)));
      cv::imshow("im2", temp);
      cv::waitKey(0);*/
      return best;
    }
    return cv::Point2f(-1, -1);
  }

  static cv::Point2f ZNCC(cv::Mat image, cv::Mat temp, int xFeature, int windowSize, int maxDist, int yBusca, cv::Mat imgMask, double& score)
  {
    int u, v;
    cv::Point2f best;
    double sum = 0;
    double sumSq1 = 0;
    double sumSq2 = 0;
    double med1 = 0;
    double med2 = 0;
    double ZNCC = 0;
    double bestMatch = -1;
    int y = yBusca;

    int yStart = yBusca;// - maxDist;
    int yEnd = yBusca;// + maxDist;
    /*if ((yStart - windowSize) < 0)
    {
      yStart = windowSize;
    }
    if ((yEnd + windowSize) >= image.rows)
    {
      yEnd = image.rows - 1 - windowSize;
    }*/


    if ((y - windowSize) < 0 && (y + windowSize) >= image.rows)
    {
      return cv::Point2f(-1, -1);
    }
    int xStart = xFeature - maxDist;
    int xEnd = xFeature + maxDist;
    if ((xStart - windowSize) < 0)
    {
      xStart = windowSize;
    }
    if ((xEnd + windowSize) >= image.cols)
    {
      xEnd = image.cols - 1 - windowSize;
    }
    int windowSize2 = windowSize * 2 + 1;
    double res1;
    double res2;
    for (int k = 0; k < windowSize2; k++)
    {
      for (int j = 0; j < windowSize2; j++)
      {
        med2 += temp.at<uchar>(k, j);
      }
    }
    med2 = med2 / (windowSize2*windowSize2);
    /*if (med2 > 240 || med2 < 15)
    {
      return cv::Point2f(-1, -1);
    }*/
    for (y = yStart; y <= yEnd; y++)
    {
      for (int x = xStart; x <= xEnd; x++)
      {
        if (imgMask.at<uchar>(y, x) != 0)
        {
          v = y - windowSize;
          u = x - windowSize;
          for (size_t k = 0; k < windowSize2; k++)
          {
            for (size_t j = 0; j < windowSize2; j++)
            {
              med1 += image.at<uchar>(v + k, u + j);
            }
          }
          med1 = med1 / (windowSize2*windowSize2);
          if (true)//med1 < 240 && med1 > 15)
          {
            for (size_t k = 0; k < windowSize2; k++)
            {
              for (size_t j = 0; j < windowSize2; j++)
              {
                res1 = image.at<uchar>(v + k, u + j) - med1;
                res2 = temp.at<uchar>(k, j) - med2;
                sum += res1 * res2;
                sumSq1 += res1*res1;
                sumSq2 += res2*res2;
              }
            }
            ZNCC = sum / sqrt(sumSq1*sumSq2);
            if (ZNCC > bestMatch)
            {
              bestMatch = ZNCC;
              best = cv::Point2f(x, y);
            }
            sum = 0;
            sumSq1 = 0;
            sumSq2 = 0;
          }
          med1 = 0;
        }
      }
     }
    if (bestMatch >= 0.5)
    {
      score = bestMatch;
      return best;
    }
    return cv::Point2f(-1, -1);
  }

  static double NCC(cv::Mat img1, cv::Mat img2, cv::Point2f pt2D, cv::Mat H_ij, int windowSize)
  {
    std::vector<cv::Point2f> pts;
    for (int i = -windowSize; i <= windowSize; i++)
    {
      for (int j = -windowSize; j <= windowSize; j++)
      {
        pts.push_back(cv::Point2f(pt2D.x + i, pt2D.y + j));
      }
    }
    int windowSize2 = pts.size();
    std::vector<cv::Point2f> ptsTransf;
    cv::perspectiveTransform(pts, ptsTransf, H_ij);
    /*int x, y;
    for (size_t i = 0; i < ptsTransf.size(); i++)
    {
      x = ptsTransf.at(i).x;
      y = ptsTransf.at(i).y;
      if (x < 0 || x >= img2.cols || y < 0 || y >= img2.rows)
      {
        return 0;
      }
    }*/
    double med1 = 0;
    double med2 = 0;
    for (size_t i = 0; i < windowSize2; i++)
    {
      med1 += img1.at<uchar>(pts.at(i));
      med2 += img2.at<uchar>(ptsTransf.at(i));
    }
    med1 /= windowSize2;
    med2 /= windowSize2;
    double res1 , res2, sum = 0, sumSq1 = 0, sumSq2 = 0;
    for (size_t i = 0; i < windowSize2; i++)
    {
      res1 = img1.at<uchar>(pts.at(i)) - med1;
      res2 = img2.at<uchar>(ptsTransf.at(i)) - med2;
      sum += res1*res2;
      sumSq1 += res1*res1;
      sumSq2 += res2*res2;
    }
    return sum / sqrt((sumSq1*sumSq2));
  }

  static double NCC(cv::Mat img1, cv::Mat img2)
  {
    double med1 = 0;
    double med2 = 0;
    int windowSize = img1.cols;
    for (size_t i = 0; i < windowSize; i++)
    {
      for (size_t j = 0; j < windowSize; j++)
      {
        med1 += img1.at<uchar>(i, j);
        med2 += img2.at<uchar>(i, j);
      }
    }
    med1 /= windowSize*windowSize;
    med2 /= windowSize*windowSize;
    double res1, res2, sum = 0, sumSq1 = 0, sumSq2 = 0;
    for (size_t i = 0; i < windowSize; i++)
    {
      for (size_t j = 0; j < windowSize; j++)
      {
        res1 = img1.at<uchar>(i, j) - med1;
        res2 = img2.at<uchar>(i, j) - med2;
        sum += res1*res2;
        sumSq1 += res1*res1;
        sumSq2 += res2*res2;
      }
    }
    return sum / sqrt((sumSq1*sumSq2));
  }

  static cv::Mat getTransformedImage(cv::Mat img, std::vector<cv::Point2f> pts ,cv::Mat H, int windowSize)
  {
    cv::Mat newImg = cv::Mat(windowSize, windowSize, CV_8UC1);
    cv::parallel_for_(cv::Range(0, pts.size()), [&](const cv::Range& range) {
      for (int r = range.start; r < range.end; r++)
      {
        int i = r / windowSize;
        int j = r % windowSize;
        int x = pts.at(r).x;
        int y = pts.at(r).y;
        int a = (H.at<double>(0, 0)*x + H.at<double>(0, 1)*y + H.at<double>(0, 2)) / (H.at<double>(2, 0)*x + H.at<double>(2, 1)*y + H.at<double>(2, 2));
        int b = (H.at<double>(1, 0)*x + H.at<double>(1, 1)*y + H.at<double>(1, 2)) / (H.at<double>(2, 0)*x + H.at<double>(2, 1)*y + H.at<double>(2, 2));
        if (a >= 0 && a < img.rows && b >= 0 && b < img.cols)
        {
        newImg.at<uchar>(i, j) = img.at<uchar>(a, b);
        }
        else
        {
        newImg.at<uchar>(i, j) = 0;
        }
      }
    });
    /*int x, y;
    int a, b;
    size_t cont = 0;
    for (size_t i = 0; i < windowSize; i++)
    {
      for (size_t j = 0; j < windowSize; j++)
      {
        x = pts.at(cont).x;
        y = pts.at(cont).y;
        a = (H.at<double>(0, 0)*x + H.at<double>(0, 1)*y + H.at<double>(0, 2)) / (H.at<double>(2, 0)*x + H.at<double>(2, 1)*y + H.at<double>(2, 2));
        b = (H.at<double>(1, 0)*x + H.at<double>(1, 1)*y + H.at<double>(1, 2)) / (H.at<double>(2, 0)*x + H.at<double>(2, 1)*y + H.at<double>(2, 2));
        if (a >= 0 && a < img.rows && b >= 0 && b < img.cols)
        {
          newImg.at<uchar>(i, j) = img.at<uchar>(a, b);
        }
        else
        {
          newImg.at<uchar>(i, j) = 0;
        }
        cont++;
      }
    }*/
    return newImg;
  }

  static bool isNormalTowardsCamera(cv::Vec3f normal, cv::Point3f pt, cv::Matx34f P)
  {
    double* norm = Utils::createDoubleVector(normal[0], normal[1], normal[2]);
    double* p = Utils::createDoubleVector(pt.x, pt.y, pt.z);
    double* camP = Utils::createDoubleVector(P(3, 0), P(3, 1), P(3, 2));
    double* newP = new double[3];
    vtkMath::Subtract(camP, p, newP);
    bool teste = true;
    if (vtkMath::Dot(norm, newP) < 0)
    {
      teste = false;
    }
    delete norm, p, camP, newP;
    return teste;
  }

  static std::string calcTime(DWORD initial_time, DWORD final_time)
  {
    DWORD time = final_time - initial_time;
    return getTimeString(time);
  }

  static std::string getTimeString(DWORD time)
  {
    int seconds = (int)(time / 1000) % 60;
    int milisseconds = time - (seconds * 1000);
    int minutes = (int)((time / (1000 * 60)) % 60);
    int hours = (int)((time / (1000 * 60 * 60)) % 24);
    std::stringstream result;
    result << hours << ":" << minutes << ":" << seconds << "." << milisseconds;
    return result.str();
  }

  static float SIGN(float x) {
    return (x >= 0.0f) ? +1.0f : -1.0f;
  }

  static float NORM(float a, float b, float c, float d) {
    return sqrt(a * a + b * b + c * c + d * d);
  }

  // quaternion = [w, x, y, z]'
  static cv::Mat mRot2Quat(cv::Matx34f P) {
    float r11 = P(0, 0);
    float r12 = P(0, 1);
    float r13 = P(0, 2);
    float r21 = P(1, 0);
    float r22 = P(1, 1);
    float r23 = P(1, 2);
    float r31 = P(2, 0);
    float r32 = P(2, 1);
    float r33 = P(2, 2);
    float q0 = (r11 + r22 + r33 + 1.0f) / 4.0f;
    float q1 = (r11 - r22 - r33 + 1.0f) / 4.0f;
    float q2 = (-r11 + r22 - r33 + 1.0f) / 4.0f;
    float q3 = (-r11 - r22 + r33 + 1.0f) / 4.0f;
    if (q0 < 0.0f) {
      q0 = 0.0f;
    }
    if (q1 < 0.0f) {
      q1 = 0.0f;
    }
    if (q2 < 0.0f) {
      q2 = 0.0f;
    }
    if (q3 < 0.0f) {
      q3 = 0.0f;
    }
    q0 = sqrt(q0);
    q1 = sqrt(q1);
    q2 = sqrt(q2);
    q3 = sqrt(q3);
    if (q0 >= q1 && q0 >= q2 && q0 >= q3) {
      q0 *= +1.0f;
      q1 *= SIGN(r32 - r23);
      q2 *= SIGN(r13 - r31);
      q3 *= SIGN(r21 - r12);
    }
    else if (q1 >= q0 && q1 >= q2 && q1 >= q3) {
      q0 *= SIGN(r32 - r23);
      q1 *= +1.0f;
      q2 *= SIGN(r21 + r12);
      q3 *= SIGN(r13 + r31);
    }
    else if (q2 >= q0 && q2 >= q1 && q2 >= q3) {
      q0 *= SIGN(r13 - r31);
      q1 *= SIGN(r21 + r12);
      q2 *= +1.0f;
      q3 *= SIGN(r32 + r23);
    }
    else if (q3 >= q0 && q3 >= q1 && q3 >= q2) {
      q0 *= SIGN(r21 - r12);
      q1 *= SIGN(r31 + r13);
      q2 *= SIGN(r32 + r23);
      q3 *= +1.0f;
    }
    else {
      printf("coding error\n");
    }
    float r = NORM(q0, q1, q2, q3);
    q0 /= r;
    q1 /= r;
    q2 /= r;
    q3 /= r;

    cv::Mat res = (cv::Mat_<float>(4, 1) << q0, q1, q2, q3);
    return res;
  }

  static cv::Mat getCenterVector(cv::Matx34f P)
  {
    cv::Mat R = cv::Mat_<double>(3, 3);
    cv::Mat t = cv::Mat_<double>(3, 1);
    Utils::getRotationMatrix(R, P);
    Utils::getTranslationVector(t, P);
    R = R.inv();
    return -R*t;
  }

};