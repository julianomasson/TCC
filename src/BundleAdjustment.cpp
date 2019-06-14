#include "stdafx.h"
#include "BundleAdjustment.h"

BundleAdjustment::BundleAdjustment()
{
}

bool RemoveInvisiblePoints(vector<CameraT>& camera_data, vector<Point3D>& point_data,
  vector<int>& ptidx, vector<int>& camidx,
  vector<Point2D>& measurements, vector<std::string>& names, vector<int>& ptc)
{
  vector<float> zz(ptidx.size());
  for (size_t i = 0; i < ptidx.size(); ++i)
  {
    CameraT& cam = camera_data[camidx[i]];
    Point3D& pt = point_data[ptidx[i]];
    zz[i] = cam.m[2][0] * pt.xyz[0] + cam.m[2][1] * pt.xyz[1] + cam.m[2][2] * pt.xyz[2] + cam.t[2];
  }
  size_t median_idx = ptidx.size() / 2;
  std::nth_element(zz.begin(), zz.begin() + median_idx, zz.end());
  float dist_threshold = zz[median_idx] * 0.001f;

  //keep removing 3D points. until all of them are infront of the cameras..
  vector<bool> pmask(point_data.size(), true);
  int points_removed = 0;
  for (size_t i = 0; i < ptidx.size(); ++i)
  {
    int cid = camidx[i], pid = ptidx[i];
    if (!pmask[pid])continue;
    CameraT& cam = camera_data[cid];
    Point3D& pt = point_data[pid];
    bool visible = (cam.m[2][0] * pt.xyz[0] + cam.m[2][1] * pt.xyz[1] + cam.m[2][2] * pt.xyz[2] + cam.t[2] > dist_threshold);
    pmask[pid] = visible; //this point should be removed
    if (!visible) points_removed++;
  }
  if (points_removed == 0) return false;
  vector<int>  cv(camera_data.size(), 0);
  //should any cameras be removed ?
  int min_observation = 20; //cameras should see at leat 20 points

  do
  {
    //count visible points for each camera
    std::fill(cv.begin(), cv.end(), 0);
    for (size_t i = 0; i < ptidx.size(); ++i)
    {
      int cid = camidx[i], pid = ptidx[i];
      if (pmask[pid])  cv[cid]++;
    }

    //check if any more points should be removed
    vector<int>  pv(point_data.size(), 0);
    for (size_t i = 0; i < ptidx.size(); ++i)
    {
      int cid = camidx[i], pid = ptidx[i];
      if (!pmask[pid]) continue; //point already removed
      if (cv[cid] < min_observation) //this camera shall be removed.
      {
        ///
      }
      else
      {
        pv[pid]++;
      }
    }

    points_removed = 0;
    for (size_t i = 0; i < point_data.size(); ++i)
    {
      if (pmask[i] == false) continue;
      if (pv[i] >= 2) continue;
      pmask[i] = false;
      points_removed++;
    }
  } while (points_removed > 0);

  ////////////////////////////////////
  vector<bool> cmask(camera_data.size(), true);
  for (size_t i = 0; i < camera_data.size(); ++i) cmask[i] = cv[i] >= min_observation;
  ////////////////////////////////////////////////////////

  vector<int> cidx(camera_data.size());
  vector<int> pidx(point_data.size());




  ///modified model.
  vector<CameraT> camera_data2;
  vector<Point3D> point_data2;
  vector<int> ptidx2;
  vector<int> camidx2;
  vector<Point2D> measurements2;
  vector<std::string> names2;
  vector<int> ptc2;


  //
  if (names.size() < camera_data.size()) names.resize(camera_data.size(), std::string("unknown"));
  if (ptc.size() < 3 * point_data.size()) ptc.resize(point_data.size() * 3, 0);

  //////////////////////////////
  int new_camera_count = 0, new_point_count = 0;
  for (size_t i = 0; i < camera_data.size(); ++i)
  {
    if (!cmask[i])continue;
    camera_data2.push_back(camera_data[i]);
    names2.push_back(names[i]);
    cidx[i] = new_camera_count++;
  }

  for (size_t i = 0; i < point_data.size(); ++i)
  {
    if (!pmask[i])continue;
    point_data2.push_back(point_data[i]);
    ptc.push_back(ptc[i]);
    pidx[i] = new_point_count++;
  }

  int new_observation_count = 0;
  for (size_t i = 0; i < ptidx.size(); ++i)
  {
    int pid = ptidx[i], cid = camidx[i];
    if (!pmask[pid] || !cmask[cid]) continue;
    ptidx2.push_back(pidx[pid]);
    camidx2.push_back(cidx[cid]);
    measurements2.push_back(measurements[i]);
    new_observation_count++;
  }

  std::cout << "NOTE: removing " << (camera_data.size() - new_camera_count) << " cameras; " << (point_data.size() - new_point_count)
    << " 3D Points; " << (measurements.size() - new_observation_count) << " Observations;\n";

  camera_data2.swap(camera_data); names2.swap(names);
  point_data2.swap(point_data);   ptc2.swap(ptc);
  ptidx2.swap(ptidx);  camidx2.swap(camidx);
  measurements2.swap(measurements);

  return true;
}

BundleAdjustment::BundleAdjustment(Graph * g)
{
  graph = g;

  camera_data.resize(g->cameras.size()); // allocate the camera data
  photo_names.resize(g->cameras.size());
  cv::Matx34f P;
  for (size_t i = 0; i < g->cameras.size(); i++)
  {
    P = g->cameras.at(i)->P;
    for (size_t j = 0; j < 3; j++)
    {
      for (size_t k = 0; k < 3; k++)
      {
        camera_data[i].m[j][k] = P(j,k);
      }
    }
    for (size_t k = 0; k < 3; k++)
    {
      camera_data[i].t[k] = P(k, 3);
    }
    camera_data[i].SetFocalLength(g->cameras.at(i)->K.at<double>(0,0));
    photo_names[i] = g->cameras.at(i)->pathImage;
  }

  point_data.resize(g->tracks.size());
  Track* t;
  Keypoint* key;
  cv::Point3f pt3D;
  int cc[3];
  cc[0] = 255; cc[1] = 255; cc[2] = 255;
  for (size_t i = 0; i < g->tracks.size(); i++)
  {
    t = g->tracks.at(i);
    pt3D = t->getPoint3D();
    if (pt3D.x != -1 && pt3D.y != -1 && pt3D.z != -1)
    {
      point_data[i].SetPoint(pt3D.x, pt3D.y, pt3D.z);
      point_color.insert(point_color.end(), cc, cc + 3);
      for (size_t j = 0; j < t->keypoints.size(); j++)
      {
        key = t->keypoints.at(j);
        for (size_t k = 0; k < g->cameras.size(); k++)
        {
          if (g->cameras.at(k) == key->cam)
          {
            camidx.push_back(k);    //camera index
            break;
          }
        }
        ptidx.push_back(i);        //point index
                                   //add a measurment to the vector
        measurements.push_back(Point2D(key->getPoint().x - key->cam->K.at<double>(0, 2), key->getPoint().y - key->cam->K.at<double>(1, 2)));//Measurment image point - central point
      }
    }
  }

  //saveNVM("teste.nvm");
  //RemoveInvisiblePoints(camera_data, point_data, ptidx, camidx, measurements, photo_names, point_color);

}

void BundleAdjustment::runBundle(bool fullBA)
{
  ParallelBA pba(ParallelBA::PBA_CPU_DOUBLE);

  if (!fullBA)
  {
    for (size_t i = 0; i < camera_data.size()-1; i++)
    {
      camera_data[i].SetConstantCamera();
    }
  }

  pba.SetCameraData(camera_data.size(), &camera_data[0]);                        //set camera parameters
  pba.SetPointData(point_data.size(), &point_data[0]);                            //set 3D point data
  pba.SetProjection(measurements.size(), &measurements[0], &ptidx[0], &camidx[0]);//set the projections

  pba.GetInternalConfig()->__lm_max_iteration = 100;
  //pba.GetInternalConfig()->__lm_mse_threshold = 0.1;

  pba.RunBundleAdjustment();
  error = pba.GetMeanSquaredError();
}

Graph * BundleAdjustment::getResult()
{
  for (size_t i = 0; i < camidx.size(); i++)
  {
    Camera* cam = graph->cameras.at(camidx[i]);
    cv::Matx34f P;
    for (size_t j = 0; j < 3; j++)
    {
      for (size_t k = 0; k < 3; k++)
      {
        P(j, k) = camera_data[camidx[i]].m[j][k];
      }
    }
    for (size_t k = 0; k < 3; k++)
    {
      P(k, 3) = camera_data[camidx[i]].t[k];
    }
    cam->P = P;
    cam->K.at<double>(0, 0) = camera_data[camidx[i]].f;
    cam->K.at<double>(1, 1) = camera_data[camidx[i]].f;
  }
  for (size_t i = 0; i < ptidx.size(); i++)//Idiota, pois ptidx tem vários indices repetidos
  {
    graph->tracks.at(ptidx.at(i))->pt3D.x = point_data[ptidx.at(i)].xyz[0];
    graph->tracks.at(ptidx.at(i))->pt3D.y = point_data[ptidx.at(i)].xyz[1];
    graph->tracks.at(ptidx.at(i))->pt3D.z = point_data[ptidx.at(i)].xyz[2];
  }
  return graph;
}

double BundleAdjustment::getError()
{
  return error;
}

void SaveNVM(const char* filename, vector<CameraT>& camera_data, vector<Point3D>& point_data,
  vector<Point2D>& measurements, vector<int>& ptidx, vector<int>& camidx,
  vector<std::string>& names, vector<int>& ptc)
{
  std::cout << "Saving model to " << filename << "...\n";
  std::ofstream out(filename);

  out << "NVM_V3_R9T\n" << camera_data.size() << '\n' << std::setprecision(12);
  if (names.size() < camera_data.size()) names.resize(camera_data.size(), std::string("unknown"));
  if (ptc.size() < 3 * point_data.size()) ptc.resize(point_data.size() * 3, 0);

  ////////////////////////////////////
  for (size_t i = 0; i < camera_data.size(); ++i)
  {
    CameraT& cam = camera_data[i];
    out << names[i] << ' ' << cam.GetFocalLength() << ' ';
    for (int j = 0; j < 9; ++j) out << cam.m[0][j] << ' ';
    out << cam.t[0] << ' ' << cam.t[1] << ' ' << cam.t[2] << ' '
      << cam.GetNormalizedMeasurementDistortion() << " 0\n";
  }

  out << point_data.size() << '\n';

  for (size_t i = 0, j = 0; i < point_data.size(); ++i)
  {
    Point3D& pt = point_data[i];
    int * pc = &ptc[i * 3];
    out << pt.xyz[0] << ' ' << pt.xyz[1] << ' ' << pt.xyz[2] << ' '
      << pc[0] << ' ' << pc[1] << ' ' << pc[2] << ' ';

    size_t je = j;
    while (je < ptidx.size() && ptidx[je] == (int)i) je++;

    out << (je - j) << ' ';

    for (; j < je; ++j)    out << camidx[j] << ' ' << " 0 " << measurements[j].x << ' ' << measurements[j].y << ' ';

    out << '\n';
  }
}

void BundleAdjustment::saveNVM(std::string filename)
{
  SaveNVM(filename.c_str(), camera_data, point_data, measurements, ptidx, camidx, photo_names, point_color);
}
