//
// Created by xtcsun on 2021/4/27.
//

#ifndef RGBD_SLAM_SLAMBASE_H
#define RGBD_SLAM_SLAMBASE_H
#include <fstream>
#include <vector>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

using namespace cv;

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

// 帧结构
struct FRAME
{
    Mat rgb,depth;
    Mat desp;
    vector<KeyPoint> kp;
};

// PnP结果
struct RESULT_OF_PNP
{
    Mat rvec,tvec;
    int inliers;
};


// 相机内参结构
struct CAMERA_INTRINSIC_PARAMETERS
{
    double cx,cy,fx,fy,scale;
};


// 同时提取关键点与特征描述子
void computeKeyPointsAndDesp(FRAME& frame);

//计算两帧之间的运动
RESULT_OF_PNP estimateMotion(FRAME& frame1,FRAME& frame2,CAMERA_INTRINSIC_PARAMETERS& camera);


// 函数接口
// image2PointCloud 将rgb图转化为点云
PointCloud::Ptr image2PointCloud(cv::Mat& rgb, cv::Mat& depth,CAMERA_INTRINSIC_PARAMETERS& camera);

// point2dTo3d 将单个点从图像坐标转换为空间坐标
// input 3维点Point3F（u,v,d）
cv::Point3f point2dTo3d(cv::Point3f& point, CAMERA_INTRINSIC_PARAMETERS& camera);

// 参数读取类
class ParameterReader
{
public:
    ParameterReader( string filename="./parameters.txt" )
    {
        ifstream fin( filename.c_str() );
        if (!fin)
        {
            cerr<<"parameter file does not exist."<<endl;
            return;
        }
        while(!fin.eof())
        {
            string str;
            getline( fin, str );
            if (str[0] == '#')
            {
                // 以‘＃’开头的是注释
                continue;
            }

            int pos = str.find("=");
            if (pos == -1)
                continue;
            string key = str.substr( 0, pos );
            string value = str.substr( pos+1, str.length() );
            data[key] = value;

            if ( !fin.good() )
                break;
        }
    }
    string getData( string key )
    {
        map<string, string>::iterator iter = data.find(key);
        if (iter == data.end())
        {
            cerr<<"Parameter name "<<key<<" not found!"<<endl;
            return string("NOT_FOUND");
        }
        return iter->second;
    }
public:
    map<string, string> data;
};

#endif //RGBD_SLAM_SLAMBASE_H
