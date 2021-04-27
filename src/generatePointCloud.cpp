//
// Created by xtcsun on 2021/4/25.
//

#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

using namespace std;

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

// 相机内参
const double camera_factor = 1000;
const double camera_cx = 325.5;
const double camera_cy = 253.5;
const double camera_fx = 518.0;
const double camera_fy = 519.0;


int main()
{
    cv::Mat rgb,depth;
    rgb = cv::imread("../data/image/162025180951762.png");
    depth = cv::imread("../data/depth/162025372828732.png");

    PointCloud::Ptr cloud(new PointCloud);

    // 遍历深度图
    for (int m = 0; m < depth.rows; ++m) {
        for (int n = 0; n < depth.cols; ++n) {
            // 获取深度图中(m,n)处的值
            ushort d = depth.ptr<ushort>(m)[n];
            if (d==0)
                continue;
            PointT p;

            // 计算这个点在空间中的坐标
            p.z = double(d)/camera_factor;
            p.x = (n-camera_cx)*p.z/camera_fx;
            p.y = (m-camera_cy)*p.z/camera_fy;

            // 从rgb图像中获取它的颜色
            // rgb是三通道的BGR格式图，所以按下面的顺序获取颜色
            p.b = rgb.ptr<uchar>(m)[n*3];
            p.g = rgb.ptr<uchar>(m)[n*3+1];
            p.r = rgb.ptr<uchar>(m)[n*3+2];

            cloud->points.push_back(p);
        }
    }

    cloud->height=1;
    cloud->width=cloud->points.size();
    cout<<"point cloud size = "<<cloud->points.size()<<endl;
    cloud->is_dense=false;
    pcl::io::savePCDFile("../pointcloud.pcd",*cloud);

    cloud->points.clear();
    cout<<"Point cloud saved."<<endl;

    return 0;
}