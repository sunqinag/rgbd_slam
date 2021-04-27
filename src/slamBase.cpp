//
// Created by xtcsun on 2021/4/27.
//

#include "slamBase.h"
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

PointCloud::Ptr image2PointCloud(cv::Mat& rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera)
{
    PointCloud::Ptr cloud(new PointCloud);

    for (int m = 0; m < depth.rows; ++m) {
        for (int n = 0; n < depth.cols; ++n) {
            // 获取深度图中(m,n)处的值
            ushort d = depth.ptr<ushort>(m)[n];
            // d 可能没有值，若此，跳过此点
            if (d==0)
                continue;

            // d 存在值，则向点云增加一个点
            PointT p;

            // 计算这个点的空间坐标
            p.z = double(d) /camera.scale;
            p.x= (n-camera.cx)*p.z/camera.fx;
            p.y = (m-camera.cy)*p.z/camera.fy;

            // 从rgb图像中获取它的颜色
            // rgb是三通道的BGR格式图，所以按下面的顺序获取颜色
            p.b = rgb.ptr<uchar>(m)[n*3];
            p.g = rgb.ptr<uchar>(m)[n*3+1];
            p.r = rgb.ptr<uchar>(m)[n*3+2];

            cloud->points.push_back(p);
        }
    }

    //设置并保存点云
    cloud->height = 1;
    cloud->width = cloud->points.size();
    cloud->is_dense=false;

    return cloud;
}

cv::Point3f point2dTo3d(cv::Point3f& point, CAMERA_INTRINSIC_PARAMETERS& camera)
{
    cv::Point3f p;// 3D 点
    p.z = double(point.z)/camera.scale;
    p.x = (point.x-camera.cx)*p.z/camera.fx;
    p.y = (point.y-camera.cy)*p.z/camera.fy;
    return p;
}

void computeKeyPointsAndDesp( FRAME& frame )
{
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();

    detector->detect(frame.rgb,frame.kp);
    descriptor->compute(frame.rgb,frame.kp,frame.desp);
}

//estimateMotion 计算两个帧之间的运动
RESULT_OF_PNP estimateMotion( FRAME& frame1, FRAME& frame2, CAMERA_INTRINSIC_PARAMETERS& camera )
{
    static ParameterReader pd;
    vector<DMatch> matches;
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    // BFMatcher matcher ( NORM_HAMMING );
    matcher->match(frame1.desp, frame2.desp, matches);
    cout << "Find total " << matches.size() << " matches." << endl;

    // 匹配筛选，把距离最大的去掉，去掉大于四倍最小距离的匹配
    vector<DMatch> goodMatches;
    double minDis = 9999;
    for (size_t i = 0; i < matches.size(); i++) {

        if (matches[i].distance < minDis && matches[i].distance != 0)
            minDis = matches[i].distance;
    }

    for (size_t i = 0; i < matches.size(); i++) {
        if (matches[i].distance < 4 * minDis)
            goodMatches.push_back(matches[i]);
    }
    cout << "good matches=" << goodMatches.size() << endl;

    // 第一帧的三维点
    vector<Point3f> pts_obj;
    // 第二帧的图像点
    vector<Point2f> pts_img;

    // 相机内参
    for (size_t i=0; i<goodMatches.size(); i++)
    {
        // query 是第一个, train 是第二个
        cv::Point2f p = frame1.kp[goodMatches[i].queryIdx].pt;
        // 获取d是要小心！x是向右的，y是向下的，所以y才是行，x是列！
        ushort d = frame1.depth.ptr<ushort>( int(p.y) )[ int(p.x) ];
        if (d == 0)
            continue;
        pts_img.push_back( cv::Point2f( frame2.kp[goodMatches[i].trainIdx].pt ) );

        // 将(u,v,d)转成(x,y,z)
        cv::Point3f pt ( p.x, p.y, d );
        cv::Point3f pd = point2dTo3d( pt, camera );
        pts_obj.push_back( pd );
    }

    double camera_matrix_data[3][3] = {
            {camera.fx, 0, camera.cx},
            {0, camera.fy, camera.cy},
            {0, 0, 1}
    };

    cout<<"solving pnp"<<endl;

    // 构建相机矩阵
    cv::Mat cameraMatrix( 3, 3, CV_64F, camera_matrix_data );
    cv::Mat rvec, tvec, inliers;
    // 求解pnp
    cv::solvePnPRansac( pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 8.0, 0.99, inliers );
    RESULT_OF_PNP result;
    result.rvec = rvec;
    result.tvec = tvec;
    result.inliers = inliers.rows;

    return result;

}