//
// Created by xtcsun on 2021/4/27.
//

#include <iostream>
#include "slamBase.h"

using namespace std;

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;

int main() {
    Mat rgb1 = cv::imread("../data/image_registration/color/1.png");
    Mat rgb2 = cv::imread("../data/image_registration/color/2.png");
    Mat depth1 = cv::imread("../data/image_registration/depth/depth1.png", -1);
    Mat depth2 = cv::imread("../data/image_registration/depth/depth2.png", -1);


    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();

    vector<KeyPoint> kp1, kp2;
    detector->detect(rgb1, kp1);
    detector->detect(rgb2, kp2);

    cout << "Key points if two images; " << kp1.size() << ", " << kp2.size() << endl;

    // 可视化显示关键点
    Mat imgShow;
    cv::drawKeypoints(rgb1, kp1, imgShow, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow("keypoints", imgShow);
    cv::imwrite("../data/keypoints.png", imgShow);
    cv::waitKey(0);


    // 计算描述子
    Mat desp1, desp2;
    descriptor->compute(rgb1, kp1, desp1);
    descriptor->compute(rgb2, kp2, desp2);

    //匹配描述子
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
    // BFMatcher matcher ( NORM_HAMMING );
    matcher->match(desp1, desp2, match);
    cout << "Find total " << match.size() << " matches." << endl;

    // 可视化：显示匹配的特征
    cv::Mat imgMatches;
    cv::drawMatches(rgb1, kp1, rgb2, kp2, match, imgMatches);
    cv::imshow("matches", imgMatches);
    cv::imwrite("./data/matches.png", imgMatches);
    cv::waitKey(0);

    // 匹配筛选，把距离最大的去掉，去掉大于四倍最小距离的匹配
    vector<DMatch> goodMatches;
    double minDis = 9999;
    for (size_t i = 0; i < match.size(); i++) {
//        cout << "match distance " << match[i].distance << endl;
        if (match[i].distance < minDis && match[i].distance != 0)
            minDis = match[i].distance;
    }

    for (size_t i = 0; i < match.size(); i++) {
        if (match[i].distance < 4 * minDis)
            goodMatches.push_back(match[i]);
    }

    // 显示 good matches
    cout << "good matches=" << goodMatches.size() << endl;
    cv::drawMatches(rgb1, kp1, rgb2, kp2, goodMatches, imgMatches);
    cv::imshow("good matches", imgMatches);
    cv::imwrite("./data/good_matches.png", imgMatches);
    cv::waitKey(0);


    // 计算图像间的运动关系
    // 关键函数：cv::solvePnPRansac()
    // 为调用此函数准备必要的参数

    vector<cv::Point3f> pts_obj;
    vector<cv::Point2f> pts_img;

    CAMERA_INTRINSIC_PARAMETERS C;
    C.cx = 325.5;
    C.cy = 253.5;
    C.fx = 518.0;
    C.fy = 519.0;
    C.scale = 1000.0;

    for (int i = 0; i < goodMatches.size(); ++i) {
        Point2f p = kp1[goodMatches[i].queryIdx].pt;
        ushort d = depth1.ptr<ushort>(int(p.y))[int(p.x)];
        if (d == 0)
            continue;
        pts_img.push_back(Point2f(kp2[goodMatches[i].trainIdx].pt));

        Point3f pt(p.x, p.y, d);
        Point3f pd = point2dTo3d(pt, C);
        pts_obj.push_back(pd);
    }

    double camera_matrix_data[3][3] = {
            {C.fx, 0,    C.cx},
            {0,    C.fy, C.cy},
            {0,    0,    1}
    };

    // 构建相机矩阵
    Mat cameraMatrix(3, 3, CV_64F, camera_matrix_data);
    Mat rvec, tvec, inliers;
    // 求解PnP
//    cv::solvePnP(pts_obj, pts_img, cameraMatrix, Mat(), rvec, tvec, false);
    cv::solvePnPRansac( pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 8.0, 0.99, inliers );
    cout << "inliers: " << inliers.rows << endl;
    cout << "R=" << rvec << endl;
    cout << "t=" << tvec << endl;

    // 画出inliers匹配
    vector<DMatch> matchesShow;
    for (int i = 0; i < inliers.rows; ++i) {
        matchesShow.push_back(goodMatches[inliers.ptr<int>(i)[0]]);
    }

    drawMatches(rgb1, kp1, rgb2, kp2, matchesShow, imgMatches);
    cv::imshow("inlier matches", imgMatches);
    cv::imwrite("../data/inliers.png", imgMatches);
    cv::waitKey(0);
    return 0;
}