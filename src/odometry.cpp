#include "odometry.hpp"


void VOEstimator::estimatePose(
    OdometryStatus& status,
    const cv::Mat& K
) {
    double f = K.at<double>(0, 0);
    cv::Point2d c(K.at<double>(0, 2), K.at<double>(1, 2));

    cv::Mat E;
    switch(method_) {
    case MOTION_ESTIMATION_METHOD::EPIPOLAR: {
        cv::Mat F = cv::findFundamentalMat(status.pts_prev, status.pts_curr, cv::FM_RANSAC, 0.999, 1, status.inlier_mask);
        E = K.t() * F * K;
        break;
    }
    case MOTION_ESTIMATION_METHOD::EPIPOLAR_5PT: {
        E = cv::findEssentialMat(status.pts_prev, status.pts_curr, f, c, cv::RANSAC, 0.999, 1, status.inlier_mask);
        break;
    }
    default:
        throw std::invalid_argument("Invalid motion estimation method");
    }

    cv::Mat R, t;
    status.inlier_num = cv::recoverPose(E, status.pts_prev, status.pts_curr, R, t, f, c, status.inlier_mask);
    status.T = cv::Mat::eye(4, 4, CV_64F);
    R.copyTo(status.T(cv::Rect(0, 0, 3, 3)));
    t.copyTo(status.T.col(3).rowRange(0, 3));
    status.inlier_ratio = static_cast<double>(status.inlier_num) / static_cast<double>(status.pts_prev.size());
    status.successful = (status.inlier_num > 100) && (status.inlier_ratio > 0.2);
}


void VOEstimator::triangulatePoints(
    OdometryStatus& status,
    const cv::Mat& K
) {
    status.pts_3d.clear();

    if (status.pts_prev.size() != status.pts_curr.size()) {
        throw std::invalid_argument("The number of points in the previous and current images must be the same.");
    }

    if(status.pts_prev.size() < 5 || !status.successful) {
        return;
    }   
        
    cv::Mat P1 = K * cv::Mat::eye(3, 4, CV_64F);
    cv::Mat Rt = status.T(cv::Rect(0, 0, 4, 3));
    cv::Mat P2 = K * Rt;

    cv::Mat points4D;
    cv::triangulatePoints(P1, P2, status.pts_prev, status.pts_curr, points4D);
    for (int i = 0; i < points4D.cols; ++i)
    {
        cv::Mat col = points4D.col(i);
        col /= col.at<float>(3);

        if(col.at<float>(2) > 0 && col.at<float>(2) < 100) {
            status.pts_3d.emplace_back(
                col.at<float>(0),
                col.at<float>(1),
                col.at<float>(2)
            );
        }
    }

}


OdometryStatus OpticalFlowVOEstimator::estimateMotion(const cv::Mat& img_prev, const cv::Mat& img_curr, const cv::Mat& K)
{
    double f = K.at<double>(0, 0);
    cv::Point2d c(K.at<double>(0, 2), K.at<double>(1, 2));


    std::vector<cv::Point2f> pts_prev, pts_curr;
    cv::goodFeaturesToTrack(img_prev, pts_prev, 10000, 0.001, 3);
    std::vector<uchar> trash;
    cv::Mat err;
    cv::calcOpticalFlowPyrLK(img_prev, img_curr, pts_prev, pts_curr, trash, err);

    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < pts_prev.size(); i++) {
        if (trash[i] == 1) {
            good_matches.push_back(cv::DMatch(i, i, 0));
        }
    }
    pts_prev.resize(good_matches.size());
    pts_curr.resize(good_matches.size());
    for (size_t i = 0; i < good_matches.size(); i++) {
        pts_prev[i] = pts_prev[good_matches[i].queryIdx];
        pts_curr[i] = pts_curr[good_matches[i].trainIdx];
    }


    OdometryStatus status;
    status.pts_prev = pts_prev;
    status.pts_curr = pts_curr;
    estimatePose(status, K);
    triangulatePoints(status, K);
    return status;
}


OdometryStatus KeyPointVOEstimator::estimateMotion(const cv::Mat& img_prev, const cv::Mat& img_curr, const cv::Mat& K)
{
    double f = K.at<double>(0, 0);
    cv::Point2d c(K.at<double>(0, 2), K.at<double>(1, 2));

    std::vector<cv::KeyPoint> keypoints_prev, keypoints_curr;
    cv::Mat descriptors_prev, descriptors_curr;

    if (type == FeatureType::ORB) {
        cv::Ptr<cv::ORB> orb = cv::ORB::create(nFeatures, scaleFactor, nLevels, orbEdgeThreshold, firstLevel, WTA_K, cv::ORB::HARRIS_SCORE, patchSize);
        orb->detectAndCompute(img_prev, cv::noArray(), keypoints_prev, descriptors_prev);
        orb->detectAndCompute(img_curr, cv::noArray(), keypoints_curr, descriptors_curr);
    } else if (type == FeatureType::SIFT) {
        cv::Ptr<cv::SIFT> sift = cv::SIFT::create(nFeatures, nOctaveLayers, contrastThreshold, siftEdgeThreshold, sigma);
        sift->detectAndCompute(img_prev, cv::noArray(), keypoints_prev, descriptors_prev);
        sift->detectAndCompute(img_curr, cv::noArray(), keypoints_curr, descriptors_curr);
    }

    std::vector<std::vector<cv::DMatch>> knn_matches;
    cv::BFMatcher matcher((type == FeatureType::ORB) ? cv::NORM_HAMMING : cv::NORM_L2); 
    matcher.knnMatch(descriptors_prev, descriptors_curr, knn_matches, 2);
    
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i].size() < 2)
            continue;

        if (knn_matches[i][0].distance < LoweRatio * knn_matches[i][1].distance)
            good_matches.push_back(knn_matches[i][0]);
    }
    
    std::vector<cv::Point2f> pts_prev(good_matches.size()), pts_curr(good_matches.size());
    for (size_t i = 0; i < good_matches.size(); i++) {
        pts_prev[i] = keypoints_prev[good_matches[i].queryIdx].pt;
        pts_curr[i] = keypoints_curr[good_matches[i].trainIdx].pt;
    }

    std::vector<cv::DMatch> initial_good_matches = good_matches;
    good_matches.clear();
    {
        // Backward match
        std::vector<std::vector<cv::DMatch>> knn_matches_back;
        matcher.knnMatch(descriptors_curr, descriptors_prev, knn_matches_back, 2);

        for (const auto& m : initial_good_matches) {
            if (m.trainIdx >= (int)knn_matches_back.size())
                continue;
            if (knn_matches_back[m.trainIdx].size() < 1)
                continue;
            const auto& m_back = knn_matches_back[m.trainIdx][0];
            if (m_back.trainIdx == m.queryIdx) {
                good_matches.push_back(m);
            }
        }
    }

    pts_prev.resize(good_matches.size());
    pts_curr.resize(good_matches.size());
    for (size_t i = 0; i < good_matches.size(); i++) {
        pts_prev[i] = keypoints_prev[good_matches[i].queryIdx].pt;
        pts_curr[i] = keypoints_curr[good_matches[i].trainIdx].pt;
    }


    OdometryStatus status;
    status.pts_prev = pts_prev;
    status.pts_curr = pts_curr;
    estimatePose(status, K);
    triangulatePoints(status, K);
    return status;
}

void KeyPointVOEstimator::setFeatureType(FeatureType type) {
    this->type = type;
}


