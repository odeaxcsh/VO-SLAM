
#include "opencv2/opencv.hpp"

#ifndef VO_ESTIMATOR_H
#define VO_ESTIMATOR_H

struct OdometryStatus
{
    int inlier_num;
    double inlier_ratio;
    bool successful;
    cv::Mat T;

    std::vector<cv::Point2f> pts_prev;
    std::vector<cv::Point2f> pts_curr;
    std::vector<cv::Point3f> pts_3d;

    cv::Mat inlier_mask;
};

enum class MOTION_ESTIMATION_METHOD
{
    EPIPOLAR,
    EPIPOLAR_5PT
};

class VOEstimator
{
public:
    VOEstimator(MOTION_ESTIMATION_METHOD method = MOTION_ESTIMATION_METHOD::EPIPOLAR) : method_(method) {}
    ~VOEstimator() = default;

    virtual OdometryStatus estimateMotion(const cv::Mat& img_prev, const cv::Mat& img_curr, const cv::Mat& K) = 0;

    void estimatePose(
        OdometryStatus& status,
        const cv::Mat& K
    );

    void triangulatePoints(
        OdometryStatus& status,
        const cv::Mat& K
    );

protected:
    MOTION_ESTIMATION_METHOD method_;
};


class OpticalFlowVOEstimator : public VOEstimator
{
public:
    OdometryStatus estimateMotion(const cv::Mat& img_prev, const cv::Mat& img_curr, const cv::Mat& K) override;
};


class KeyPointVOEstimator : public VOEstimator
{
public:
    enum class FeatureType {ORB, SIFT};

    OdometryStatus estimateMotion(const cv::Mat& img_prev, const cv::Mat& img_curr, const cv::Mat& K) override;

    void setFeatureType(FeatureType type);
private:
    /* Ratio for Lowe Test */
    double LoweRatio = 0.7;
    /* Feature matching algorithm */
    FeatureType type = FeatureType::ORB;
    /* Number of features for ORB and SIFT */
    int nFeatures = 10000;

    /* Orb param: scaleFactor */
    float scaleFactor = 1.2f;
    /* Orb param: nLevels */
    int nLevels = 8;
    /* Orb param: orbEdgeThreshold */
    int orbEdgeThreshold = 31;
    /* Orb param: patchSize */
    int patchSize = 31;
    /* Orb param: firstLevel */
    int firstLevel = 0;
    /* Orb param: WTA_K */
    int WTA_K = 2;

    /* Sift param: nOctaveLayers */
    int nOctaveLayers = 3;
    /* Sift param: contrastThreshold */
    double contrastThreshold = 0.04;
    /* Sift param: edgeThreshold */
    double siftEdgeThreshold = 10;
    /* Sift param: sigma */
    double sigma = 1.6;
};

// class MapBasedVOEstimator {
// public:
//     MapBasedVOEstimator(const std::vector<cv::Point3f>& map_points) : map_points_(map_points) {}
//     OdometryStatus estiamteGlobalMotion(std::vector<cv::Point2f>& pts_prev, std::vector<cv::Point2f>& pts_curr, const cv::Mat& K)
//     {
//         return status;
//     }
//
// private:
//     std::vector<cv::Point3f> map_points_;
// };



#endif // VO_ESTIMATOR_H
