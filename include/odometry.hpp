
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
    OdometryStatus estimateMotion(const cv::Mat& img_prev, const cv::Mat& img_curr, const cv::Mat& K) override;
};



class MapBasedVOEstimator
{
public:
    MapBasedVOEstimator(const std::vector<cv::Point3f>& map_points) : map_points_(map_points) {}
    OdometryStatus estiamteGlobalMotion(std::vector<cv::Point2f>& pts_prev, std::vector<cv::Point2f>& pts_curr, const cv::Mat& K);

private:
    std::vector<cv::Point3f> map_points_;
};




#endif // VO_ESTIMATOR_H
