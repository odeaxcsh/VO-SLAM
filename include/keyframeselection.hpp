#ifndef KEYFRAME_SELECTION_H
#define KEYFRAME_SELECTION_H


#include <vector>
#include <opencv2/core.hpp>

#include "odometry.hpp"

enum class Mode { TIME, FRAME_INTERVAL, MOTION, ERROR, INLIER_RATIO };


class KeyFrameSelection
{
public:
    KeyFrameSelection(const std::vector<Mode>& modes, const std::vector<double>& thresholds);
    bool update(const OdometryStatus& status);

private:
    bool checkMotion(const cv::Mat& T, double translation_threshold);
    bool checkError(const OdometryStatus& status, double threshold);
    bool checkInlierRatio(const OdometryStatus& status, double threshold);

private:
    std::vector<Mode> modes_;
    std::vector<double> thresholds_;
    int frame_count_ = 0;
    std::chrono::steady_clock::time_point start_time_;
    cv::Mat last_keyframe_T_;
};

#endif
