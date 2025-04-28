// KeyFrameSelection.cpp
#include "keyframeselection.hpp"
#include <cmath>

KeyFrameSelection::KeyFrameSelection(const std::vector<Mode>& modes, const std::vector<double>& thresholds)
    : modes_(modes), thresholds_(thresholds)
{
}

bool KeyFrameSelection::update(const OdometryStatus& status)
{
    frame_count_++;
    auto current_time = std::chrono::steady_clock::now();
    auto time_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(current_time - start_time_).count();
    bool should_add_keyframe = false;

    for (size_t i = 0; i < modes_.size(); ++i)
    {
        switch (modes_[i])
        {
        case Mode::TIME:
            if (time_elapsed >= thresholds_[i]) {
                should_add_keyframe = true;
            }
            break;
        case Mode::FRAME_INTERVAL:
            if (frame_count_ >= static_cast<int>(thresholds_[i]))
            {
                should_add_keyframe = true;
            }
            break;
        case Mode::MOTION:
            if (checkMotion(status.T, thresholds_[i]))
            {
                should_add_keyframe = true;
            }

            break;
        case Mode::ERROR:
            if (checkError(status, thresholds_[i]))
            {
                should_add_keyframe = true;
            }
            break;
        case Mode::INLIER_RATIO:
            if (checkInlierRatio(status, thresholds_[i]))
            {
                should_add_keyframe = true;
            }
            break;
        default:
            break;
        }

        if (should_add_keyframe)
            break;
    }

    if (should_add_keyframe)
    {
        frame_count_ = 0;
        start_time_ = std::chrono::steady_clock::now();
        last_keyframe_T_ = status.T.clone();
    }

    return should_add_keyframe;
}

bool KeyFrameSelection::checkMotion(const cv::Mat& T, double translation_threshold)
{
    if (T.empty())
        return false;

    cv::Mat R = T(cv::Rect(0, 0, 3, 3));
    cv::Mat t = T(cv::Rect(3, 0, 1, 3));

    double translation_norm = cv::norm(t);
    double rotation_angle = std::acos(std::min(1.0, std::max(-1.0, (cv::trace(R)[0] - 1.0) / 2.0)));
    return translation_norm > translation_threshold || rotation_angle > 0.087; // Example threshold
}

bool KeyFrameSelection::checkError(const OdometryStatus& status, double threshold)
{
    if (status.pts_prev.empty() || status.pts_curr.empty())
        return false;

    double error_sum = 0.0;
    for (size_t i = 0; i < status.pts_prev.size(); ++i)
    {
        error_sum += cv::norm(status.pts_prev[i] - status.pts_curr[i]);
    }

    double avg_error = error_sum / status.pts_prev.size();
    return avg_error > threshold;
}


bool KeyFrameSelection::checkInlierRatio(const OdometryStatus& status, double threshold)
{
    return status.inlier_ratio < threshold; 
}
