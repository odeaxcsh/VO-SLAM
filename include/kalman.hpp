#include <opencv2/opencv.hpp>

class MotionKalmanFilter
{
public:
    MotionKalmanFilter(double dt = 1.0/30.0)
    {
        dt_ = dt;
        kf_ = cv::KalmanFilter(9, 3, 0);

        // Transition matrix
        kf_.transitionMatrix = (cv::Mat_<float>(9, 9) <<
            1,0,0, dt_,0,0, 0.5*dt_*dt_,0,0,
            0,1,0, 0,dt_,0, 0,0.5*dt_*dt_,0,
            0,0,1, 0,0,dt_, 0,0,0.5*dt_*dt_,
            0,0,0, 1,0,0, dt_,0,0,
            0,0,0, 0,1,0, 0,dt_,0,
            0,0,0, 0,0,1, 0,0,dt_,
            0,0,0, 0,0,0, 1,0,0,
            0,0,0, 0,0,0, 0,1,0,
            0,0,0, 0,0,0, 0,0,1
        );

        // Measurement matrix: We only observe position
        kf_.measurementMatrix = (cv::Mat_<float>(3,9) <<
            1,0,0,0,0,0,0,0,0,
            0,1,0,0,0,0,0,0,0,
            0,0,1,0,0,0,0,0,0
        );

        // Process noise (small for now)
        setIdentity(kf_.processNoiseCov, cv::Scalar::all(1e-3));
        // Measurement noise
        setIdentity(kf_.measurementNoiseCov, cv::Scalar::all(1e-2));
        // Posteriori error covariance
        setIdentity(kf_.errorCovPost, cv::Scalar::all(1));
    }

    void initialize(const cv::Point3d& initial_pos)
    {
        kf_.statePost = (cv::Mat_<float>(9,1) << 
            initial_pos.x, initial_pos.y, initial_pos.z,
            0, 0, 0,
            0, 0, 0);
    }

    cv::Point3d predict()
    {
        cv::Mat prediction = kf_.predict();
        return cv::Point3d(prediction.at<float>(0), prediction.at<float>(1), prediction.at<float>(2));
    }

    cv::Point3d correct(const cv::Point3d& measurement)
    {
        cv::Mat meas(3, 1, CV_32F);
        meas.at<float>(0) = measurement.x;
        meas.at<float>(1) = measurement.y;
        meas.at<float>(2) = measurement.z;

        cv::Mat corrected = kf_.correct(meas);
        return cv::Point3d(corrected.at<float>(0), corrected.at<float>(1), corrected.at<float>(2));
    }

private:
    cv::KalmanFilter kf_;
    double dt_;
};
