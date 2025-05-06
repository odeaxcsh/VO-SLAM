#include <opencv2/opencv.hpp>
#include <cmath>

// Utils
// RH: Borrowing from the real-time pose estimation tutorial in opencv
cv::Mat rot2euler(const cv::Mat & rotationMatrix)
{
    cv::Mat euler(3,1,CV_64F);

    double m00 = rotationMatrix.at<double>(0,0);
    double m02 = rotationMatrix.at<double>(0,2);
    double m10 = rotationMatrix.at<double>(1,0);
    double m11 = rotationMatrix.at<double>(1,1);
    double m12 = rotationMatrix.at<double>(1,2);
    double m20 = rotationMatrix.at<double>(2,0);
    double m22 = rotationMatrix.at<double>(2,2);

    double bank, attitude, heading;

    // Assuming the angles are in radians.
    if (m10 > 0.998) { // singularity at north pole
        bank = 0;
        attitude = CV_PI/2;
        heading = atan2(m02,m22);
    }
    else if (m10 < -0.998) { // singularity at south pole
        bank = 0;
        attitude = -CV_PI/2;
        heading = atan2(m02,m22);
    }
    else
    {
        bank = atan2(-m12,m11);
        attitude = asin(m10);
        heading = atan2(-m20,m00);
    }

    euler.at<double>(0) = bank;
    euler.at<double>(1) = attitude;
    euler.at<double>(2) = heading;

    return euler;
}

// Converts a given Euler angles to Rotation Matrix
// Convention used is Y-Z-X Tait-Bryan angles
// Reference:
// https://www.euclideanspace.com/maths/geometry/rotations/conversions/eulerToMatrix/index.htm
cv::Mat euler2rot(const cv::Mat & euler)
{
    cv::Mat rotationMatrix(3,3,CV_64F);

    double bank = euler.at<double>(0);
    double attitude = euler.at<double>(1);
    double heading = euler.at<double>(2);

    // Assuming the angles are in radians.
    double ch = cos(heading);
    double sh = sin(heading);
    double ca = cos(attitude);
    double sa = sin(attitude);
    double cb = cos(bank);
    double sb = sin(bank);

    double m00, m01, m02, m10, m11, m12, m20, m21, m22;

    m00 = ch * ca;
    m01 = sh*sb - ch*sa*cb;
    m02 = ch*sa*sb + sh*cb;
    m10 = sa;
    m11 = ca*cb;
    m12 = -ca*sb;
    m20 = -sh*ca;
    m21 = sh*sa*cb + ch*sb;
    m22 = -sh*sa*sb + ch*cb;

    rotationMatrix.at<double>(0,0) = m00;
    rotationMatrix.at<double>(0,1) = m01;
    rotationMatrix.at<double>(0,2) = m02;
    rotationMatrix.at<double>(1,0) = m10;
    rotationMatrix.at<double>(1,1) = m11;
    rotationMatrix.at<double>(1,2) = m12;
    rotationMatrix.at<double>(2,0) = m20;
    rotationMatrix.at<double>(2,1) = m21;
    rotationMatrix.at<double>(2,2) = m22;

    return rotationMatrix;
}

//////////////////////////////////////////////////////////////////////////
// RH: Kalman Filter (Full Pose) from:
// https://github.com/opencv/opencv/blob/16a3d37dc159dbcaaf8ee74cf63669f0203f9655/samples/cpp/tutorial_code/calib3d/real_time_pose_estimation/src/main_detection.cpp#L392


// Need to declare our KF, then init with this func, then use the update function instead of our separate predict/correct funcs
void initKalmanFilter(cv::KalmanFilter &KF, int nStates, int nMeasurements, int nInputs, double dt)
{
    KF.init(nStates, nMeasurements, nInputs, CV_64F);                 // init Kalman Filter

    setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-5));       // set process noise
    setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-2));   // set measurement noise
    setIdentity(KF.errorCovPost, cv::Scalar::all(1));             // error covariance

    /** DYNAMIC MODEL (18 parameters) **/

    //  [1 0 0 dt  0  0 dt2   0   0 0 0 0  0  0  0   0   0   0]
    //  [0 1 0  0 dt  0   0 dt2   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 1  0  0 dt   0   0 dt2 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  1  0  0  dt   0   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  1  0   0  dt   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  1   0   0  dt 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  0   1   0   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  0   0   1   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  0   0   0   1 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  0   0   0   0 1 0 0 dt  0  0 dt2   0   0]
    //  [0 0 0  0  0  0   0   0   0 0 1 0  0 dt  0   0 dt2   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 1  0  0 dt   0   0 dt2]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  1  0  0  dt   0   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  1  0   0  dt   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  1   0   0  dt]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   1   0   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   1   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   0   1]

    // position
    KF.transitionMatrix.at<double>(0,3) = dt;
    KF.transitionMatrix.at<double>(1,4) = dt;
    KF.transitionMatrix.at<double>(2,5) = dt;
    KF.transitionMatrix.at<double>(3,6) = dt;
    KF.transitionMatrix.at<double>(4,7) = dt;
    KF.transitionMatrix.at<double>(5,8) = dt;
    KF.transitionMatrix.at<double>(0,6) = 0.5*pow(dt,2);
    KF.transitionMatrix.at<double>(1,7) = 0.5*pow(dt,2);
    KF.transitionMatrix.at<double>(2,8) = 0.5*pow(dt,2);

    // orientation
    KF.transitionMatrix.at<double>(9,12) = dt;
    KF.transitionMatrix.at<double>(10,13) = dt;
    KF.transitionMatrix.at<double>(11,14) = dt;
    KF.transitionMatrix.at<double>(12,15) = dt;
    KF.transitionMatrix.at<double>(13,16) = dt;
    KF.transitionMatrix.at<double>(14,17) = dt;
    KF.transitionMatrix.at<double>(9,15) = 0.5*pow(dt,2);
    KF.transitionMatrix.at<double>(10,16) = 0.5*pow(dt,2);
    KF.transitionMatrix.at<double>(11,17) = 0.5*pow(dt,2);


    /** MEASUREMENT MODEL **/

    //  [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    //  [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    //  [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    //  [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]
    //  [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
    //  [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0]

    KF.measurementMatrix.at<double>(0,0) = 1;  // x
    KF.measurementMatrix.at<double>(1,1) = 1;  // y
    KF.measurementMatrix.at<double>(2,2) = 1;  // z
    KF.measurementMatrix.at<double>(3,9) = 1;  // roll
    KF.measurementMatrix.at<double>(4,10) = 1; // pitch
    KF.measurementMatrix.at<double>(5,11) = 1; // yaw
}

/**********************************************************************************************************/
void updateKalmanFilter( cv::KalmanFilter &KF, cv::Mat &measurement,
                        cv::Mat &translation_estimated,cv::Mat &rotation_estimated )
{
    // First predict, to update the internal statePre variable
    cv::Mat prediction = KF.predict();

    // The "correct" phase that is going to use the predicted value and our measurement
    cv::Mat estimated = KF.correct(measurement);

    // Estimated translation
    translation_estimated.at<double>(0) = estimated.at<double>(0);
    translation_estimated.at<double>(1) = estimated.at<double>(1);
    translation_estimated.at<double>(2) = estimated.at<double>(2);

    // Estimated euler angles
    cv::Mat eulers_estimated(3, 1, CV_64F);
    eulers_estimated.at<double>(0) = estimated.at<double>(9);
    eulers_estimated.at<double>(1) = estimated.at<double>(10);
    eulers_estimated.at<double>(2) = estimated.at<double>(11);

    // Convert estimated quaternion to rotation matrix
    rotation_estimated = euler2rot(eulers_estimated);
}

/**********************************************************************************************************/
void fillMeasurements( cv::Mat &measurements,
                       const cv::Mat &translation_measured, const cv::Mat &rotation_measured)
{
    // Convert rotation matrix to euler angles
    cv::Mat measured_eulers(3, 1, CV_64F);
    measured_eulers = rot2euler(rotation_measured);

    // Set measurement to predict
    measurements.at<double>(0) = translation_measured.at<double>(0); // x
    measurements.at<double>(1) = translation_measured.at<double>(1); // y
    measurements.at<double>(2) = translation_measured.at<double>(2); // z
    measurements.at<double>(3) = measured_eulers.at<double>(0);      // roll
    measurements.at<double>(4) = measured_eulers.at<double>(1);      // pitch
    measurements.at<double>(5) = measured_eulers.at<double>(2);      // yaw
}


//////////////////////////////////////////////////////////////////////////
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

//////////////////////////////////////////////////////////////////////////////////////////////////////
// RH: Implementation of EKF with full Pose prediction and correction
//////////////////////////////////////////////////////////////////////////////////////////////////////

//#include <iostream>
//#include <cmath>
//
//class EKF3D {
//public:
//    using Vector6d = Eigen::Matrix<double, 6, 1>;
//    using Matrix6d = Eigen::Matrix<double, 6, 6>;
//    using Matrix3x6d = Eigen::Matrix<double, 3, 6>;
//    using Vector3d = Eigen::Vector3d;
//    using Matrix3d = Eigen::Matrix3d;
//
//    Vector6d x;   // [x, y, z, roll, pitch, yaw]
//    Matrix6d P;
//    Matrix6d Q;
//    Matrix3d R;
//
//    EKF3D() {
//        x.setZero();
//        P.setIdentity();
//        Q = Matrix6d::Identity() * 0.01;
//        R = Matrix3d::Identity() * 0.1;
//    }
//
//    Matrix3d rotationMatrix(double roll, double pitch, double yaw) {
//        using namespace Eigen;
//        return AngleAxisd(yaw, Vector3d::UnitZ()) *
//               AngleAxisd(pitch, Vector3d::UnitY()) *
//               AngleAxisd(roll, Vector3d::UnitX());
//    }
//
//    Matrix3d dR_droll(double roll, double pitch, double yaw) {
//        using namespace Eigen;
//        double cr = cos(roll), sr = sin(roll);
//        double cp = cos(pitch), sp = sin(pitch);
//        double cy = cos(yaw), sy = sin(yaw);
//
//        Matrix3d dRx;
//        dRx << 0, 0, 0,
//               0, -sr, -cr,
//               0, cr, -sr;
//
//        Matrix3d Ry;
//        Ry << cp, 0, sp,
//              0, 1, 0,
//              -sp, 0, cp;
//
//        Matrix3d Rz;
//        Rz << cy, -sy, 0,
//              sy, cy, 0,
//              0, 0, 1;
//
//        return Rz * Ry * dRx;
//    }
//
//    Matrix3d dR_dpitch(double roll, double pitch, double yaw) {
//        using namespace Eigen;
//        double cr = cos(roll), sr = sin(roll);
//        double cp = cos(pitch), sp = sin(pitch);
//        double cy = cos(yaw), sy = sin(yaw);
//
//        Matrix3d Rx;
//        Rx << 1, 0, 0,
//              0, cr, -sr,
//              0, sr, cr;
//
//        Matrix3d dRy;
//        dRy << -sp, 0, cp,
//                0, 0, 0,
//               -cp, 0, -sp;
//
//        Matrix3d Rz;
//        Rz << cy, -sy, 0,
//              sy, cy, 0,
//              0, 0, 1;
//
//        return Rz * dRy * Rx;
//    }
//
//    Matrix3d dR_dyaw(double roll, double pitch, double yaw) {
//        using namespace Eigen;
//        double cr = cos(roll), sr = sin(roll);
//        double cp = cos(pitch), sp = sin(pitch);
//        double cy = cos(yaw), sy = sin(yaw);
//
//        Matrix3d Rx;
//        Rx << 1, 0, 0,
//              0, cr, -sr,
//              0, sr, cr;
//
//        Matrix3d Ry;
//        Ry << cp, 0, sp,
//              0, 1, 0,
//              -sp, 0, cp;
//
//        Matrix3d dRz;
//        dRz << -sy, -cy, 0,
//                cy, -sy, 0,
//                0, 0, 0;
//
//        return dRz * Ry * Rx;
//    }
//
//    void predict(const Vector6d& u, double dt) {
//        double roll = x(3), pitch = x(4), yaw = x(5);
//        Eigen::Vector3d v(u(0), u(1), u(2));
//        Eigen::Vector3d omega(u(3), u(4), u(5));
//
//        Matrix3d R = rotationMatrix(roll, pitch, yaw);
//        Vector3d delta_pos = R * v * dt;
//
//        x.segment<3>(0) += delta_pos;
//        x.segment<3>(3) += omega * dt;
//
//        // Normalize angles
//        for (int i = 3; i < 6; ++i)
//            x(i) = atan2(sin(x(i)), cos(x(i)));
//
//        // Jacobian F
//        Matrix6d F = Matrix6d::Identity();
//
//        // Position wrt orientation
//        Matrix3d dR_dphi = dR_droll(roll, pitch, yaw);
//        Matrix3d dR_dtheta = dR_dpitch(roll, pitch, yaw);
//        Matrix3d dR_dpsi = dR_dyaw(roll, pitch, yaw);
//
//        Vector3d v_dt = v * dt;
//        F.block<3, 3>(0, 3) <<
//            dR_dphi * v_dt, dR_dtheta * v_dt, dR_dpsi * v_dt;
//
//        // Covariance update
//        P = F * P * F.transpose() + Q;
//    }
//
//    void update(const Vector3d& z) {
//        Matrix3x6d H = Matrix3x6d::Zero();
//        H(0, 0) = 1;
//        H(1, 1) = 1;
//        H(2, 2) = 1;
//
//        Vector3d z_pred = x.segment<3>(0);
//        Vector3d y = z - z_pred;
//
//        Eigen::Matrix3d S = H * P * H.transpose() + R;
//        Eigen::Matrix<double, 6, 3> K = P * H.transpose() * S.inverse();
//
//        x = x + K * y;
//        P = (Matrix6d::Identity() - K * H) * P;
//    }
//
//    void printState() const {
//        std::cout << "Position: [x=" << x(0) << ", y=" << x(1) << ", z=" << x(2) << "]\n";
//        std::cout << "Orientation (roll=" << x(3) << ", pitch=" << x(4) << ", yaw=" << x(5) << ")\n";
//    }
//};


