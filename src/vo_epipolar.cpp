#include <iostream>
#include <vector>
#include <string>
#include "opencv2/opencv.hpp"
#include "opencv2/viz.hpp" // OpenCV Viz module for 3D visulalization

#include "kalman.hpp"

#include "odometry.hpp"
#include "keyframeselection.hpp"


#include <optional>


struct TrajectoryVisualization {
    cv::viz::WCloud points;
    std::vector<cv::viz::WLine> lines;
};

std::optional<TrajectoryVisualization> visualizeTrajectory(const std::vector<cv::Point3d>& trajectory,
                                              double opacity,
                                              double line_width,
                                              double point_size,
                                              const cv::viz::Color& color)
{
    
    if(trajectory.empty()) {
        return std::nullopt;
    }

    cv::Mat traj_cloud(static_cast<int>(trajectory.size()), 1, CV_32FC3);
    for (size_t i = 0; i < trajectory.size(); ++i) {
        traj_cloud.at<cv::Vec3f>(static_cast<int>(i), 0) =
            cv::Vec3f(static_cast<float>(trajectory[i].x),
                      static_cast<float>(trajectory[i].y),
                      static_cast<float>(trajectory[i].z));
    }


    
    auto points = cv::viz::WCloud(traj_cloud);
    TrajectoryVisualization vis = {points, {}};

    vis.points.setRenderingProperty(cv::viz::POINT_SIZE, point_size);
    vis.points.setRenderingProperty(cv::viz::OPACITY, opacity);
    vis.points.setColor(color);

    if(line_width <= 0) {
        vis.lines.clear(); // No lines to draw
        return vis; 
    }

    // Create line widgets connecting successive trajectory points.
    for (size_t i = 0; i + 1 < trajectory.size(); ++i) {
        cv::Point3f pt1(static_cast<float>(trajectory[i].x),
                        static_cast<float>(trajectory[i].y),
                        static_cast<float>(trajectory[i].z));

        cv::Point3f pt2(static_cast<float>(trajectory[i+1].x),
                        static_cast<float>(trajectory[i+1].y),
                        static_cast<float>(trajectory[i+1].z));

        cv::viz::WLine line_widget(pt1, pt2);
        line_widget.setRenderingProperty(cv::viz::LINE_WIDTH, line_width);
        line_widget.setColor(color);
        vis.lines.push_back(line_widget);
    }
    
    return vis;
}

void drawTrajectory(cv::viz::Viz3d& viz_window, const TrajectoryVisualization& vis, std::string name)
{
    viz_window.showWidget(name, vis.points);    
    for (size_t i = 0; i < vis.lines.size(); ++i) {
        viz_window.showWidget(name + " Line" + std::to_string(i), vis.lines[i]);
    }
}




int main()
{
    const char* video_file = "../data/KITTI07/image_0/%06d.png";
    double f = 707.0912;
    cv::Point2d c(601.8873, 183.1104);
    bool use_5pt = true;
    int min_inlier_num = 100;
    double min_inlier_ratio = 0.2;
    const char* traj_file = "vo_epipolar.xyz";

    bool initialized = false;
    // MotionKalmanFilter kalman_filter;
    cv::KalmanFilter kalman_filter;

    cv::VideoCapture video;
    if (!video.open(video_file)) return -1;

    cv::Mat gray_prev;
    video >> gray_prev;
    if (gray_prev.empty()) {
        video.release();
        return -1;
    }
    if (gray_prev.channels() > 1) cv::cvtColor(gray_prev, gray_prev, cv::COLOR_RGB2GRAY);

    cv::Mat K = (cv::Mat_<double>(3, 3) << f, 0, c.x, 0, f, c.y, 0, 0, 1);
    cv::Mat camera_pose = cv::Mat::eye(4, 4, CV_64F);
    FILE* camera_traj = fopen(traj_file, "wt");
    if (!camera_traj) return -1;
    
    // 3D Visualization
    cv::viz::Viz3d viz_window("Camera Trajectory");

    std::vector<cv::Point3d> trajectory;
    std::vector<cv::Point3d> points_3d;
    std::vector<cv::Point3d> original_traj, predicted_points_3d;
    
    cv::viz::WCoordinateSystem world_coord(10.0); // World coordinate axes
    viz_window.showWidget("World", world_coord);

    std::shared_ptr<VOEstimator> Odometry = std::make_shared<KeyPointVOEstimator>();
    reinterpret_cast<KeyPointVOEstimator*>(Odometry.get())->setFeatureType(KeyPointVOEstimator::FeatureType::SIFT);

    auto KeyFrameSelector = KeyFrameSelection({Mode::FRAME_INTERVAL}, {1});
    
    // RH: Declare and init measurement vector: len(6) <- (trans, rpy_euler)
    cv::Mat measurements(6, 1, CV_64FC1); measurements.setTo(cv::Scalar(0));
    // bool good_measurement = false;

    while (true) {
        cv::Mat img, gray;
        video >> img;  // Grab an image from the video
        if (img.empty()) break;
        if (img.channels() > 1) cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);
        else gray = img.clone();


        auto status = Odometry->estimateMotion(gray_prev, gray, K);
        gray_prev = gray.clone();

        if(KeyFrameSelector.update(status)) {

        }

        cv::Vec3b info_color = status.successful ? cv::Vec3b(0, 255, 0) : cv::Vec3b(0, 0, 255);

        // Show the image and & camera pose 
        if (img.channels() < 3) 
            cv::cvtColor(img, img, cv::COLOR_GRAY2RGB);

        
        for (int i = 0; i < status.pts_prev.size(); i++) {
            if (status.inlier_mask.at<uchar>(i) > 0)
                cv::line(img, status.pts_prev[i], status.pts_curr[i], cv::Vec3b(0, 0, 255));
            else 
                cv::line(img, status.pts_prev[i], status.pts_curr[i], cv::Vec3b(0, 127, 0));
        }

        cv::Mat T_inv = status.T.inv();
        camera_pose = camera_pose * T_inv;
        
        cv::Affine3d camera_pose_affine = cv::Affine3d(camera_pose);
        cv::Matx<double, 3, 3> R = camera_pose_affine.rotation();
        cv::Vec3d t = camera_pose_affine.translation();
        
        // RH: Create copies (should really declare these above the loop for speed, but unsure it matters)
        cv::Mat R_est = cv::Mat(R);
        cv::Mat t_est = cv::Mat(t);

        double x = t[0], y = t[1], z = t[2];
        cv::String info = cv::format("Inliers: %d (%d%%),  XYZ: [%.3f, %.3f, %.3f]", status.inlier_num, status.inlier_ratio*100, x, y, z);

        cv::imshow("Camera Pose", img);


        if (!initialized) {
            // kalman_filter.initialize(cam_pos);

            // RH: 9 pos + 9 orient (dynamics), 3 xyz + 3 rpy (measurement), 0 u (controls), fps=1/30
            initKalmanFilter(kalman_filter, 18, 6, 0, (1/30));
            initialized = true;
        }
        // cv::Point3d predicted_pos = kalman_filter.predict();
        // cv::Point3d filtered_pos = kalman_filter.correct(cam_pos);

        // RH: Should only update measurement vector if the latest estimate is good!!!
        // if (bgoodMeasurement) {...}
        fillMeasurements( measurements, t_est, R_est );

        // RH: Should I have this return the prediction too?
        // cv::Mat t_est(3, 1, CV_64FC1);
        // cv::Mat R_est(3, 3, CV_64FC1);
        updateKalmanFilter( kalman_filter, measurements, t_est, R_est);

        // RH: USE t_est as corrected position!!! AND USE R_est as corrected orientation!!!
        cv::Point3d t_corr = cv::Point3d(t_est);

        trajectory.push_back(t_corr);
        // predicted_points_3d.push_back(t_pred);
        original_traj.push_back(t); // RH: Are filtered_pos and cam_pos not the same?

        // Use estimated and corrected R_est and t_est pose
        for (int i = 0; i < status.pts_3d.size(); i++) {
            if (status.inlier_mask.at<uchar>(i) > 0) {
                cv::Point3d p = status.pts_3d[i];
                cv::Mat p_vec = (cv::Mat_<double>(3,1) << p.x, p.y, p.z);
                cv::Mat p_rotated = (R_est * p_vec);
                cv::Mat p_trans = p_rotated + t_est;
                points_3d.push_back(cv::Point3d(
                    p_trans.at<double>(0),
                    p_trans.at<double>(1),
                    p_trans.at<double>(2)
                ));
            }
        }
        


        auto vis = visualizeTrajectory(trajectory, 1, 3.0, 3.0, cv::viz::Color::green());
        // auto vis_pred = visualizeTrajectory(predicted_points_3d, 1, 3.0, 3.0, cv::viz::Color::red());
        auto vis_corrected = visualizeTrajectory(original_traj, 1, 3.0, 3.0, cv::viz::Color::blue());
        auto vis_3d = visualizeTrajectory(points_3d, 0.3, -1, 1.0, cv::viz::Color::yellow());

        // std::vector<std::string> names = {"Trajectory", "Predicted Trajectory", "Corrected Trajectory", "3D Points"};
        std::vector<std::string> names = {"Trajectory", "Corrected Trajectory", "3D Points"};
        // std::vector<std::optional<TrajectoryVisualization>> vis_list = {vis, vis_pred, vis_corrected, vis_3d};
        std::vector<std::optional<TrajectoryVisualization>> vis_list = {vis, vis_corrected, vis_3d};

        for (size_t i = 0; i < vis_list.size(); ++i) {
            if (vis_list[i].has_value()) {
                drawTrajectory(viz_window, vis_list[i].value(), names[i]);
            }
        }

        viz_window.setViewerPose(cv::viz::makeCameraPose(
            cv::Vec3d(0, 150, 0) + t,
            t,
            cv::Vec3d(0, 0, -1)
        ));

        viz_window.spinOnce(1, true);
        int key = cv::waitKey(1);
        if (key == 32) key = cv::waitKey(); // Space
        if (key == 27) break; // ESC
    }

    cv::waitKey();
    fclose(camera_traj);
    video.release();
    return 0;
}
