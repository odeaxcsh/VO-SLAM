#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <thread>
#include <chrono>

#include <Eigen/Dense>

#include "opencv2/opencv.hpp"
#include <opencv2/core/persistence.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/core/eigen.hpp>


#include "kalman.hpp"
#include "odometry.hpp"
#include "keyframeselection.hpp"

#include <open3d/Open3D.h>

#include <optional>

Eigen::MatrixXd cvMatToEigen(const cv::Mat& mat, int x, int y) {
    Eigen::MatrixXd eigenMat(x, y);
    if (mat.cols == y && mat.rows == x && mat.type() == CV_64F) {
        for (int i = 0; i < x; ++i) {
            for (int j = 0; j < y; ++j) {
                eigenMat(i, j) = mat.at<double>(i, j);
            }
        }
    }
    return eigenMat;
}

cv::Mat eigen_to_cvmat(const Eigen::MatrixXd& eigen_matrix) {
    cv::Mat cv_mat;
    cv::eigen2cv(eigen_matrix, cv_mat);
    return cv_mat;
}

int main()
{
    // --- File paths ---
    const std::string image_folder = "../data/Synthetic_base/images/";
    const std::string image_pattern = image_folder + "*.png";
    const std::string calib_file = "../data/Synthetic_base/camera_calibration.json";
    const std::string gt_timestamp_file = "../data/Synthetic_base/timings.csv";
    const std::string estimated_traj_file = "../data/Synthetic_base/estimated_trajectory.txt";
    
    // --- Load ground truth timestamps ---
    std::vector<double> timestamps;
    std::ifstream gt_file(gt_timestamp_file);
    if (!gt_file.is_open()) {
        std::cerr << "ERROR: Could not open timestamp file: " << gt_timestamp_file << std::endl;
        return -1;
    }
    std::string line;

    // Skip the header line
    if (!std::getline(gt_file, line)) {
        std::cerr << "ERROR: Could not read header line from " << gt_timestamp_file << std::endl;
        gt_file.close();
        return -1;
    }

    while (std::getline(gt_file, line)) {
        std::stringstream ss(line);
        std::string segment;
        long long timestamp_ns = 0;

        // Get the first segment (timestamp) before the comma
        if (std::getline(ss, segment, ',')) {
            try {
                // Convert string segment to long long for nanoseconds
                timestamp_ns = std::stoll(segment);
                // Convert nanoseconds to seconds (double) and add to vector
                timestamps.push_back(static_cast<double>(timestamp_ns) / 1.0e9);
            } catch (const std::invalid_argument& ia) {
                std::cerr << "Warning: Invalid timestamp format on line: " << line << " (" << ia.what() << "). Skipping." << std::endl;
            } catch (const std::out_of_range& oor) {
                std::cerr << "Warning: Timestamp out of range on line: " << line << " (" << oor.what() << "). Skipping." << std::endl;
            }
        } else if (!line.empty()) { // Handle lines that might not have a comma but aren't empty
            std::cerr << "Warning: Malformed line (no comma found or empty timestamp): " << line << ". Skipping." << std::endl;
        }
        // We only care about the first column (timestamp), so we don't need to read the rest of the line.
    }
    gt_file.close();

    // --- Load camera intrinsics from JSON file ---
    double fx, fy, cx_d, cy_d;
    int width = 1920, height = 1080;
    try {
        cv::FileStorage fs(calib_file, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            std::cerr << "ERROR: Could not open calibration file: " << calib_file << std::endl;
            return -1;
        }
        fs["fx"] >> fx;
        fs["fy"] >> fy;
        fs["cx"] >> cx_d;
        fs["cy"] >> cy_d;
        // Optionally read width and height if present in JSON
        if (!fs["width"].empty()) fs["width"] >> width;
        if (!fs["height"].empty()) fs["height"] >> height;
        fs.release();
    } catch (const cv::Exception& e) {
        std::cerr << "Error processing calibration file: " << e.what() << std::endl;
        return -1;
    }

    // --- Load image files ---
    std::vector<std::string> image_files;
    cv::glob(image_pattern, image_files, false);
    if (image_files.empty()) {
        std::cerr << "ERROR: No image files found matching pattern: " << image_pattern << std::endl;
        return -1;
    }
    std::sort(image_files.begin(), image_files.end());

    if (timestamps.size() != image_files.size()) {
        std::cerr << "ERROR: Timestamp count (" << timestamps.size()
                << ") doesn't match image count (" << image_files.size() << ")" << std::endl;
        // Add a small debug print to see the last few timestamps read vs expected
        if (!timestamps.empty()) {
            std::cerr << "Last few timestamps read: ";
            size_t start_idx = (timestamps.size() > 5) ? timestamps.size() - 5 : 0;
            for (size_t i = start_idx; i < timestamps.size(); ++i) {
                std::cerr << timestamps[i] << " ";
            }
            std::cerr << std::endl;
        }
        if (!image_files.empty()) {
            std::cerr << "Last few image files found: ";
            size_t start_idx = (image_files.size() > 5) ? image_files.size() - 5 : 0;
            for (size_t i = start_idx; i < image_files.size(); ++i) {
                std::cerr << image_files[i] << " ";
            }
            std::cerr << std::endl;
        }

        return -1;
    }

    // --- Initialize trajectory file ---
    FILE* estimated_traj = fopen(estimated_traj_file.c_str(), "wt");
    if (!estimated_traj) {
        std::cerr << "ERROR: Could not open trajectory file: " << estimated_traj_file << std::endl;
        return -1;
    }
    std::cout << "Saving the estimate trajectory to " << estimated_traj_file << std::endl;

    // --- VO Parameters ---
    bool use_5pt = true;
    int min_inlier_num = 50;
    double min_inlier_ratio = 0.1;
    cv::Mat camera_pose_cv = cv::Mat::eye(4, 4, CV_64F);

    // --- Open3D Visualization Setup ---
    open3d::visualization::Visualizer visualizer;
    visualizer.CreateVisualizerWindow("Open3D Visual Odometry", 1024, 768);
    visualizer.GetRenderOption().background_color_ = Eigen::Vector3d(1.0, 1.0, 1.0);
    visualizer.GetRenderOption().point_size_ = 2.0;
    visualizer.GetRenderOption().line_width_ = 5.0;

    auto coord_axes = open3d::geometry::TriangleMesh::CreateCoordinateFrame(1.0);
    visualizer.AddGeometry(coord_axes);

    auto trajectory_lineset = std::make_shared<open3d::geometry::LineSet>();
    std::vector<Eigen::Vector3d> trajectory_points_eigen;
    trajectory_points_eigen.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
    trajectory_lineset->points_ = trajectory_points_eigen;
    visualizer.AddGeometry(trajectory_lineset);

    // --- Initial camera visualization ---
    Eigen::Matrix3d K;
    K << fx, 0, cx_d, 0, fy, cy_d, 0, 0, 1;
    auto camera_frustum = open3d::geometry::LineSet::CreateCameraVisualization(
        width, height, K, cvMatToEigen(camera_pose_cv, 4, 4).inverse(), 0.5);
    camera_frustum->PaintUniformColor(Eigen::Vector3d(0.0, 0.0, 1.0));
    visualizer.AddGeometry(camera_frustum);

    // --- Write initial pose ---
    Eigen::Matrix4d camera_pose = Eigen::Matrix4d::Identity();
    Eigen::Quaterniond q_initial(camera_pose.block<3,3>(0,0));
    q_initial.normalize();
    fprintf(estimated_traj, "%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
            timestamps[0],
            camera_pose(0,3), camera_pose(1,3), camera_pose(2,3),
            q_initial.x(), q_initial.y(), q_initial.z(), q_initial.w());


    bool initialized = false;
    MotionKalmanFilter basic_kalman_filter;
    cv::KalmanFilter kalman_filter;

    //std::vector<cv::Point3d> trajectory;
    std::vector<cv::Point3d> points_3d;
    std::vector<cv::Point3d> original_traj, predicted_points_3d;
    
    std::shared_ptr<VOEstimator> Odometry = std::make_shared<KeyPointVOEstimator>();
    reinterpret_cast<KeyPointVOEstimator*>(Odometry.get())->setFeatureType(KeyPointVOEstimator::FeatureType::SIFT);

    auto KeyFrameSelector = KeyFrameSelection({Mode::FRAME_INTERVAL}, {1});
    
    // RH: Declare and init measurement vector: len(6) <- (trans, rpy_euler)
    cv::Mat measurements(6, 1, CV_64FC1); measurements.setTo(cv::Scalar(0));
    // bool good_measurement = false;

    // --- Main processing loop ---
    cv::Mat gray_prev = cv::imread(image_files[0], cv::IMREAD_GRAYSCALE);
    for (size_t frame_idx = 1; frame_idx < image_files.size(); ++frame_idx) {
        cv::Mat img = cv::imread(image_files[frame_idx]);
        if (img.empty()) continue;

        cv::Mat gray;
        if (img.channels() > 1) {
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = img.clone();
        }

        auto status = Odometry->estimateMotion(gray_prev, gray, eigen_to_cvmat(K));
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

        Eigen::Matrix4d T_inv = cvMatToEigen(status.T.inv(), 4, 4);
        camera_pose = camera_pose * T_inv;
        
        cv::Affine3d camera_pose_affine = cv::Affine3d(eigen_to_cvmat(camera_pose));
        cv::Matx<double, 3, 3> R = camera_pose_affine.rotation();
        cv::Vec3d t = camera_pose_affine.translation();
        
        // cv::Mat R_est = cv::Mat(R);
        // cv::Mat t_est = cv::Mat(t);

        double x = t[0], y = t[1], z = t[2];
        cv::String info = cv::format("Inliers: %d (%d%%),  XYZ: [%.3f, %.3f, %.3f]", status.inlier_num, status.inlier_ratio*100, x, y, z);

        cv::imshow("Camera Pose", img);


        if (!initialized) {
            basic_kalman_filter.initialize(t);

            // RH: 9 pos + 9 orient (dynamics), 3 xyz + 3 rpy (measurement), 0 u (controls), fps=1/30
            // initKalmanFilter(kalman_filter, 18, 6, 0, (1/30));
            initialized = true;
        }
        cv::Point3d predicted_pos = basic_kalman_filter.predict();
        Eigen::Vector3d filtered_pos = cvMatToEigen((cv::Mat)basic_kalman_filter.correct(t), 3, 1);

        // fillMeasurements( measurements, t_est, R_est );
        // updateKalmanFilter( kalman_filter, measurements, t_est, R_est);
        // Eigen::Vector3d t_corr = cvMatToEigen(t_est, 3, 1);

        // trajectory_points_eigen.push_back(t_corr);

        // // --- Write estimated pose ---
        // Eigen::Quaterniond q((Eigen::Matrix3d) cvMatToEigen(R_est, 3, 3));
        // q.normalize();
        // fprintf(estimated_traj, "%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
        //         timestamps[frame_idx],
        //         t_corr(0), t_corr(1), t_corr(2),
        //         q.x(), q.y(), q.z(), q.w());

        trajectory_points_eigen.push_back(filtered_pos);
        Eigen::Matrix3d Re = cvMatToEigen(cv::Mat(R), 3, 3);
        Eigen::Quaterniond q(Re);
        q.normalize();
        fprintf(estimated_traj, "%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                timestamps[frame_idx],
                filtered_pos(0), filtered_pos(1), filtered_pos(2),
                q.x(), q.y(), q.z(), q.w());

        // --- Update 3D visualization ---
        trajectory_points_eigen.emplace_back(x, y, z);
        if (trajectory_points_eigen.size() > 1) {
            trajectory_lineset->points_ = trajectory_points_eigen;
            trajectory_lineset->lines_.clear();
            for (size_t i = 0; i < trajectory_points_eigen.size()-1; ++i) {
                trajectory_lineset->lines_.emplace_back(i, i+1);
            }
            trajectory_lineset->colors_.assign(trajectory_lineset->lines_.size(), {1,0,0});
            visualizer.UpdateGeometry(trajectory_lineset);
        }

        Eigen::Matrix4d current_pose = Eigen::Matrix4d::Identity();
        // current_pose.block<3,3>(0,0) = cvMatToEigen(R_est, 3, 3);
        // current_pose.block<3,1>(0,3) = cvMatToEigen(t_est, 3, 1);
        current_pose.block<3,3>(0,0) = Re;
        current_pose.block<3,1>(0,3) = filtered_pos;

        auto new_frustum = open3d::geometry::LineSet::CreateCameraVisualization(
            width, height, K, current_pose.inverse(), 0.5);
        new_frustum->PaintUniformColor(Eigen::Vector3d(0,0,1));
        visualizer.RemoveGeometry(camera_frustum);
        camera_frustum = new_frustum;
        visualizer.AddGeometry(camera_frustum);

        visualizer.ResetViewPoint(true);
        visualizer.PollEvents();
        visualizer.UpdateRender();

        // --- Handle input ---
        int key = cv::waitKey(1);
        if (key == 27) break;  // ESC to exit
        gray_prev = gray.clone();

    }

    // --- Cleanup ---
    fclose(estimated_traj);
    cv::destroyAllWindows();
    while (visualizer.PollEvents()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    visualizer.DestroyVisualizerWindow();
    return 0;
}
