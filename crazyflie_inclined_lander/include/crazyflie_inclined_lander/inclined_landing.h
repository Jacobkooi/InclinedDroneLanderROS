//
// Created by Jacob, Code Skeleton by Hai Zhu
//

#ifndef CRAZYFLIE_INCLINED_LANDER_INCLINED_LANDING_H
#define CRAZYFLIE_INCLINED_LANDER_INCLINED_LANDING_H


#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <ros/ros.h>
#include <tf/tf.h>
#include <torch/script.h>
#include <iostream>
#include <memory>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Float64.h>
#include <nav_msgs/Odometry.h>
#include <bebop2_msgs/FullStateWithCovarianceStamped.h>

// Define a class, including a constructor, member variables and member functions
class Inclined_Landing {
public:
    //! Constructor, "main" will need to instantiate a ROS nodehandle, then pass it to the constructor
    explicit Inclined_Landing(ros::NodeHandle nh);

    torch::jit::script::Module module_landing;
    torch::jit::script::Module module_setpoint;

    struct Control{
        double pwm, theta, phi;
    };

    struct Quaternion{
        double w, x, y, z;
    };

    ros::Publisher      pub_;
    Eigen::Vector3d     pos_;          // measured position information
    Eigen::Vector3d     vel_;

    Eigen::Vector3d     landinggoal_pos_;          // measured position information
    Eigen::Vector3d     landinggoal_vel_;

    Eigen::Vector3d     orient_optitrack;
    Eigen::Vector3d     angle;

    Control             control;
    Control             control_clip;
    double              hover_PWM;
    double              landing_pwm;
    double              landed_pwm;
    double              landing_angle;
    double              x_offset;
    double              y_offset;
    double              z_offset;
    double              landing_threshold;
    double              pi;
    bool                emergency_landing;
    bool                inclined_landing;
    bool                landed;
    bool                onboard_angles;
    Quaternion          quat;

    // Create Ivalue object which is needed as the input for loaded gpu scriptmodule
    std::vector<torch::jit::IValue> inputs_landing;
    std::vector<torch::jit::IValue> inputs_setpoint;

    float clip(float n, float lower, float upper) {
        return std::max(lower, std::min(n, upper));
    }

    void Check_Succesfull_Landing() {
        if (!onboard_angles) {
            if (abs(pos_(0) - landinggoal_pos_(0)) < landing_threshold &&
                abs(pos_(2) - landinggoal_pos_(2)) < landing_threshold &&
                abs(vel_(0) - landinggoal_pos_(0)) < 10*landing_threshold &&
                abs(vel_(2) - landinggoal_pos_(2)) < 10*landing_threshold &&
                abs(orient_optitrack(1) - landing_angle) < 0.05) {
//            landed = true;
                emergency_landing = true;
            }
        }
        else {
            if (abs(pos_(0) - landinggoal_pos_(0)) < 0.15 &&
                abs(pos_(2) - landinggoal_pos_(2)) < 0.15 &&
                abs(vel_(0) - landinggoal_pos_(0)) < 1.5 &&
                abs(vel_(2) - landinggoal_pos_(2)) < 1.5 &&
                abs(angle(1) - landing_angle) < 0.05) {
//            landed = true;
                emergency_landing = true;
            }
        }
    }

    void ToControlInput_pwmfromhover_2d(std::vector<float> network_output, double pwm_from_hover) {
        if (emergency_landing && !landed){
            control.pwm = landing_pwm;
            control.theta = 0;
            control.phi = 0;
        } else if (landed){
            control.pwm = landed_pwm;
            control.theta = -25.7;
            control.phi = 0;
        } else {
            control_clip.pwm = clip(network_output[0], -1, 1);
            control_clip.theta = clip(network_output[1], -1, 1);

            control.pwm = hover_PWM + control_clip.pwm*pwm_from_hover;
            control.phi = 0;
            control.theta = control_clip.theta * 30;
        }
    }

    void ToControlInput_pwmfromhover_3d(std::vector<float> network_output, double pwm_from_hover) {
        if (emergency_landing){
            control.pwm = landing_pwm;
            control.theta = 0;
            control.phi = 0;
        } else {
            control_clip.pwm = clip(network_output[0], -1, 1);
            control_clip.phi = clip(network_output[1], -1, 1);
            control_clip.theta = clip(network_output[2], -1, 1);

            control.pwm = hover_PWM + control_clip.pwm * pwm_from_hover;
            control.phi = control_clip.phi * 30;
            control.theta = control_clip.theta * 30;
        }
    }

    std::vector<float> Create_Network_Output_landing(torch::Device cuda) {
        // Very important to lead the input torch tensor to the CUDA device
        if (!onboard_angles) {
            inputs_landing.emplace_back(torch::tensor({{pos_(0), pos_(2), vel_(0),
                                                               vel_(2), orient_optitrack(1)}}).to(cuda));
        } else {
            inputs_landing.emplace_back(torch::tensor({{pos_(0), pos_(2), vel_(0),
                                                               vel_(2), angle(1)}}).to(cuda));
        }
        // Use the forward function to get the policy output in an Ivalue object
        auto output_landing = module_landing.forward(inputs_landing);
        // Must convert Ivalue to a Tuple and then to a Tensor before processing to Float
        // Adopted from https://g-airborne.com/bringing-your-deep-learning-model-to-production-with-libtorch-part-2-tracing-your-pytorch-model/
        auto output_tuple_landing = output_landing.toTuple()->elements()[0].toTensor();
        auto output_size_landing = output_tuple_landing.sizes()[1];
        auto output_vector_landing = std::vector<float>(output_size_landing);
        for (int i = 0; i < output_size_landing; i++) {
            output_vector_landing[i] = output_tuple_landing[0][i].item<float>();
        }
        inputs_landing.clear();
        return output_vector_landing;
    }

    std::vector<float> Create_Network_Output_setpoint(torch::Device cuda) {
        // Very important to lead the input torch tensor to the CUDA device
        if (!onboard_angles) {
            inputs_setpoint.emplace_back(torch::tensor({{pos_(0), pos_(1),
                                                                pos_(2), vel_(0), vel_(1),
                                                                vel_(2), orient_optitrack(0),
                                                                orient_optitrack(1)}}).to(cuda));
        } else {
            inputs_setpoint.emplace_back(torch::tensor({{pos_(0), pos_(1),
                                                                pos_(2), vel_(0), vel_(1),
                                                                vel_(2), angle(0),
                                                                angle(1)}}).to(cuda));
        }

        // Use the forward function to get the policy output in an Ivalue object
        auto output_setpoint = module_setpoint.forward(inputs_setpoint);
        // Must convert Ivalue to a Tuple and then to a Tensor before processing to Float
        // Adopted from https://g-airborne.com/bringing-your-deep-learning-model-to-production-with-libtorch-part-2-tracing-your-pytorch-model/
        auto output_tuple_setpoint = output_setpoint.toTuple()->elements()[0].toTensor();
        auto output_size_setpoint = output_tuple_setpoint.sizes()[1];
        auto output_vector_setpoint = std::vector<float>(output_size_setpoint);
        for (int i = 0; i < output_size_setpoint; i++) {
            output_vector_setpoint[i] = output_tuple_setpoint[0][i].item<float>();
        }
        inputs_setpoint.clear();
        return output_vector_setpoint;

    }

private:

    void ToEulerAngles(Quaternion q) {

        // roll (x-axis rotation)
        double sinr_cosp = 2 * (q.w * q.x + q.y * q.z);
        double cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y);
        angle(0) = std::atan2(sinr_cosp, cosr_cosp);

        // pitch (y-axis rotation)
        double sinp = 2 * (q.w * q.y - q.z * q.x);
        if (std::abs(sinp) >= 1)
            angle(1) = std::copysign(M_PI / 2, sinp); // use 90 degrees if out of range
        else
            angle(1) = std::asin(sinp);

        // yaw (z-axis rotation)
        double siny_cosp = 2 * (q.w * q.z + q.x * q.y);
        double cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
        angle(2) = std::atan2(siny_cosp, cosy_cosp);
    }

    //! Ros node handle
    ros::NodeHandle     nh_;        // we will need this, to pass between "main" and constructor

    //! Some objects to support subscriber, service, and publisher
    ros::Subscriber     sub_position;
    ros::Subscriber     sub_orientation;


    //! Time information for filter
    ros::Time           time_stamp_;            // time stamp of current measurement
    ros::Time           time_stamp_previous_;   // time stamp of last measurement
    double              dt_;                    // time difference between two measurements


    //! Initializations
    void initializeSubscribers();
    void initializePublishers();

    //! Subscriber callback
    void subscriberCallback_pos(const bebop2_msgs::FullStateWithCovarianceStamped &msg);
    void subscriberCallback_cf(const geometry_msgs::PoseStamped &msg);

};

#endif //CRAZYFLIE_INCLINED_LANDER_INCLINED_LANDING

