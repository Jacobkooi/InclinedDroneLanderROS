//
// Created by Jacob, Code Skeleton by Hai Zhu.
//

#include <crazyflie_inclined_lander/inclined_landing.h>

// DON'T FORGET TO TRIM IF THE QUADCOPTER DRIFTS. CHECK THE GITHUB OF CRAZYFLIE_ROS.
int main(int argc, char **argv)
{
    // Set up ROS
    ros::init(argc, argv, "inclined_landing_node");     // node name
    ros::NodeHandle nh;                                         // create a node handle
    // Initialize a class object and pass node handle for constructor
    Inclined_Landing networknode(nh);
    // Define CUDA device
    torch::DeviceType device_type;
    device_type = torch::kCUDA;
    torch::Device device(device_type);
    // Select the rate at which we want to control the drone
    ros::Rate loop_rate(80);
    // start time
    ros::Time startTime = ros::Time::now();
    // differential time
    ros::Duration diffTime = ros::Time::now() - startTime;
    // Define control, start and landing times
    double dt = diffTime.toSec();
    double start_time = 3;
    double test_time = 14;
    double landing_time = 0.5;
    double total_time = test_time + landing_time + start_time;
    networknode.onboard_angles = true;
    networknode.landing_threshold = 0.10;

    // Load the torchscript modules. Always load a GPU model and use Cuda 11.0 Libtorch / Cudnn
    // Load the 2D policy network for inclined landing
    networknode.module_landing = torch::jit::load("/home/jacob/PycharmProjects/Crazyflie_Torch/Torchscript_models/PPO_17000_3miltimesteps_0015noise_6obs_FIRST_RESULTS.pt");
//    networknode.module_setpoint = torch::jit::load("/home/jacob/PycharmProjects/Deep_Inclined_Drone_Lander/Torchscript_models/97gammaweakmotor.pt");
//    networknode.module_landing = torch::jit::load("/home/jacob/PycharmProjects/Deep_Inclined_Drone_Lander/Torchscript_models/2dinclined_weakmotor_300000timesteps.pt");

    // Load the 3D policy network for set-point tracking
     networknode.module_setpoint = torch::jit::load("/home/jacob/PycharmProjects/Crazyflie_Torch/Torchscript_models/PPO_Euclidean_1000000_timesteps.pt");

    std::vector<float> test1 = networknode.Create_Network_Output_landing(device);
    std::vector<float> test2 = networknode.Create_Network_Output_setpoint(device);

    ///// QUAD WEIGHS 32.57 with rubber legs and glue!!! //////
    // Main loop that is ran with the loop rate
    while (ros::ok() && dt <= total_time)
    {
//         Safety for when the quad is near the edges, this triggers a landing
        if (networknode.pos_.x() >= 3.6 || networknode.pos_.x() <= -3.6 || networknode.pos_.y() >= 2.4 || networknode.pos_.z() >= 2.2){
            networknode.emergency_landing = true;
        }
       //  Approaching from the right
        if (dt >= 7){
            networknode.x_offset = 0;
            networknode.z_offset = -0.8;
        }
        if (dt >= 11){
            networknode.x_offset = 0;
            networknode.z_offset = 0;
            networknode.inclined_landing = true;
        }


        std::vector<float> output_vector_landing;
        std::vector<float> output_vector_setpoint;

        if (!networknode.inclined_landing) {
            // For a still unknown reason, we must keep the landing network active or it gets a slight pause during
            // the switching of the networks which crashes the Crazyflie, maybe something with GPU allocation
            output_vector_landing = networknode.Create_Network_Output_landing(device);
            // Calculate the setpoint output vector
            output_vector_setpoint = networknode.Create_Network_Output_setpoint(device);
        } else{
            // When we want inclined landing, this calculates the landing output vector
            output_vector_landing = networknode.Create_Network_Output_landing(device);
        }

        if (networknode.inclined_landing) {
            networknode.ToControlInput_pwmfromhover_2d(output_vector_landing, 16500);
            // Check if the quadrotor has landed within the bounds
//            networknode.Check_Succesfull_Landing();

        } else{
           networknode.ToControlInput_pwmfromhover_3d(output_vector_setpoint, 14000);
        }

        // And finally publish the control outputs, linear.x = pitch, linear.y = roll, linear.z = PWM, angular.z = yawdot
        geometry_msgs::Twist msg;
        bebop2_msgs::FullStateWithCovarianceStamped msg_2;
        msg_2.state.x = networknode.pos_(0);
        msg_2.state.y = networknode.pos_(1);
        msg_2.state.z = networknode.pos_(2);
        msg_2.state.roll = networknode.angle(0);
        msg_2.state.pitch = networknode.angle(1);
        msg_2.state.yaw = networknode.angle(2);
        msg_2.state.pitch_dot = networknode.control.theta;
        msg_2.state.yaw_dot = networknode.control.pwm;

        msg.linear.x = networknode.control.theta;
        msg.linear.y = -networknode.control.phi;  // Negative for the onboard controller, which takes +phi to the right
        msg.linear.z = networknode.control.pwm;
        msg.angular.x = 0;
        msg.angular.y = 0;
        msg.angular.z = 0;

        // Make a descending landing after the test time is over
        if (dt >= test_time + start_time){
            msg.linear.x = 0;
            msg.linear.y = 0;
            msg.linear.z = networknode.landing_pwm;
        }
        // Need to start giving 0 thrust commands for the first 0.5-1 seconds for the onboard security
        if (dt <= start_time){
            msg.linear.x = 0;
            msg.linear.y = 0;
            msg.linear.z = 0;
        }
        // Publish the control output to /cf1/cmd_vel
        networknode.pub_.publish(msg);
        networknode.pub_handig.publish(msg_2);
        ros::spinOnce();
        loop_rate.sleep();
        diffTime = ros::Time::now() - startTime;
        dt = diffTime.toSec();
    }

    return 0;
}