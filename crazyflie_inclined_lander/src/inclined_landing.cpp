//
// Created by Jacob, Code Skeleton by Hai Zhu
//

#include <crazyflie_inclined_lander/inclined_landing.h>

// Constructor:  this will get called whenever an instance of this class is created
Inclined_Landing::Inclined_Landing(ros::NodeHandle nh): nh_(nh)
{
    ROS_INFO("In class constructor of inclined_landing");

    // Initialization subscriber and publisher
    initializeSubscribers();
    initializePublishers();
    // Other initialization
    pos_.setZero();
    vel_.setZero();
    landinggoal_pos_.setZero();
    landinggoal_vel_.setZero();
    orient_optitrack.setZero();
    angle.setZero();
    hover_PWM = 43504;  // ForcetoPWM_forster(9.81*0.03215) Check weight of crazyflie!!
    landing_pwm = 37500;
    landed_pwm = 12000;
    control.theta = 0;
    control.phi = 0;
    control.pwm = 0;
    control_clip.theta = 0;
    control_clip.pwm = 0;
    emergency_landing = false;
    inclined_landing = false;
    x_offset = 0;
    y_offset = 0;
    z_offset = 0;
    pi = 3.1416;

    time_stamp_ = ros::Time::now();
    time_stamp_previous_ = ros::Time::now();
    dt_ = 0.01;
}

// Set up subscribers
void Inclined_Landing::initializeSubscribers()
{
    ROS_INFO("Initializing subscribers");
    sub_position = nh_.subscribe("/Crazyflie101/full_state_estimation", 1, &Inclined_Landing::subscriberCallback_pos, this);
    sub_orientation = nh_.subscribe("/cf1/pose", 1, &Inclined_Landing::subscriberCallback_cf, this);
}

// Set up publisher
void Inclined_Landing::initializePublishers()
{
    ROS_INFO("Initializing publishers");
    pub_ = nh_.advertise<geometry_msgs::Twist>("/cf1/cmd_vel", 1, true);
}

// Subscriber callback function for the Optitrack states
void Inclined_Landing::subscriberCallback_pos(const bebop2_msgs::FullStateWithCovarianceStamped &msg)
{
    // get measured position
    pos_(0) = msg.state.x +object_offset + x_offset;
    pos_(1) = msg.state.y + y_offset;
    pos_(2) = msg.state.z + z_offset;

    // get measured velocity
    vel_(0)= msg.state.x_dot;
    vel_(1) = msg.state.y_dot;
    vel_(2) = msg.state.z_dot;
    // get measured angles
    orient_optitrack(0) = msg.state.roll;
    orient_optitrack(1) = msg.state.pitch;
    orient_optitrack(2) = msg.state.yaw;

    // current time stamp of the message
    time_stamp_      = msg.header.stamp;
    // time difference. If using the node_rate to derive, then comment the following lines
    dt_ = (time_stamp_ - time_stamp_previous_).toSec();
    // set time
    time_stamp_previous_ = time_stamp_;

    if (inclined_landing) {
        Check_Succesfull_Landing();
    }
}

// Callback if using the onboard orientation estimates
void Inclined_Landing::subscriberCallback_cf(const geometry_msgs::PoseStamped &msg)
{
    // get measured position
    quat.x = msg.pose.orientation.x;
    quat.y = msg.pose.orientation.y;
    quat.z = msg.pose.orientation.z;
    quat.w = msg.pose.orientation.w;
    // convert the quaternions to euler angles, angle(0) = roll, angle(1) = pitch, angle(2) = yaw
    ToEulerAngles(quat);
    angle(0) *= -1; // We train with roll angle to the left, but onboard angles have roll angle to the right.
    // current time stamp of the message
    time_stamp_      = msg.header.stamp;
    // time difference. If using the node_rate to derive, then comment the following lines
    dt_ = (time_stamp_ - time_stamp_previous_).toSec();
    // set time
    time_stamp_previous_ = time_stamp_;
}
