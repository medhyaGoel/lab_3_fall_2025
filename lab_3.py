import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np
np.set_printoptions(precision=3, suppress=True)

Kp = 3
Kd = 0.1

class InverseKinematics(Node):

    def __init__(self):
        super().__init__('inverse_kinematics')
        self.joint_subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.listener_callback,
            10)
        self.joint_subscription  # prevent unused variable warning

        self.command_publisher = self.create_publisher(
            Float64MultiArray,
            '/forward_command_controller/commands',
            10
        )

        self.pd_timer_period = 1.0 / 200  # 200 Hz
        self.ik_timer_period = 1.0 / 20   # 10 Hz
        self.pd_timer = self.create_timer(self.pd_timer_period, self.pd_timer_callback)
        self.ik_timer = self.create_timer(self.ik_timer_period, self.ik_timer_callback)

        self.joint_positions = None
        self.joint_velocities = None
        self.target_joint_positions = None

        self.ee_triangle_positions = np.array([
            [0.05, 0.0, -0.12],  # Touchdown
            [-0.05, 0.0, -0.12], # Liftoff
            [0.0, 0.0, -0.06]    # Mid-swing
        ])

        center_to_rf_hip = np.array([0.07500, -0.08350, 0])
        self.ee_triangle_positions = self.ee_triangle_positions + center_to_rf_hip
        self.current_target = 0
        self.t = 0

    def listener_callback(self, msg):
        joints_of_interest = ['leg_front_r_1', 'leg_front_r_2', 'leg_front_r_3']
        self.joint_positions = np.array([msg.position[msg.name.index(joint)] for joint in joints_of_interest])
        self.joint_velocities = np.array([msg.velocity[msg.name.index(joint)] for joint in joints_of_interest])

    def forward_kinematics(self, theta1, theta2, theta3):
        ################################################################################################
        # TODO: Compute the forward kinematics for the front right leg (should be easy after lab 2!)
        ################################################################################################
        def rotation_x(angle):

            return np.array(

                [

                    [1, 0, 0, 0],

                    [0, np.cos(angle), -np.sin(angle), 0],

                    [0, np.sin(angle), np.cos(angle), 0],

                    [0, 0, 0, 1],

                ]

            )



        def rotation_y(angle):

            return np.array(

                [

                    [np.cos(angle), 0, np.sin(angle), 0],

                    [0, 1, 0, 0],

                    [-np.sin(angle), 0, np.cos(angle), 0],

                    [0, 0, 0, 1],

                ]

            )



        def rotation_z(angle):

            return np.array(

                [

                    [np.cos(angle), -np.sin(angle), 0, 0],

                    [np.sin(angle), np.cos(angle), 0, 0],

                    [0, 0, 1, 0],

                    [0, 0, 0, 1],

                ]

            )



        def translation(x, y, z):

            return np.array(

                [

                    [1, 0, 0, x],

                    [0, 1, 0, y],

                    [0, 0, 1, z],

                    [0, 0, 0, 1],

                ]

            )



        # T_0_1 (base_link to leg_front_l_1)

        T_0_1 = translation(0.07500, -0.0445, 0) @ rotation_x(1.57080) @ rotation_z(-theta1)



        # T_1_2 (leg_front_l_1 to leg_front_l_2)

        ## Implement the transformation matrix from leg_front_l_1 to leg_front_l_2

        T_1_2 =  translation(0, 0, 0.039) @ rotation_y(-1.5708) @ rotation_z(theta2)



        # T_2_3 (leg_front_l_2 to leg_front_l_3)

        ## Implement the transformation matrix from leg_front_l_2 to leg_front_l_3

        T_2_3 = translation(0, -0.0494, 0.0685) @ rotation_y(1.5708) @ rotation_z(-theta3)



        # T_3_ee (leg_front_l_3 to end-effector)

        T_3_ee = translation(0.06231, -0.06216, -0.018)



        # Compute the final transformation. T_0_ee is the multiplication of the previous transformation matrices

        T_0_ee = T_0_1 @ T_1_2 @ T_2_3 @ T_3_ee



        # Extract the end-effector position. The end effector position is a 3x1 vector (not in homogenous coordinates)

        end_effector_position = T_0_ee[0:3, 3]



        return end_effector_position

    def inverse_kinematics(self, target_ee, initial_guess=[0, 0, 0]):
        def cost_function(theta):
            # Compute the cost function and the squared L2 norm of the error
            # return the cost and the squared L2 norm of the error
            ################################################################################################
            current_ee = self.forward_kinematics(*theta)
            l1 = np.abs(current_ee - target_ee)
            cost = np.sum(l1**2)
            return cost, l1
            # HINT: You can use the * notation on a list to "unpack" a list
            ################################################################################################

        def gradient(theta, epsilon=1e-3):
            # Compute the gradient of the cost function using finite differences
            ################################################################################################
            grad = np.zeros_like(theta)
            for i in range(len(theta)):
                theta_plus = np.array(theta)
                theta_minus = np.array(theta)
                theta_plus[i] += epsilon
                theta_minus[i] -= epsilon
                cost_plus, _ = cost_function(theta_plus)
                cost_minus, _ = cost_function(theta_minus)
                grad[i] = (cost_plus - cost_minus) / (2 * epsilon)
            return grad
            ################################################################################################

        theta = np.array(initial_guess)
        learning_rate = 5 #  Set the learning rate
        max_iterations = None # TODO: Set the maximum number of iterations
        tolerance = None # TODO: Set the tolerance for the L1 norm of the error

        cost_l = []
        for _ in range(max_iterations):
            grad = gradient(theta)

            # Update the theta (parameters) using the gradient and the learning rate
            ################################################################################################
            # TODO: Implement the gradient update. Use the cost function you implemented, and use tolerance t
            # to determine if IK has converged
            theta -= learning_rate * grad
            cost, l1 = cost_function(theta)
            cost_l.append(cost)
            if l1 < tolerance:
                break
            # TODO (BONUS): Implement the (quasi-)Newton's method instead of finite differences for faster convergence
            ################################################################################################

        # print(f'Cost: {cost_l}') # Use to debug to see if you cost function converges within max_iterations

        return theta

    def interpolate_triangle(self, t):

        # Interpolate between the three triangle positions in the self.ee_triangle_positions

        # based on the current time t

        ################################################################################################

        # TODO: Implement the interpolation function

        ################################################################################################

        # Normalize t to loop every 3 seconds

        t_mod = t % 3.0

       

        # Determine which edge we're on and interpolate

        if t_mod < 1.0:

            # Between vertex 0 and vertex 1 (0 <= t < 1)

            alpha = t_mod

            return (1 - alpha) * self.ee_triangle_positions[0] + alpha * self.ee_triangle_positions[1]

        elif t_mod < 2.0:

            # Between vertex 1 and vertex 2 (1 <= t < 2)

            alpha = t_mod - 1.0

            return (1 - alpha) * self.ee_triangle_positions[1] + alpha * self.ee_triangle_positions[2]

        else:

            # Between vertex 2 and vertex 0 (2 <= t < 3)

            alpha = t_mod - 2.0

            return (1 - alpha) * self.ee_triangle_positions[2] + alpha * self.ee_triangle_positions[0]

    def ik_timer_callback(self):
        if self.joint_positions is not None:
            target_ee = self.interpolate_triangle(self.t)
            self.target_joint_positions = self.inverse_kinematics(target_ee, self.joint_positions)
            current_ee = self.forward_kinematics(*self.joint_positions)

            # update the current time for the triangle interpolation
            ################################################################################################
            # TODO: Implement the time update
            ################################################################################################
            
            self.get_logger().info(f'Target EE: {target_ee}, Current EE: {current_ee}, Target Angles: {self.target_joint_positions}, Target Angles to EE: {self.forward_kinematics(*self.target_joint_positions)}, Current Angles: {self.joint_positions}')

    def pd_timer_callback(self):
        if self.target_joint_positions is not None:

            command_msg = Float64MultiArray()
            command_msg.data = self.target_joint_positions.tolist()
            self.command_publisher.publish(command_msg)

def main():
    rclpy.init()
    inverse_kinematics = InverseKinematics()
    
    try:
        rclpy.spin(inverse_kinematics)
    except KeyboardInterrupt:
        print("Program terminated by user")
    finally:
        # Send zero torques
        zero_torques = Float64MultiArray()
        zero_torques.data = [0.0, 0.0, 0.0]
        inverse_kinematics.command_publisher.publish(zero_torques)
        
        inverse_kinematics.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
