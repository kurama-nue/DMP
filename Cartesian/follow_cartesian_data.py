import sys
import rospy 
import rospkg
import pandas as pd
import os
import moveit_commander
from geometry_msgs.msg import PoseStamped
from math import pi


class CartesianPlayer:
    def __init__(self) -> None:
        moveit_commander.roscpp_initialize("my_gen3")
        rospy.init_node('Cartesian_Cordinate_Follower')
        arm_group_name = "arm"
        self.robot = moveit_commander.RobotCommander("robot_description")
        self.scene = moveit_commander.PlanningSceneInterface(ns=rospy.get_namespace())
        self.arm_group = moveit_commander.MoveGroupCommander(arm_group_name, ns=rospy.get_namespace())


        self.rate = rospy.Rate(5)
        self.dataFileLocation = os.path.join("~/catkin_workspace", "cartesian_data.csv")
        try:
            self.dataFrame = pd.read_csv(self.dataFileLocation)
            print("Loaded Data sucesfully")
        except:
            print("No data file found please retry with valid location")
            exit()
        
        self.index = 0
    
    def go_to_next_cartesian_position(self):
        arm_group = self.arm_group
        tolerance = 0.01
        constraints = None
        arm_group.set_goal_position_tolerance(tolerance)
        arm_group.set_path_constraints(constraints)
        pose = self.get_cartesian_pose()
        pose.pose.position.x = self.dataFrame.loc[self.index, 'p_x']
        pose.pose.position.y = self.dataFrame.loc[self.index, 'p_y']
        pose.pose.position.z = self.dataFrame.loc[self.index, 'p_z']
        arm_group.set_pose_target(pose)
        print(f"going to cartesian next pose at index {self.index}")
        print(pose)
        arm_group.go(wait = True)
        self.index += 5
        if(self.index >= self.dataFrame.size):
            print("executed all data exiting")
            exit()
        try:
          self.is_gripper_present = rospy.get_param(rospy.get_namespace() + "is_gripper_present", False)
          if self.is_gripper_present:
            gripper_joint_names = rospy.get_param(rospy.get_namespace() + "gripper_joint_names", [])
            self.gripper_joint_name = gripper_joint_names[0]
          else:
            self.gripper_joint_name = ""
          self.degrees_of_freedom = rospy.get_param(rospy.get_namespace() + "degrees_of_freedom", 7)
        except:
            pass
        if self.is_gripper_present:
            gripper_group_name = "gripper"
            self.gripper_group = moveit_commander.MoveGroupCommander(gripper_group_name, ns=rospy.get_namespace())

        

    def get_cartesian_pose(self):
        arm_group = self.arm_group
        currentPose = arm_group.get_current_pose()
        return currentPose
        
    
if __name__ == "__main__":
    arm = CartesianPlayer()
    try:
        while(not rospy.is_shutdown()):
            arm.go_to_next_cartesian_position()
    except KeyboardInterrupt:
        pass

