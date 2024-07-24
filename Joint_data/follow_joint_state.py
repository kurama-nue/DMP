#!/usr/bin/env python3
import sys
import rospy 
import rospkg
import pandas as pd
import os
import moveit_commander
from geometry_msgs.msg import PoseStamped
from math import pi
        
        
        
        
        
class JointStatePlayer1:
    def __init__(self) -> None:
        moveit_commander.roscpp_initialize("my_gen3")
        rospy.init_node('Joint_State_Follower')
        arm_group_name = "arm"
        self.robot = moveit_commander.RobotCommander("robot_description")
        self.scene = moveit_commander.PlanningSceneInterface(ns=rospy.get_namespace())
        self.arm_group = moveit_commander.MoveGroupCommander(arm_group_name, ns=rospy.get_namespace())

        self.rate = rospy.Rate(1)
        self.dataFileLocation = os.path.join("/home/rahul/LabData/DMP-main", "Joint_State_data33.csv")
        try:
            self.dataFrame = pd.read_csv(self.dataFileLocation)
            print("Loaded Data sucesfully")
        except:
            print("No data file found please retry with valid location")
            exit()
        self.index = 0
    
    def go_to_next_joint_state(self):
        if(self.index < len(self.dataFrame)):
            arm_group = self.arm_group
            tolerance = 0.01
            constraints = None
            arm_group.set_goal_position_tolerance(tolerance)
            arm_group.set_path_constraints(constraints)
            pose = arm_group.get_current_joint_values()
            pose[0] = self.dataFrame.loc[self.index, 'joint_angle_0']
            pose[1] = self.dataFrame.loc[self.index, 'joint_angle_1']
            pose[2] = self.dataFrame.loc[self.index, 'joint_angle_2']
            pose[3] = self.dataFrame.loc[self.index, 'joint_angle_3']
            pose[4] = self.dataFrame.loc[self.index, 'joint_angle_4']
            pose[5] = self.dataFrame.loc[self.index, 'joint_angle_5']
            pose[6] = self.dataFrame.loc[self.index, 'joint_angle_6']
            arm_group.set_joint_value_target(pose)
            print(f"going to cartesian next pose at index {self.index}")
            print(pose)
            arm_group.go(wait = True)
            if self.index < len(self.dataFrame) - 1:
                self.index += 20
            else:
                self.index += 1
        else:
            exit()
        

        

    def get_cartesian_pose1(self):
        arm_group = self.arm_group
        currentPose = arm_group.get_current_pose()
        return currentPose
        
    
if __name__ == "__main__":

    arm = JointStatePlayer1()
    # arm.go_to_home()
    try:
        while(not rospy.is_shutdown()):
            arm.go_to_next_joint_state()
            arm.rate.sleep()
    except KeyboardInterrupt:
        pass

    # arm.go_to_home()

