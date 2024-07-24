#!/usr/bin/env python3
import sys
import rospy 
import rospkg
import pandas as pd
import os
import moveit_commander
from geometry_msgs.msg import PoseStamped
from math import pi


class CartesianRecorder:
    def __init__(self) -> None:
        moveit_commander.roscpp_initialize("my_gen3")
        rospy.init_node('record_data')
        arm_group_name = "arm"
        self.arm_group = moveit_commander.MoveGroupCommander(arm_group_name, ns=rospy.get_namespace())
        self.rate = rospy.Rate(2)
        self.dataFileLocation = os.path.join("~/catkin_workspace", "DMP_cartesian_data.csv")
        self.dataFrame = pd.DataFrame(columns=['p_x', 'p_y', 'p_z', 'o_w', 'o_x', 'o_y', 'o_z'])
        try:
            self.dataFrame = pd.read_csv(self.dataFileLocation)
            print("Loaded Data")
        except:
            print("new file created")
            pass

    def get_cartesian_pose(self):
        arm_group = self.arm_group
        currentPose = arm_group.get_current_pose().pose
        data = {'p_x' : currentPose.position.x,
                'p_y' : currentPose.position.y,
                'p_z' : currentPose.position.z,
                'o_w' : currentPose.orientation.w,
                'o_x' : currentPose.orientation.x,
                'o_y' : currentPose.orientation.y,
                'o_z' : currentPose.orientation.z}
        self.dataFrame.loc[self.dataFrame.size] = data
        
    
    def saveData(self):
        print(f"Saved to {self.dataFileLocation}")
        self.dataFrame.to_csv(self.dataFileLocation)
    
if __name__ == "__main__":
    recorder = CartesianRecorder()
    try:
        while(not rospy.is_shutdown()):
            recorder.get_cartesian_pose()
            # recorder.rate.sleep()
    except KeyboardInterrupt:
        pass
    recorder.saveData()

