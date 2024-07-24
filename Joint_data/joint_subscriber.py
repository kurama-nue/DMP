#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState
import moveit_commander
import os
import pandas as pd
from math import pi, degrees
from geometry_msgs.msg import PoseStamped


def joint_state_callback(data):
    joint_angles = data.position
    joint_velocities = data.velocity
    joint_effort = data.effort
    print("Joint angles:", joint_angles)
    print("Joint velocities:", joint_velocities)
    print("Joint effort:", joint_effort)
    
    dataa = {
        'timestamp': rospy.get_time()
    }
    for i in range(len(joint_angles)):
        dataa[f'joint_angle_{i}'] = joint_angles[i]
        dataa[f'joint_velocity_{i}'] = joint_velocities[i]
        dataa[f'joint_effort_{i}'] = joint_effort[i]
    
    dataFrame.loc[dataFrame.size] = dataa

def manipulator_subscriber():
    moveit_commander.roscpp_initialize("my_gen3")
    rospy.init_node('joint_state_subscriber')
    rospy.Subscriber('/my_gen3/joint_states', JointState, joint_state_callback)
    rospy.spin()
    print(f"Saved to {dataFileLocation}")
    dataFrame.to_csv(dataFileLocation, index=False)
       
dataFileLocation = os.path.join(os.path.expanduser("~"), "catkin_workspace", "Joint_State_data.csv")
columns = ['timestamp']
for j in range(7):
    columns.append(f'joint_angle_{j}')
    columns.append(f'joint_velocity_{j}')
    columns.append(f'joint_effort_{j}')
dataFrame = pd.DataFrame(columns=columns)

try:
    dataFrame = pd.read_csv(dataFileLocation)
    print("Loaded Data")
except:
    print("New file created")
    pass

if __name__ == '__main__':
    manipulator_subscriber()

