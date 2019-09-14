import rospy
from std_msgs.msg import String
from sensor_msgs.msg import NavSatFix
from mavros_msgs.srv import *
from nav_msgs.msg import *
from geometry_msgs.msg import TwistStamped,PoseStamped,Twist
from mavros_msgs.msg import *
import tf
import numpy as np
latitude = 0.
longitude = 0.
euler = 0.

class moving_test():
    def __init__(self):
        self.x_goal = 100.
        self.y_goal = 50.
        rospy.init_node('mission1_node', anonymous=True)
        rospy.wait_for_service('/mavros/set_mode')
        armService = rospy.ServiceProxy('/mavros/cmd/arming', mavros_msgs.srv.CommandBool)
        armService(True)
        # some messages to be published
        self.velocity_pub = rospy.Publisher('/mavros/rc/override', OverrideRCIn, queue_size=10)
        # some messages to be subscribed
        self.orientation = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.poseCallback)
        # self.a = rospy.Subscriber("/mavros/global_position/raw/fix", NavSatFix, self.globalPositionCallback)
        rospy.on_shutdown(self.shutdown)

    def poseCallback(selfself, pose):
        global euler, x_present, y_present
        quaternion = (pose.pose.orientation.x,
					  pose.pose.orientation.y,
					  pose.pose.orientation.z,
					  pose.pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        x_present = pose.pose.position.x
        y_present = pose.pose.position.y

    def set_goal(self):
        pass

    # def globalPositionCallback(self, globalPositionCallback):
    #     global latitude
    #     global longitude
    #     latitude = globalPositionCallback.latitude
    #     longitude = globalPositionCallback.longitude

    def shutdown(self):
        rospy.loginfo("stop moving")
        go = OverrideRCIn()
        go.channels = [1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500]
        self.velocity_pub.publish(go)

    def spin(self):
        go = OverrideRCIn()
        while euler == 0.:
            pass
        if self.y_goal - y_present > 0:
            angle = np.arccos((self.x_goal - x_present)/np.sqrt(np.square(self.x_goal - x_present)+np.square(self.y_goal - y_present)))
            while np.abs(euler[2] - angle) > np.pi / 20:
                go.channels = [1500, 1500, 1500, 1502, 1500, 1500, 1500, 1500]
                self.velocity_pub.publish(go)
            go.channels = [1500, 1500, 1500, 1450, 1500, 1500, 1500, 1500]
            self.velocity_pub.publish(go)
            rospy.sleep(2.)
        else:
            angle = - np.arccos((self.x_goal - x_present)/np.sqrt(np.square(self.x_goal - x_present)+np.square(self.y_goal - y_present)))
            while np.abs(euler[2] - angle) > np.pi / 20:
                go.channels = [1500, 1500, 1500, 1498, 1500, 1500, 1500, 1500]
                self.velocity_pub.publish(go)
            go.channels = [1500, 1500, 1500, 1550, 1500, 1500, 1500, 1500]
            self.velocity_pub.publish(go)
            rospy.sleep(2.)
        go.channels = [1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500]
        self.velocity_pub.publish(go)
        rospy.sleep(2.)

    def forward(self):
        go = OverrideRCIn()
        distance = np.sqrt(np.square(x_present - self.x_goal) + np.square(y_present - self.y_goal))
        while distance > 2.:
            go.channels = [1500, 1500, 1500, 1500, 1510, 1500, 1500, 1500]
            distance = np.sqrt(np.square(x_present - self.x_goal) + np.square(y_present - self.y_goal))
            self.velocity_pub.publish(go)

        # print latitude, longitude, euler


if __name__ == '__main__':
    mov = moving_test()
    while not rospy.is_shutdown():
        mov.spin()
        mov.forward()
