import rospy
from std_msgs.msg import String
from sensor_msgs.msg import NavSatFix
from mavros_msgs.srv import *
from nav_msgs.msg import *
from geometry_msgs.msg import TwistStamped,PoseStamped,Twist
from mavros_msgs.msg import *
from ping_nodelet.msg import Ping

class moving_test():
    def __init__(self):
        rospy.init_node('mission1_node', anonymous=True)
        rospy.wait_for_service('/mavros/set_mode')
        ModeService = rospy.ServiceProxy('/mavros/set_mode', mavros_msgs.srv.SetMode)
        ModeService(custom_mode='POSHOLD')
        rospy.wait_for_service('/mavros/cmd/arming')
        armService = rospy.ServiceProxy('/mavros/cmd/arming', mavros_msgs.srv.CommandBool)
        armService(True)
        # some messages to be published
        self.velocity_pub = rospy.Publisher('/mavros/rc/override', OverrideRCIn, queue_size=10)
        # some messages to be subscribed
        self.depth_sub = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.depthInfo)
        self.distance_sub = rospy.Subscriber('/ping_nodelet/ping', Ping, self.distanceInfo)

        # echo sounder
        self.depth = -1.0
        self.distance = 1.0
        self.confidence = 100
        rospy.on_shutdown(self.shutdown)

    def shutdown(self):
        rospy.loginfo("stop moving")
        go = OverrideRCIn()
        go.channels = [1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500]
        self.velocity_pub.publish(go)
        rospy.wait_for_service('/mavros/set_mode')
        ModeService = rospy.ServiceProxy('/mavros/set_mode', mavros_msgs.srv.SetMode)
        ModeService(custom_mode='MANUAL')
        rospy.wait_for_service('/mavros/cmd/arming')
        armService = rospy.ServiceProxy('/mavros/cmd/arming', mavros_msgs.srv.CommandBool)
        armService(False)

    def depthInfo(self, pose):
        self.depth = pose.pose.position.z

    def getdepth(self):
        return self.depth

    def distanceInfo(self, sounder):
        self.distance = sounder.distance
        self.confidence = sounder.confidence
        
    def getdistance(self):
        return self.distance, self.confidence
        
    def forward(self):
        go = OverrideRCIn()
        depth = self.getdepth()
        distance, confidence = self.getdistance()
        if depth < -0.18: # this value could be adjusted in different environment
            if distance > 0.45 and confidence == 100:
                go.channels = [1500, 1500, 1501, 1500, 1560, 1500, 1500, 1500]
                self.velocity_pub.publish(go)
            else:
                go.channels = [1500, 1500, 1501, 1500, 1500, 1500, 1500, 1500]
                self.velocity_pub.publish(go)
        else:
            if distance > 0.45 and confidence == 100:
                go.channels = [1500, 1500, 1500, 1500, 1560, 1500, 1500, 1500]
                self.velocity_pub.publish(go)
            else:
                go.channels = [1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500]
                self.velocity_pub.publish(go)

if __name__ == '__main__':
    mov = moving_test()
    while not rospy.is_shutdown():
        # mov.spin()
        mov.forward()
