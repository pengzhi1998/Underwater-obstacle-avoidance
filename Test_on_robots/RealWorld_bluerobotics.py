import rospy
import time
import numpy as np
import cv2
import copy

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import LaserScan
from mavros_msgs.msg import *
from mavros_msgs.srv import *
from ping_nodelet.msg import Ping

class RealWorld():
    def __init__(self):
        rospy.init_node('RealWorld', anonymous=False)
        rospy.wait_for_service('/mavros/set_mode')
        ModeService = rospy.ServiceProxy('/mavros/set_mode', mavros_msgs.srv.SetMode)
        ModeService(custom_mode='MANUAL')
        rospy.wait_for_service('/mavros/cmd/arming')
        armService = rospy.ServiceProxy('/mavros/cmd/arming', mavros_msgs.srv.CommandBool)
        armService(True)
        #------------Params--------------------
        self.depth_image_size = [160, 128]
        self.rgb_image_size = [304, 228]
        self.depth = -1.0
        self.distance = 1.0
        self.confidence = 100
        self.flag_depth = 0
        self.flag_move = 0
        self.bridge = CvBridge()
        self.rgb_image = None

        # self.action_table = [0.34, 0.26, np.pi/6 left, np.pi/12, 0., -np.pi/12 right, -np.pi/6]
        self.action_table = [1640, 1630, 1420, 1440, 1500, 1560, 1580]
        self.self_speed = [1635, 1500]

        self.start_time = time.time()

        self.step_target = [0., 0.]
        self.action = 0

        #-----------Publisher and Subscriber-------------
        self.cmd_vel = rospy.Publisher('/mavros/rc/override', OverrideRCIn, queue_size = 10)

        self.rgb_image_sub = rospy.Subscriber('camera/image_raw', Image, self.RGBImageCallBack)
        self.depth_sub = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.depthInfo)
        self.distance_sub = rospy.Subscriber('/ping_nodelet/ping', Ping, self.distanceInfo)
        rospy.sleep(2.)
        # What function to call when you ctrl + c
        rospy.on_shutdown(self.shutdown)

    def depthInfo(self, pose):
        self.depth = pose.pose.position.z

    def getdepth(self):
        return self.depth

    def distanceInfo(self, sounder):
        self.distance = sounder.distance
        self.confidence = sounder.confidence

    def getdistance(self):
        return self.distance, self.confidence

    def RGBImageCallBack(self, img):
        self.rgb_image = img

    def GetRGBImageObservation(self):
        # ros image to cv2 image
        while self.rgb_image == None:
            print "rgb_image not get yet"
        try:
            cv_img = self.bridge.imgmsg_to_cv2(self.rgb_image, "bgr8")
        except Exception as e:
            raise e
        # resize
        cv_img = cv_img[240:1680, :] # crop the image to make it the same leng-width ratio
        dim = (self.rgb_image_size[0], self.rgb_image_size[1])
        cv_resized_img = cv2.resize(cv_img, dim, interpolation = cv2.INTER_AREA)
        return(cv_resized_img)

    def FlagSetting(self):
        depth = self.getdepth()
        distance, confidence = self.getdistance()
        print depth, distance, confidence
        # constantly adjusting the depth of the bluerov
        if depth < -0.20:
            self.flag_depth = 0
        else:
            self.flag_depth = 1
        # if there is an obstacle in front of it, switch the control mode, control the robot with FCRN_DDDQN
        if distance < 0.8 and confidence == 100:
            self.flag_move = 1

    def Control(self, action):
        self.FlagSetting()
        if action <2:
            self.self_speed[0] = self.action_table[action]
        else:
            self.self_speed[1] = self.action_table[action]
        move_cmd = OverrideRCIn()
        move_cmd.channels[0] = 1500
        move_cmd.channels[1] = 1500
        move_cmd.channels[5] = 1500
        move_cmd.channels[6] = 1500
        move_cmd.channels[7] = 1500

        if self.flag_depth == 1:
            move_cmd.channels[2] = 1500 # throttle, should be controlled to keep in the height
        else:
            move_cmd.channels[2] = 1505

        if self.flag_move == 1:
            move_cmd.channels[3] = self.self_speed[1] # yaw
            move_cmd.channels[4] = self.self_speed[0] # forward
        else:
            move_cmd.channels[3] = 1500
            move_cmd.channels[4] = 1635
        print move_cmd
        self.cmd_vel.publish(move_cmd)

    def shutdown(self):
        # stop turtlebot
        rospy.loginfo("Stop Moving")
        go = OverrideRCIn()
        go.channels = [1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500]
        self.cmd_vel.publish(go)
        rospy.wait_for_service('/mavros/set_mode')
        ModeService = rospy.ServiceProxy('/mavros/set_mode', mavros_msgs.srv.SetMode)
        ModeService(custom_mode='MANUAL')
        rospy.wait_for_service('/mavros/cmd/arming')
        armService = rospy.ServiceProxy('/mavros/cmd/arming', mavros_msgs.srv.CommandBool)
        armService(False)
        rospy.sleep(1)
