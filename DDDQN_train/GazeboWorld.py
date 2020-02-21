import rospy
import math
import time
import numpy as np
import cv2
import copy
import tf
import random

import os

# print "{}".format(os.environ['ROS_PACKAGE_PATH'])

from geometry_msgs.msg import Twist
from geometry_msgs.msg import Quaternion
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from kobuki_msgs.msg import BumperEvent

class GazeboWorld():
	def __init__(self):
		# initiliaze
		rospy.init_node('GazeboWorld', anonymous=False)

		#-----------Default Robot State-----------------------
		# self.set_self_state = ModelState()
		# self.set_self_state.model_name = 'turtlebot3'
		# self.set_self_state.pose.position.x = -0.5
		# self.set_self_state.pose.position.y = -0.5
		# self.set_self_state.pose.position.z = 0.0
		# self.set_self_state.pose.orientation.x = 0.0
		# self.set_self_state.pose.orientation.y = 0.0
		# self.set_self_state.pose.orientation.z = 0.0
		# self.set_self_state.pose.orientation.w = 1.0
		# self.set_self_state.twist.linear.x = 0.
		# self.set_self_state.twist.linear.y = 0.
		# self.set_self_state.twist.linear.z = 0.
		# self.set_self_state.twist.angular.x = 0.
		# self.set_self_state.twist.angular.y = 0.
		# self.set_self_state.twist.angular.z = 0.
		# self.set_self_state.reference_frame = 'world'

		self.set_self_state1 = ModelState()
		self.set_self_state1.model_name = 'turtlebot3'
		self.set_self_state1.pose.position.x = -10. + np.random.uniform(-0.1, 0.1)
		self.set_self_state1.pose.position.y = -4. + np.random.uniform(-0.1, 0.1)
		self.set_self_state1.pose.position.z = 0.0
		self.set_self_state1.pose.orientation.x = 0.0
		self.set_self_state1.pose.orientation.y = 0.0
		self.set_self_state1.pose.orientation.z = 0.0
		self.set_self_state1.pose.orientation.w = 1.0
		self.set_self_state1.twist.linear.x = 0.
		self.set_self_state1.twist.linear.y = 0.
		self.set_self_state1.twist.linear.z = 0.
		self.set_self_state1.twist.angular.x = 0.
		self.set_self_state1.twist.angular.y = 0.
		self.set_self_state1.twist.angular.z = 0.
		self.set_self_state1.reference_frame = 'world'

		self.set_self_state2 = ModelState()
		self.set_self_state2.model_name = 'turtlebot3'
		self.set_self_state2.pose.position.x = -5. + np.random.uniform(-0.1, 0.1)
		self.set_self_state2.pose.position.y = -7. + np.random.uniform(-0.1, 0.1)
		self.set_self_state2.pose.position.z = 0.0
		self.set_self_state2.pose.orientation.x = 0.0
		self.set_self_state2.pose.orientation.y = 0.0
		self.set_self_state2.pose.orientation.z = 0.0
		self.set_self_state2.pose.orientation.w = 1.0
		self.set_self_state2.twist.linear.x = 0.
		self.set_self_state2.twist.linear.y = 0.
		self.set_self_state2.twist.linear.z = 0.
		self.set_self_state2.twist.angular.x = 0.
		self.set_self_state2.twist.angular.y = 0.
		self.set_self_state2.twist.angular.z = 0.
		self.set_self_state2.reference_frame = 'world'

		self.set_self_state3 = ModelState()
		self.set_self_state3.model_name = 'turtlebot3'
		self.set_self_state3.pose.position.x = 0.0 + np.random.uniform(-0.1, 0.1)
		self.set_self_state3.pose.position.y = -1. + np.random.uniform(-0.1, 0.1)
		self.set_self_state3.pose.position.z = 0.0
		self.set_self_state3.pose.orientation.x = 0.0
		self.set_self_state3.pose.orientation.y = 0.0
		self.set_self_state3.pose.orientation.z = 0.0
		self.set_self_state3.pose.orientation.w = 1.0
		self.set_self_state3.twist.linear.x = 0.
		self.set_self_state3.twist.linear.y = 0.
		self.set_self_state3.twist.linear.z = 0.
		self.set_self_state3.twist.angular.x = 0.
		self.set_self_state3.twist.angular.y = 0.
		self.set_self_state3.twist.angular.z = 0.
		self.set_self_state3.reference_frame = 'world'

		self.set_self_state4 = ModelState()
		self.set_self_state4.model_name = 'turtlebot3'
		self.set_self_state4.pose.position.x = -6. + np.random.uniform(-0.1, 0.1)
		self.set_self_state4.pose.position.y = -1. + np.random.uniform(-0.1, 0.1)
		self.set_self_state4.pose.position.z = 0.0
		self.set_self_state4.pose.orientation.x = 0.0
		self.set_self_state4.pose.orientation.y = 0.0
		self.set_self_state4.pose.orientation.z = 0.0
		self.set_self_state4.pose.orientation.w = 1.0
		self.set_self_state4.twist.linear.x = 0.
		self.set_self_state4.twist.linear.y = 0.
		self.set_self_state4.twist.linear.z = 0.
		self.set_self_state4.twist.angular.x = 0.
		self.set_self_state4.twist.angular.y = 0.
		self.set_self_state4.twist.angular.z = 0.
		self.set_self_state4.reference_frame = 'world'

		self.set_self_state5 = ModelState()
		self.set_self_state5.model_name = 'turtlebot3'
		self.set_self_state5.pose.position.x = 4.93 + np.random.uniform(-0.05, 0.05)
		self.set_self_state5.pose.position.y = 3.92 + np.random.uniform(-0.05, 0.05)
		self.set_self_state5.pose.position.z = 0.0
		self.set_self_state5.pose.orientation.x = 0.0
		self.set_self_state5.pose.orientation.y = 0.0
		self.set_self_state5.pose.orientation.z = 0.0
		self.set_self_state5.pose.orientation.w = 1.0
		self.set_self_state5.twist.linear.x = 0.
		self.set_self_state5.twist.linear.y = 0.
		self.set_self_state5.twist.linear.z = 0.
		self.set_self_state5.twist.angular.x = 0.
		self.set_self_state5.twist.angular.y = 0.
		self.set_self_state5.twist.angular.z = 0.
		self.set_self_state5.reference_frame = 'world'

		self.set_self_state6 = ModelState()
		self.set_self_state6.model_name = 'turtlebot3'
		self.set_self_state6.pose.position.x = -13.1 + np.random.uniform(-0.05, 0.05)
		self.set_self_state6.pose.position.y = 3.99 + np.random.uniform(-0.05, 0.05)
		self.set_self_state6.pose.position.z = 0.0
		self.set_self_state6.pose.orientation.x = 0.0
		self.set_self_state6.pose.orientation.y = 0.0
		self.set_self_state6.pose.orientation.z = 0.0
		self.set_self_state6.pose.orientation.w = 1.0
		self.set_self_state6.twist.linear.x = 0.
		self.set_self_state6.twist.linear.y = 0.
		self.set_self_state6.twist.linear.z = 0.
		self.set_self_state6.twist.angular.x = 0.
		self.set_self_state6.twist.angular.y = 0.
		self.set_self_state6.twist.angular.z = 0.
		self.set_self_state6.reference_frame = 'world'

		self.set_self_state7 = ModelState()
		self.set_self_state7.model_name = 'turtlebot3'
		self.set_self_state7.pose.position.x = 4.5 + np.random.uniform(-0.05, 0.05)
		self.set_self_state7.pose.position.y = -11.96 + np.random.uniform(-0.05, 0.05)
		self.set_self_state7.pose.position.z = 0.0
		self.set_self_state7.pose.orientation.x = 0.0
		self.set_self_state7.pose.orientation.y = 0.0
		self.set_self_state7.pose.orientation.z = 0.0
		self.set_self_state7.pose.orientation.w = 1.0
		self.set_self_state7.twist.linear.x = 0.
		self.set_self_state7.twist.linear.y = 0.
		self.set_self_state7.twist.linear.z = 0.
		self.set_self_state7.twist.angular.x = 0.
		self.set_self_state7.twist.angular.y = 0.
		self.set_self_state7.twist.angular.z = 0.
		self.set_self_state7.reference_frame = 'world'

		self.set_self_state8 = ModelState()
		self.set_self_state8.model_name = 'turtlebot3'
		self.set_self_state8.pose.position.x = -12.5 + np.random.uniform(-0.05, 0.05)
		self.set_self_state8.pose.position.y = -11.75 + np.random.uniform(-0.05, 0.05)
		self.set_self_state8.pose.position.z = 0.0
		self.set_self_state8.pose.orientation.x = 0.0
		self.set_self_state8.pose.orientation.y = 0.0
		self.set_self_state8.pose.orientation.z = 0.0
		self.set_self_state8.pose.orientation.w = 1.0
		self.set_self_state8.twist.linear.x = 0.
		self.set_self_state8.twist.linear.y = 0.
		self.set_self_state8.twist.linear.z = 0.
		self.set_self_state8.twist.angular.x = 0.
		self.set_self_state8.twist.angular.y = 0.
		self.set_self_state8.twist.angular.z = 0.
		self.set_self_state8.reference_frame = 'world'

		#------------Params--------------------
		self.depth_image_size = [160, 128]
		self.rgb_image_size = [304, 228]
		self.bridge = CvBridge()

		self.object_state = [0, 0, 0, 0]
		self.object_name = []

		# 0. | left 90/s | left 45/s | right 45/s | right 90/s | acc 1/s | slow down -1/s
		self.action_table = [0.34, 0.26, np.pi/6, np.pi/12, 0., -np.pi/12, -np.pi/6]

		self.self_speed = [.2, 0.0]
		self.default_states = None

		self.start_time = time.time()
		self.max_steps = 10000

		self.depth_image = None
		self.bump = False

		#-----------Publisher and Subscriber-------------
		# self.cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size = 10)
		self.cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size = 10)
		self.set_state = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size = 10)
		self.resized_depth_img = rospy.Publisher('camera/depth/image_resized',Image, queue_size = 10)
		self.resized_rgb_img = rospy.Publisher('camera/rgb/image_resized',Image, queue_size = 10)

		self.object_state_sub = rospy.Subscriber('gazebo/model_states', ModelStates, self.ModelStateCallBack)
		self.rgb_image_sub = rospy.Subscriber('camera/rgb/image_raw', Image, self.RGBImageCallBack)
		self.depth_image_sub = rospy.Subscriber('camera/depth/image_raw', Image, self.DepthImageCallBack)
		self.laser_sub = rospy.Subscriber('scan', LaserScan, self.LaserScanCallBack)
		self.odom_sub = rospy.Subscriber('odom', Odometry, self.OdometryCallBack)
		self.bumper_sub = rospy.Subscriber('mobile_base/events/bumper', BumperEvent, self.BumperCallBack)

		rospy.sleep(2.)

		# What function to call when you ctrl + c
		rospy.on_shutdown(self.shutdown)


	def ModelStateCallBack(self, data):
		# self state
		# idx = data.name.index('turtlebot3_waffle')
		idx = data.name.index('turtlebot3')
		quaternion = (data.pose[idx].orientation.x,
					  data.pose[idx].orientation.y,
					  data.pose[idx].orientation.z,
					  data.pose[idx].orientation.w)
		euler = tf.transformations.euler_from_quaternion(quaternion)
		roll = euler[0]
		pitch = euler[1]
		yaw = euler[2]
		self.self_state = [data.pose[idx].position.x,
					  	  data.pose[idx].position.y,
					  	  yaw,
					  	  data.twist[idx].linear.x,
						  data.twist[idx].linear.y,
						  data.twist[idx].angular.z]
		for lp in xrange(len(self.object_name)):
			idx = data.name.index(self.object_name[lp])
			quaternion = (data.pose[idx].orientation.x,
						  data.pose[idx].orientation.y,
						  data.pose[idx].orientation.z,
						  data.pose[idx].orientation.w)
			euler = tf.transformations.euler_from_quaternion(quaternion)
			roll = euler[0]
			pitch = euler[1]
			yaw = euler[2]

			self.object_state[lp] = [data.pose[idx].position.x,
									 data.pose[idx].position.y,
									 yaw]
		if self.default_states is None:
			self.default_states = copy.deepcopy(data)


	def DepthImageCallBack(self, img):
		self.depth_image = img

	def RGBImageCallBack(self, img):
		self.rgb_image = img

	def LaserScanCallBack(self, scan):
		self.scan_param = [scan.angle_min, scan.angle_max, scan.angle_increment, scan.time_increment,
						   scan.scan_time, scan.range_min, scan. range_max]
		self.scan = np.array(scan.ranges)

	def OdometryCallBack(self, odometry):
		# store the current position information in the world into the variable self_position_x/y
		self.self_position_x = odometry.pose.pose.position.x
		self.self_position_y = odometry.pose.pose.position.y
		self.quaternion = (odometry.pose.pose.orientation.x, odometry.pose.pose.orientation.y,
						   odometry.pose.pose.orientation.z, odometry.pose.pose.orientation.w)
		# self.self_orientation_x = odometry.pose.pose.orientation.x
		# self.self_orientation_y = odometry.pose.pose.orientation.y
		# self.self_orientation_z = odometry.pose.pose.orientation.z
		# self.self_orientation_w = odometry.pose.pose.orientation.w
		self.self_linear_x_speed = odometry.twist.twist.linear.x
		self.self_linear_y_speed = odometry.twist.twist.linear.y
		self.self_rotation_z_speed = odometry.twist.twist.angular.z

	def BumperCallBack(self, bumper_data):
		if bumper_data.state == BumperEvent.PRESSED:
			self.bump = True
		else:
			self.bump = False

	def GetDepthImageObservation(self):
		# ros image to cv2 image

		try:
			cv_img = self.bridge.imgmsg_to_cv2(self.depth_image, "32FC1")
		except Exception as e:
			raise e
		try:
			cv_rgb_img = self.bridge.imgmsg_to_cv2(self.rgb_image, "bgr8")
		except Exception as e:
			raise e
		cv_img = np.array(cv_img, dtype=np.float32)
		# resize
		dim = (self.depth_image_size[0], self.depth_image_size[1])
		# cv_img = cv_img[45:435, 60:580]
		cv_img = cv2.resize(cv_img, dim, interpolation = cv2.INTER_AREA)

		cv_img[np.isnan(cv_img)] = 0.
		cv_img[cv_img < 0.4] = 0.
		# cv_img/=(10./255.)

		# cv_img/=(10000./255.)
		# print 'max:', np.amax(cv_img), 'min:', np.amin(cv_img)
		# cv_img[cv_img > 5.] = -1.

		# guassian noise
		gauss = np.random.normal(0., 0.15, dim)
		gauss = gauss.reshape(dim[1], dim[0])
		cv_img = np.array(cv_img, dtype=np.float32)
		cv_img = cv_img + gauss
		cv_img[cv_img<0.4] = 0.

		cv_img = np.array(cv_img, dtype=np.float32)
		# cv_img*=(10./255.)

		# cv2 image to ros image and publish
		try:
			resized_img = self.bridge.cv2_to_imgmsg(cv_img, "passthrough")
		except Exception as e:
			raise e
		self.resized_depth_img.publish(resized_img)
		# return(cv_img/5.)
		# cv2.imwrite("img.png", cv_img * 25)
		return(cv_img)

	def GetRGBImageObservation(self):
		# ros image to cv2 image
		try:
			cv_img = self.bridge.imgmsg_to_cv2(self.rgb_image, "bgr8")
		except Exception as e:
			raise e
		# resize
		dim = (self.rgb_image_size[0], self.rgb_image_size[1])
		cv_resized_img = cv2.resize(cv_img, dim, interpolation = cv2.INTER_AREA)
		# cv2 image to ros image and publish
		try:
			resized_img = self.bridge.cv2_to_imgmsg(cv_resized_img, "bgr8")
		except Exception as e:
			raise e
		self.resized_rgb_img.publish(resized_img)
		return(cv_resized_img)

	def PublishDepthPrediction(self, depth_img):
		# cv2 image to ros image and publish
		cv_img = np.array(depth_img, dtype=np.float32)
		try:
			resized_img = self.bridge.cv2_to_imgmsg(cv_img, "passthrough")
		except Exception as e:
			raise e
		self.resized_depth_img.publish(resized_img)

	def GetLaserObservation(self):
		scan = copy.deepcopy(self.scan)
		scan[np.isinf(scan)] = 30.
		return scan

	def GetSelfState(self):
		return self.self_state;

	def GetSelfLinearXSpeed(self):
		return self.self_linear_x_speed

	def GetSelfOdomeInfo(self):
		euler = tf.transformations.euler_from_quaternion(self.quaternion) # the euler orientation
		Eular = euler[2]
		# distance = np.sqrt((self.state2_goal_x - self.self_position_x)**2 + (self.state2_goal_y - self.self_position_y)**2)
		v = np.sqrt(self.self_linear_x_speed**2 + self.self_linear_y_speed**2)
		return [v, self.self_rotation_z_speed, Eular]

	def GetTargetState(self, name):
		return self.object_state[self.TargetName.index(name)]

	def GetSelfSpeed(self):
		return np.array(self.self_speed)

	def GetBump(self):
		return self.bump

	def SetObjectPose(self, name='mobile_base', random_flag=False):
		if name is 'mobile_base' :
			rand = random.random()
			# object_state = copy.deepcopy(self.set_self_state)
			# quaternion = tf.transformations.quaternion_from_euler(0., 0., np.random.uniform(-np.pi, np.pi))
			if rand < 0.125:
				object_state = copy.deepcopy(self.set_self_state1)
				quaternion = tf.transformations.quaternion_from_euler(0., 0., np.random.uniform(-np.pi, np.pi))
			elif rand < 0.250:
				object_state = copy.deepcopy(self.set_self_state2)
				quaternion = tf.transformations.quaternion_from_euler(0., 0., np.random.uniform(-np.pi, np.pi))
			elif rand < 0.375:
				object_state = copy.deepcopy(self.set_self_state3)
				quaternion = tf.transformations.quaternion_from_euler(0., 0., np.random.uniform(-np.pi, np.pi))
			elif rand < 0.500:
				object_state = copy.deepcopy(self.set_self_state4)
				quaternion = tf.transformations.quaternion_from_euler(0., 0., np.random.uniform(-np.pi, np.pi))
			elif rand < 0.625:
				object_state = copy.deepcopy(self.set_self_state5)
				quaternion = tf.transformations.quaternion_from_euler(0., 0., 3.1415 + np.random.uniform(-np.pi, np.pi) / 8)
			elif rand < 0.750:
				object_state = copy.deepcopy(self.set_self_state6)
				quaternion = tf.transformations.quaternion_from_euler(0., 0., 0.0 + np.random.uniform(-np.pi, np.pi) / 8)
			elif rand < 0.875:
				object_state = copy.deepcopy(self.set_self_state7)
				quaternion = tf.transformations.quaternion_from_euler(0., 0., 0.0 + np.random.uniform(-np.pi, np.pi) / 8)
			else:
				object_state = copy.deepcopy(self.set_self_state8)
				quaternion = tf.transformations.quaternion_from_euler(0., 0., 3.1415 + np.random.uniform(-np.pi, np.pi) / 8)


			object_state.pose.orientation.x = quaternion[0]
			object_state.pose.orientation.y = quaternion[1]
			object_state.pose.orientation.z = quaternion[2]
			object_state.pose.orientation.w = quaternion[3]
		else:
			object_state = self.States2State(self.default_states, name)

		self.set_state.publish(object_state)

	def States2State(self, states, name):
		to_state = ModelState()
		from_states = copy.deepcopy(states)
		idx = from_states.name.index(name)
		to_state.model_name = name
		to_state.pose = from_states.pose[idx]
		to_state.twist = from_states.twist[idx]
		to_state.reference_frame = 'world'
		return to_state


	def ResetWorld(self):
		self.total_reward = 0
		self.SetObjectPose() # reset robot
		# for x in xrange(len(self.object_name)):
		# 	self.SetObjectPose(self.object_name[x]) # reset target
		self.self_speed = [.4, 0.0]
		self.step_target = [0., 0.]
		self.step_r_cnt = 0.
		self.start_time = time.time()
		rospy.sleep(0.5)

	def Control(self, action):
		if action < 2:
			self.self_speed[0] = self.action_table[action]
			# self.self_speed[1] = 0.
		else:
			self.self_speed[1] = self.action_table[action]
		move_cmd = Twist()
		move_cmd.linear.x = self.self_speed[0]
		move_cmd.linear.y = 0.
		move_cmd.linear.z = 0.
		move_cmd.angular.x = 0.
		move_cmd.angular.y = 0.
		move_cmd.angular.z = self.self_speed[1]
		self.cmd_vel.publish(move_cmd)

	def shutdown(self):
		# stop turtlebot
		rospy.loginfo("Stop Moving")
		self.cmd_vel.publish(Twist())
		rospy.sleep(1)

	def GetRewardAndTerminate(self, t):
		terminate = False
		reset = False
		[v, theta, Eular] = self.GetSelfOdomeInfo()
		laser = self.GetLaserObservation()  # laser[0] is the front and laser[180] is the rear
		Laser = []
		for i in range(-90, 90):
			Laser = np.append(Laser, laser[i])

		Distance = np.amin(Laser)
		Angle = np.abs(np.argmin(Laser) - 90)

		# initial reward function:
		# reward = v * np.cos(theta) * 0.2 - 0.01
		#
		# if self.GetBump() or np.amin(laser) < 0.9:
		# 	reward = -10.
		# 	terminate = True
		# 	reset = True
		# if t > 500:
		# 	reset = True
		#   print "SUCCESS!!!!!!!!!!!!!!!!!!!!!!!"

		# improved reward function:
		reward = v * np.cos(1.05 * theta) * 0.355 - 0.009
		# compute the value of laser scan

		if 0.9 < Distance < 1.2:
			reward = (- 2.4 + 2 * Distance) * (180 - Angle) / 135
		if self.GetBump() or Distance < 0.9:
			reward = (-8.0) * ((((180. - Angle) / 135. - 1.) * 3. / 8.) + 1.)
			terminate = True
			reset = True

		self.total_reward = self.total_reward + reward
		total_reward = self.total_reward
		if t > 500:
			reset = True
			print "SUCCESS!!!!!!!!!!!!!!!!!!!!!!!"

		if

		return reward, terminate, reset, total_reward, evaluation_index
