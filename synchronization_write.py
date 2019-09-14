# # for the real world
# import pickle
# from cv_bridge import CvBridge, CvBridgeError
#
# from message_filters import ApproximateTimeSynchronizer, Subscriber
# import sensor_msgs.msg
# import rospy
# from collections import deque
#
# import numpy
#
# bridge = CvBridge()
# rgb_path = 'test/rgb'
# depth_path= 'test/depth'
# data = deque()
#
# def write(data1, data2):
#     try:
#         rgb_image = bridge.imgmsg_to_cv2(data1,"bgr8")
#     except CvBridgeError as e:
#         print e
#     timestr = "%.6f" %  data1.header.stamp.to_sec()
#     print "rgb min {} max {}".format(numpy.min(rgb_image), numpy.max(rgb_image))
#
#     try:
#         depth_image = bridge.imgmsg_to_cv2(data2, data2.encoding)
#     except CvBridgeError as e:
#         print e
#     timestr = "%.6f" %  data2.header.stamp.to_sec()
#     print "depth {} {} min {} max {}".format(data2.encoding, depth_image.dtype, numpy.nanmin(depth_image), numpy.nanmax(depth_image))
#     data.append((rgb_image, depth_image))
#
# def shut():
#
#     pickle.dump(data, open("data.p", "wb"))
#     rospy.loginfo("Pickle the data")
#     rospy.sleep(0.5)
#
# def main():
#     rospy.init_node("syn")
#     rgb = Subscriber("/camera/rgb/image_rect_color", sensor_msgs.msg.Image)
#     # depth = Subscriber("/camera/depth_registered/image_raw", sensor_msgs.msg.Image)
#     depth = Subscriber("/camera/depth_registered/image_raw", sensor_msgs.msg.Image)
#     ats = ApproximateTimeSynchronizer([rgb, depth], queue_size=5, slop=0.1)
#     ats.registerCallback(write)
#     rospy.spin()
#     rospy.on_shutdown(shut)
#
# if __name__ == '__main__':
#     main()

# # for the gazebo world
# import pickle
# from cv_bridge import CvBridge, CvBridgeError
#
# from message_filters import ApproximateTimeSynchronizer, Subscriber
# import sensor_msgs.msg
# import rospy
# from collections import deque
#
# import numpy
#
# bridge = CvBridge()
# rgb_path = 'test/rgb'
# depth_path= 'test/depth'
# data = deque()
#
# def write(data1, data2):
#     try:
#         rgb_image = bridge.imgmsg_to_cv2(data1,"bgr8")
#     except CvBridgeError as e:
#         print e
#     timestr = "%.6f" %  data1.header.stamp.to_sec()
#     print "rgb min {} max {}".format(numpy.min(rgb_image), numpy.max(rgb_image))
#
#     try:
#         depth_image = bridge.imgmsg_to_cv2(data2, data2.encoding)
#     except CvBridgeError as e:
#         print e
#     timestr = "%.6f" %  data2.header.stamp.to_sec()
#     print "depth {} {} min {} max {}".format(data2.encoding, depth_image.dtype, numpy.nanmin(depth_image), numpy.nanmax(depth_image))
#     data.append((rgb_image, depth_image))
#
# def shut():
#     pickle.dump(data, open("data.p", "wb"))
#     rospy.loginfo("Pickle the data")
#     rospy.sleep(0.5)
#
# def main():
#     rospy.init_node("syn")
#     rgb = Subscriber("/camera/rgb/image_raw", sensor_msgs.msg.Image)
#     depth = Subscriber("/camera/depth/image_raw", sensor_msgs.msg.Image)
#     # depth = Subscriber("/camera/depth_registered/sw_registered/image_rect", sensor_msgs.msg.Image)
#     ats = ApproximateTimeSynchronizer([rgb, depth], queue_size=5, slop=0.1)
#     ats.registerCallback(write)
#     rospy.spin()
#     rospy.on_shutdown(shut)
#
# if __name__ == '__main__':
#     main()

from message_filters import ApproximateTimeSynchronizer, Subscriber
import rosbag
import sensor_msgs.msg
import rospy
import numpy as np

bag = rosbag.Bag('test0.bag', 'w')

def write(data1, data2):
    bag.write("/camera/rgb/image_rect_color", data1)
    print "rgb"
    bag.write("/camera/depth_registered/image_raw", data2)
    print "depth"

def shut():
    rospy.loginfo("Shut down the record")
    rospy.sleep(0.5)

def main():
    i = 0
    rospy.init_node("syn")
    rgb = Subscriber("/camera/rgb/image_rect_color", sensor_msgs.msg.Image)
    depth = Subscriber("/camera/depth_registered/image_raw", sensor_msgs.msg.Image)
    ats = ApproximateTimeSynchronizer([rgb, depth], queue_size=5, slop=0.1)
    ats.registerCallback(write)
    rospy.spin()
    rospy.on_shutdown(shut)

if __name__ == '__main__':
    main()
