import copy
from PIL import Image, ImageEnhance
import numpy as np
import pickle
import cv2
np.set_printoptions(threshold=np.inf)

data_rgb = pickle.load(open("data_rgb.p", "rb"))
data_depth = pickle.load(open("data_depth.p", "rb"))
path = 'overlap_showing/'
def blend_two_images():
    index = 0
    for rgb_image, depth_image in zip(data_rgb, data_depth):
        mask = copy.deepcopy(depth_image)
        mask = np.isnan(mask)
        mask = 1.0 - (mask + np.zeros((480, 640)))
        print "max, min:", np.nanmax(depth_image), np.nanmin(depth_image)
        # cv2.imshow("depth", depth_image/5)
        # cv2.waitKey()
        # depth_image = depth_image[:, :, np.newaxis] # rgb_image, (640, 480, 1), interpolation=cv2.INTER_CUBIC)
        # rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        # # cv2.imshow("rgb", rgb_image)
        # cv2.waitKey()
        # rgb_image = np.asarray(rgb_image, np.float32)
        # cv2.imshow("img", rgb_image)
        # cv2.waitKey()
        # depth_image = np.asarray(depth_image, np.float32)
        # # print rgb_image.shape, depth_image.type
        # img = cv2.addWeighted(depth_image, 0.9, rgb_image, 0.1, 0)
        # cv2.imshow("img", img)
        # cv2.waitKey()
        image_name = "depth" + str(index) + ".png"
        depth_image = np.array(depth_image, np.float32)
        depth_image = depth_image * 1000
        depth_image = np.array(depth_image, np.uint16)
        cv2.imwrite(path + image_name, depth_image)
        image_name = "rgb" + str(index) + ".png"
        cv2.imwrite(path + image_name, rgb_image)
        index += 1
    img1 = Image.open(path + "depth5500.png")
    enhancer = ImageEnhance.Brightness(img1)
    enhanced_img = enhancer.enhance(2)
    # print "before reshaping:", list(img1.getdata())
    # box1 = (60, 45, 580, 435)
    # img1 = img1.crop(box1)
    # img1 = img1.resize((640, 480))
    # print "after reshaping:", img1.size
    img1 = enhanced_img.convert('RGBA')
    img1.show()

    img2 = Image.open(path + "rgb5500.png")
    # print "before reshaping:", img2.size
    # box2 = (60, 45, 580, 435)
    # img2 = img2.crop(box2)
    # img2 = img2.resize((640, 480))
    # print "after reshaping:", img2.size
    img2 = img2.convert('RGBA')
    img2.show()
    a = cv2.imread(path + "depth36.png", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    img = Image.blend(img1, img2, 0.3)
    img.show()
    img.save("blend.png")


blend_two_images()
