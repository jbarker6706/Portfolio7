#!/usr/bin/env python

import rospy
from std_msgs.msg import String

import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

cardir = "/home/jbarker6706/Documents/MSDS462/assignment7/test/testcars/"
roaddir = "/home/jbarker6706/Documents/MSDS462/assignment7/test/testemptyroads/"
humandir = "/home/jbarker6706/Documents/MSDS462/assignment7/test/testhumans/"

rospy.init_node('camera_sim_publisher')

pub = rospy.Publisher('camera_sim', String, queue_size=10)

rate = rospy.Rate(2)

# load model
model = load_model('small_last4.h5')
model.summary()

image_class = ["car", "open road", "human", "people on road"]

def classify_image(image):
    image_array = img_to_array(image)
    image_array = image_array.reshape((1, image_array.shape[0], image_array.shape[1], image_array.shape[2]))
    image_array = preprocess_input(image_array)
    yhat = model.predict(image_array)
    print(yhat)
    return image_class[np.argmax(yhat)]

msg = []
car_image = load_img(cardir+"5.png", target_size=(224,224))
msg.append(classify_image(car_image))

road_image = load_img(roaddir+"4.png", target_size=(224,224))
msg.append(classify_image(road_image))

human_image = load_img(humandir+"6.png", target_size=(224,224))
msg.append(classify_image(human_image))

count = 0

while not rospy.is_shutdown():
    pub.publish(msg[count%3])
    count += 1
    rate.sleep()
