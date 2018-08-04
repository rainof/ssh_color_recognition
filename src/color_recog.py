#importing modules

#!/usr/bin/env python


import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Point
import cv2   
import numpy as np
from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier
import os
import os.path


def color_detect():
    rospy.init_node('color_detection', anonymous=True)
    color_pub = rospy.Publisher('detect_color', String, queue_size=10)
    r = rospy.Rate(10)
    cap=cv2.VideoCapture(0)
    (ret, frame) = cap.read()
    prediction = 'n.a.'
    PATH = './training.data'
    if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
        print ('training data is ready, classifier is loading...')
    else:
        print ('training data is being created...')
        open('training.data', 'w')
        color_histogram_feature_extraction.training()
        print ('training data is ready, classifier is loading...')
    old_predict = prediction
    count = 0

    while not rospy.is_shutdown():
        (ret, frame) = cap.read() # record            
        cv2.putText(frame, 'Prediction: ' + prediction, (15, 45), cv2.FONT_HERSHEY_PLAIN, 4, 300)
        # Display the resulting frame
        cv2.imshow('color classifier', frame)

        color_histogram_feature_extraction.color_histogram_of_test_image(frame)

        prediction = knn_classifier.main('training.data', 'test.data')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if old_predict == prediction:
            count += 1
            print(">> " + prediction)
        else:
            count = 0
            old_predict = prediction

        if count > 20:
            color_pub.publish(prediction)
        r.sleep()

    cap.release()
    cv2.destroyAllWindows()	

if __name__ == '__main__':
    try:
        color_detect()
    except rospy.ROSInterruptException: pass
