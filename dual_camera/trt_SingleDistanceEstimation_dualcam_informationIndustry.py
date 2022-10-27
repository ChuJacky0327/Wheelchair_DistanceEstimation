"""trt_yolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""

import os
import time
import argparse

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.camera2 import add_camera_args2, Camera2
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo import TRT_YOLO

import warnings
import numpy as np

import serial
from time import sleep
import sys
from paho.mqtt import client as mqtt_client
import random
import Jetson.GPIO as GPIO
import json




warnings.simplefilter(action='ignore', category=FutureWarning)
WINDOW_NAME = 'Face_detection'




'''######################  MQTT  ###################'''
#broker = 'broker.hivemq.com' #test
#port = 1883  #TCP
broker = '34.81.245.131' #informationIndustry
port = 31883  #informationIndustry
#topic_warn_left = 'warn_left'
#topic_object_left = 'object_left'
#topic_distance_left = 'distance_left'
#topic_warn_right = 'warn_right'
#topic_object_right = 'object_right'
#topic_distance_right = 'distance_right'
topic_1left_event = 'SmartWheelchair/WHILL_MODEL_M_YILAN_001/event'
topic_1right_event = 'SmartWheelchair/1_right/event'
topic_1_message= 'SmartWheelchair/1/message'
topic_1_check = 'SmartWheelchair/1/sanity_check'
topic_1_response = 'SmartWheelchair/1/sanity_response'
client_id = f'python-mqtt-{random.randint(0, 1000)}'
def publish(client,warn_left,object1_left,distance_mq_left,warn_right,object1_right,distance_mq_right,now_time,person_leftup,person_rightdown):
    '''
    msg_warn_left = warn_left
    msg_distance_left = distance_mq_left
    msg_object_left = object1_left
    msg_warn_right = warn_right
    msg_distance_right = distance_mq_right
    msg_object_right = object1_right
    result_warn_left = client.publish(topic_warn_left, msg_warn_left)
    result_distance_left = client.publish(topic_distance_left, msg_distance_left)
    result_object_left = client.publish(topic_object_left, msg_object_left)
    result_warn_right = client.publish(topic_warn_right, msg_warn_right)
    result_distance_right = client.publish(topic_distance_right, msg_distance_right)
    result_object_right = client.publish(topic_object_right, msg_object_right)
    '''

    msg_1left_event = json.dumps({'event' : "CollisonDetect",'level' : warn_left,'time' : now_time,'distancee':distance_mq_left,'detected_object' : object1_left,"position" : {"leftup" : person_leftup,"rightdown" : person_rightdown}})
    result_1left_event = client.publish(topic_1left_event, msg_1left_event)
    #print(msg_1left_event)
    msg_1right_event = json.dumps({'event' : "CollisonDetect",'level' : warn_right,'time' : now_time,'distancee':distance_mq_right,'detected_object' : object1_right})
    result_1right_event = client.publish(topic_1right_event, msg_1right_event)
    
    msg_1_message= json.dumps({'message_type' : "PeerToPeer",'time' : now_time,'message' : 'welcome'})
    result_1_message = client.publish(topic_1_message, msg_1_message)
    msg_1_check = now_time
    result_1_check = client.publish(topic_1_check, msg_1_check)
    msg_1_response = now_time
    result_1_response = client.publish(topic_1_response, msg_1_response)
    #print(msg_1_response)


    '''
    status = result_warn_left[0]
    if status == 0:
        print(f"Send `{msg_warn_left}` to topic `{topic_warn_left}`")
    else:
        print(f"Failed to send message to topic {topic_warn_left}")
    '''

def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)
    # Set Connecting Client ID
    client = mqtt_client.Client(client_id)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

def run(warn_left,object1_left,distance_mq_left,warn_right,object1_right,distance_mq_right,now_time,person_leftup,person_rightdown):
    client = connect_mqtt()
    publish(client,warn_left,object1_left,distance_mq_left,warn_right,object1_right,distance_mq_right,now_time,person_leftup,person_rightdown)  

'''######################  MQTT  ###################'''




def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-c', '--category_num', type=int, default=1,
        help='number of object categories [1]')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov4|yolov4-tiny]-'
              '[{dimension}], where dimension could be a single '
              'number (e.g. 288, 416, 608)'))
    args = parser.parse_args()
    return args

def weight_choice(leftup ,rightup):# (boxWH[0],boxWH[2])
    midpoint = (abs(rightup-leftup)/2)+leftup
    result = abs(320-midpoint)
    if result >0 and result < 64:
        weight = 1
    elif result >64 and result < 128:
        weight = 0.8
    elif result >128 and result <192:
        weight = 0.6
    elif result >192 and result <256:
        weight = 0.4
    elif result >256 :
        weight = 0.2
    return weight

def BlindSpot(W,H,leftup ,rightup):#(W,H,boxWH[0],boxWH[2]) for object
    area = W*H
    img_area = 640*480
    midpoint = (abs(rightup-leftup)/2)+leftup
    result = 320-midpoint
    if area > img_area*0.5:
        blind = True
        if result >0 :
            print ("blind spot, left have obstacle, please turn right")
        else :
            print ("blind spot, right have obstacle, please turn left")
    else:
        blind = False
    return blind

def distance(object_width , W):
    #focallength = 881.25 #for640
    focallength = 439.16 #for320
    cm = focallength * object_width / W
    return cm

def choice_distance(left_weight,right_weight,left_cm,right_cm,left_blind,right_blind,classname):#if there are blind spots on the left and right camera ,stop without judging the distance
    if left_blind == True or right_blind == True :
        print('blind spot ,stop without judging the distance' )
    else :
        if left_weight > right_weight :
            dis = classname + str(round(left_cm,2)) + 'cm'
            #print(classname + str(left_cm) + 'cm')
            return dis
        elif left_weight < right_weight :
            dis = classname + str(round(right_cm,2)) + 'cm'
            #print(classname + str(right_cm) + 'cm')
            return dis
        elif left_weight == right_weight :
            dis = classname + str(round(right_cm,2)) + 'cm'
            #print(classname + str(right_cm) + 'cm')
            return dis

def loop_and_detect(cam,cam2, trt_yolo, conf_th, vis):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    outpin_pin = 12
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(outpin_pin,GPIO.OUT,initial=GPIO.HIGH)
    warn_left = ''
    object1_left = ''
    distance_mq_left = ''
    warn_right = ''
    object1_right = ''
    distance_mq_right = ''
    person_leftup = 0
    person_rightdown = 0
    full_scrn = False
    count = 0
    fps = 0.0
    tic = time.time()
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        left_img = cam.read()
        right_img = cam2.read()
        if (left_img is None) or (right_img is None):
            break
        left_img = cv2.flip(left_img,0)
        right_img = cv2.flip(right_img,0)
        left_boxes, left_confs, left_clss = trt_yolo.detect(left_img, conf_th)
        right_boxes, right_confs, right_clss = trt_yolo.detect(right_img, conf_th)
        
        #print(left_clss)
        #print(right_clss)
        try :
            left_person_cm = 0
            left_bottle_cm= 0
            for i in range(0,len(left_clss)):
                if left_clss[i] == 39 :
                    bottle_width = 8 
                    boxWH = left_boxes[i]
                    W = boxWH[2] - boxWH[0]
                    H = boxWH[3] - boxWH[1]
                    left_bottle_cm = distance(bottle_width , W)
                    left_bottle_weight = weight_choice(boxWH[0],boxWH[2])
                    left_bottle_blind = BlindSpot(W,H,boxWH[0],boxWH[2])
                    #print(boxWH)
                    #print(W,H)
                    #print('bottle :' + str(cm) + 'cm , left')
                    #print('bottle OK')
                if left_clss[i] == 0 :
                    person_width = 55
                    boxWH = left_boxes[i]# [0][1]=leftup   [2][3]=rightdown 
                    W = boxWH[2] - boxWH[0]
                    H = boxWH[3] - boxWH[1] 
                    left_person_cm = distance(person_width , W)
                    left_person_weight = weight_choice(boxWH[0],boxWH[2])
                    left_person_blind = BlindSpot(W,H,boxWH[0],boxWH[2])
                    person_leftup,person_rightdown = boxWH[0],boxWH[2]
                    person_leftup,person_rightdown =str(person_leftup),str(person_rightdown)
                    #print(boxWH)
                    #print('person left')
        except :
            print('left object NO')

        try :
            right_person_cm = 0 
            right_bottle_cm= 0 
            for i in range(0,len(right_clss)):
                if right_clss[i] == 39 :
                    bottle_width = 8 
                    boxWH = right_boxes[i]
                    W = boxWH[2] - boxWH[0]
                    H = boxWH[3] - boxWH[1]
                    right_bottle_cm = distance(bottle_width , W)
                    right_bottle_weight = weight_choice(boxWH[0],boxWH[2])
                    right_bottle_blind = BlindSpot(W,H,boxWH[0],boxWH[2])
                    #print(boxWH)
                    #print(W,H)
                    #print('bottle :' + str(cm) + 'cm , right')
                    #print('bottle OK')
                if right_clss[i] == 0 :
                    person_width = 55
                    boxWH = right_boxes[i]
                    W = boxWH[2] - boxWH[0]
                    H = boxWH[3] - boxWH[1] 
                    right_person_cm = distance(person_width , W)
                    right_person_weight = weight_choice(boxWH[0],boxWH[2])
                    right_person_blind = BlindSpot(W,H,boxWH[0],boxWH[2])
                    person_leftup,person_rightdown = boxWH[0],boxWH[2]
                    person_leftup,person_rightdown =str(person_leftup),str(person_rightdown)
                    #print(W,H)
                    #print('person right')
        except :
            print('right object NO')

        #print('person :' + str(round(left_person_cm,2)) + 'cm , left')
        #print('person :' + str(round(right_person_cm,2)) + 'cm , right')
        '''
        try : #person
            if right_person_cm != 0 and left_person_cm != 0 :
                classname = 'person'
                dis = choice_distance(left_person_weight,right_person_weight,left_person_cm,right_person_cm,left_person_blind,right_person_blind,classname)
                cv2.putText(left_img,dis,(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
                cv2.putText(right_img,dis,(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
            elif  right_person_cm == 0 and left_person_cm != 0 :
                #print('person :' + str(left_person_cm) + 'cm , left')
                cv2.putText(left_img,('person :' + str(round(left_person_cm,2)) + 'cm , left'),(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
            elif  right_person_cm != 0 and left_person_cm == 0 :
                #print('person :' + str(right_person_cm) + 'cm , right')
                cv2.putText(right_img,('person :' + str(round(right_person_cm,2)) + 'cm , right'),(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
            else :
                pass
        except:
            pass
        '''
        left_img = vis.draw_bboxes(left_img, left_boxes, left_confs, left_clss)
        right_img = vis.draw_bboxes(right_img, right_boxes, right_confs, right_clss)
        left_img = show_fps(left_img, fps)
        right_img = show_fps(right_img, fps)
        
        #cv2.putText(left_img,('person :' + str(round(left_person_cm,2)) + 'cm , left'),(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
        #cv2.putText(right_img,('person :' + str(round(right_person_cm,2)) + 'cm , right'),(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)

        if round(left_person_cm,2) < 300 and round(left_person_cm,2) != 0:
            #cv2.putText(left_img,('person :' + str(round(left_person_cm,2)) + 'cm , right'),(30,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3,cv2.LINE_AA)#for 640
            cv2.putText(left_img,('person :' + str(round(left_person_cm,2)) + 'cm , right'),(5,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3,cv2.LINE_AA)#for 320
            warn_left = 'warn_left'
            object1_left = 'person'
            distance_mq_left = str(round(left_person_cm,2))
            for i in range (0,200):
                GPIO.output(outpin_pin,GPIO.HIGH)
                GPIO.output(outpin_pin,GPIO.LOW)
        elif round(left_person_cm,2) > 300:
            #cv2.putText(left_img,('person :' + str(round(left_person_cm,2)) + 'cm , right'),(40,50),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),3,cv2.LINE_AA)#for 640
            cv2.putText(left_img,('person :' + str(round(left_person_cm,2)) + 'cm , right'),(15,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2,cv2.LINE_AA)#for 320
            warn_left = 'safe_left'
            object1_left = 'person'
            distance_mq_left = str(round(left_person_cm,2))
            GPIO.output(outpin_pin,GPIO.LOW)

        if round(right_person_cm,2) < 300 and round(right_person_cm,2) != 0:
            #cv2.putText(right_img,('person :' + str(round(right_person_cm,2)) + 'cm , right'),(30,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3,cv2.LINE_AA)#for 640
            cv2.putText(right_img,('person :' + str(round(right_person_cm,2)) + 'cm , right'),(5,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3,cv2.LINE_AA)#for 320
            warn_right = 'warn_right'
            object1_right  = 'person'
            distance_mq_right = str(round(right_person_cm,2))
            for i in range (0,200):
                GPIO.output(outpin_pin,GPIO.HIGH)
                GPIO.output(outpin_pin,GPIO.LOW)
        elif round(right_person_cm,2) > 300:
            #cv2.putText(right_img,('person :' + str(round(right_person_cm,2)) + 'cm , right'),(40,50),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),3,cv2.LINE_AA)#for 640
            cv2.putText(right_img,('person :' + str(round(right_person_cm,2)) + 'cm , right'),(15,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2,cv2.LINE_AA)#for 320
            warn_right = 'safe_right'
            object1_right  = 'person'
            distance_mq_right = str(round(right_person_cm,2))
            GPIO.output(outpin_pin,GPIO.LOW)


        
        cv2.imshow(WINDOW_NAME, left_img)
        cv2.imshow('right', right_img)
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        now_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())

        '''######################  MQTT  ###################'''
        run(warn_left,object1_left,distance_mq_left,warn_right,object1_right,distance_mq_right,now_time,person_leftup,person_rightdown)
        '''######################  MQTT  ###################'''

        tic = toc
        key = cv2.waitKey(1)

        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)


def main():
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)
    cam2 = Camera2(args)
    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    cls_dict = get_cls_dict(args.category_num)
    yolo_dim = args.model.split('-')[-1]
    if 'x' in yolo_dim:
        dim_split = yolo_dim.split('x')
        if len(dim_split) != 2:
            raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)
        w, h = int(dim_split[0]), int(dim_split[1])
    else:
        h = w = int(yolo_dim)
    if h % 32 != 0 or w % 32 != 0:
        raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)

    trt_yolo = TRT_YOLO(args.model, (h, w), args.category_num)

    open_window(
        WINDOW_NAME, 'left',
        cam.img_width, cam.img_height)
    vis = BBoxVisualization(cls_dict)
    loop_and_detect(cam,cam2, trt_yolo, conf_th=0.3, vis=vis)

    cam.release()
    cam2.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

