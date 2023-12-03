#from picamera import PiCamera

#from espeak import espeak

from PIL import ImageFont, ImageDraw, Image

import cv2

import argparse

import boto3

import os

from threading import Thread

import numpy as np  


import random
import time

from paho.mqtt import client as mqtt_client

import base64
broker = 'broker.emqx.io'
port = 1883
topic = "python/mqtt"
# generate client ID with pub prefix randomly
client_id = f'python-mqtt-{random.randint(0, 1000)}'
# username = 'emqx'
# password = 'public'

def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    client.username_pw_set('emqx', 'public')
    client.on_connect = on_connect
    client.connect(broker, port)
    return client


def publish(client,name):
    msg = f"{name}"
    result = client.publish(topic, msg)
    # result: [0, 1]
    status = result[0]
    if status == 0:
        print(f"Sent")
    else:
        print(f"Failed to send message to topic {topic}")




def recognizeFace(client,image,collection):

    face_matched = False

    with open(image, 'rb') as file:

        response = client.search_faces_by_image(CollectionId=collection, Image={'Bytes': file.read()}, MaxFaces=1, FaceMatchThreshold=85)

        if (not response['FaceMatches']):

            face_matched = False

        else:

            face_matched = True

    return face_matched, response



def detectFace(frame,face_cascade):	

    face_detected = False

    #Detect faces

    faces = face_cascade.detectMultiScale(frame,

        scaleFactor=1.1,

        minNeighbors=5,

        minSize=(30, 30),

        flags = cv2.CASCADE_SCALE_IMAGE)
    x1 = 0
    y1 = 0
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0), 2)
        x1,y1 = x,y
    print("Found {0} faces!".format(len(faces)))

    timestr = time.strftime("%Y%m%d-%H%M%S")

    image = '{0}/image_{1}.png'.format(directory, timestr)

    if len(faces) > 0 :

        face_detected = True

        cv2.imwrite(image,frame) 

        print ('Your image was saved to %s' % image)



    return face_detected, image, x1, y1

# def detectFaceforVis(frame2,face_cascade):	

#     #Detect faces
    
#     gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

#     faces = face_cascade.detectMultiScale(gray,1.1,2)


#     for (x,y,w,h) in faces:
#         cv2.rectangle(frame2, (x,y), (x+w,y+h),(0,255,0), 2)
#     cv2.imshow('Visualizing...',frame2)
#     if cv2.waitKey(20) & 0xFF == ord('q'): 
#         return

def collect_data():	



    #get args

    parser = argparse.ArgumentParser(description='Facial recognition')

    parser.add_argument('--collection', help='Collection Name', default='NetSecShow-faces')

    parser.add_argument('--face_cascade', help='Path to face cascade.', default='./haarcascade_frontalface_alt2.xml')

    args = parser.parse_args()

    #intialize opencv2 face detection

    face_cascade_name = args.face_cascade

    face_cascade = cv2.CascadeClassifier()

    #Load the cascades

    if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):

        print('--(!)Error loading face cascade')

        exit(0)

    cam = cv2.VideoCapture(0)

    #Read the video stream

    #initialize rekognition sdk

    client = boto3.client('rekognition')

    pre_name = ''
    while True:

        #calling read() twice as a workaround to clear the buffer.

        ret,frame = cam.read()
        if frame is None:
            #videostart.stop()

            print('--(!)1 No captured frame -- Break!')

            break

        face_detected, image, x, y = detectFace(frame,face_cascade)
        if (not face_detected):
            pre_name = ''
        if (face_detected):
            try:
                setdelname = pre_name
                face_matched, response = recognizeFace(client, image , args.collection)

                if (face_matched):
                    name = '%s' % (response['FaceMatches'][0]['Face']['ExternalImageId'])
                    name += "=" * ((4 - len(name) % 4) % 4)
                    base64_bytes = name.encode('utf-8')
                    message_bytes = base64.b64decode(base64_bytes)
                    message = message_bytes.decode('utf-8')
                    print ('Identity matched '+ message +' with %r similarity and %r confidence...' % (round(response['FaceMatches'][0]['Similarity'], 1), round(response['FaceMatches'][0]['Face']['Confidence'], 2)))

                    #espeak.synth('Hello %s! What is my purpose?' % (response['FaceMatches'][0]['Face']['ExternalImageId']))

                    name = '%s' % (response['FaceMatches'][0]['Face']['ExternalImageId'])
                    base64_bytes = name.encode('utf-8')
                    message_bytes = base64.b64decode(base64_bytes)
                    message = message_bytes.decode('utf-8')
                    if (message == pre_name): 
                        print('')
                    else:
                        if (setdelname != ''):
                            cv2.destroyWindow('FACE')
                        print ('Hello ' + message)
                        fontpath = "./clear-sans/ClearSans-Regular.ttf"
                        font = ImageFont.truetype(fontpath, 32)
                        img_pil = Image.fromarray(frame)
                        draw = ImageDraw.Draw(img_pil)
                        draw.text((x, y-40),  message, font = font, fill = (0,255,0,0))
                        frame = np.array(img_pil)

                        #ft.putText(img=frame, text = message ,org = (x, y-10), fontHeight=60,color = (36,255,12),thickness= 3, line_type=cv2.LINE_AA, bottomLeftOrigin=True)
                        cv2.imshow('FACE', frame)
                        pre_name = message
                        #publish(client1,name)
                    


                else:

                    cv2.putText(frame, 'UNKNOWN' , (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                    cv2.imshow('FACE', frame)

                    #espeak.synth('Unknown human detected. Who are you stranger?')
                
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break
                
            except Exception:
                pass
        if cv2.waitKey(20) & 0xFF == ord('q'):

            break

    # When everything done, release the capture

    cv2.destroyAllWindows()

def show_video():
    parser = argparse.ArgumentParser(description='Facial recognition')
    parser.add_argument('--face_cascade', help='Path to face cascade.', default='./haarcascade_frontalface_default.xml')

    args = parser.parse_args()

    #intialize opencv2 face detection

    face_cascade_name = args.face_cascade

    face_cascade = cv2.CascadeClassifier()

    #Load the cascades

    if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):

        print('--(!)Error loading face cascade')

        exit(0)

    cam2 = cv2.VideoCapture(1)
    while True:
        ret,frame2 = cam2.read()
        # detectFaceforVis(frame2,face_cascade)
        cv2.imshow('Visualizing...',frame2)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()



dirname = os.path.dirname(__file__)

directory = os.path.join(dirname, 'faces')

def run():
    # client = connect_mqtt()
    # client.loop_start()


    t1 = Thread(target=collect_data)
    t2 = Thread(target=show_video)


    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # main(1)

if __name__ == '__main__':
    run()


