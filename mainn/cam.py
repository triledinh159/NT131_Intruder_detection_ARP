import os
import argparse
import cv2
import numpy as np
import sys
import time
import gunicorn
from threading import Thread
import importlib.util
import socket
import boto3
import base64
import concurrent.futures
from botocore.exceptions import EndpointConnectionError, BotoCoreError
from flask import Flask, render_template, Response
from io import BytesIO
import json
import requests
time.sleep(10)
dirname = os.path.dirname(__file__)
client = boto3.client('rekognition')
#mac_address = "fa:a9:30:c6:8d:10"
directory = os.path.join(dirname, './static/faces')
alive_mac_addresses = set()

def call_api(user_id, username):
	# Function to call the API
	api_url = f""
	response = requests.get(api_url)
	if response.status_code == 200:
		print(f"API call successful for user ID {user_id}, username {username}")
	else:
		print(f"API call failed for user ID {user_id}, username {username}. Status code: {response.status_code}")

def alertNofi():
    api_url = f""
    response = requests.get(api_url)

def is_device_alive(mac_address):
	host = '10.42.0.1'
	port = 12345

	client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	client_socket.connect((host, port))

	client_socket.send(mac_address.encode('utf-8'))

	response = client_socket.recv(1024).decode('utf-8')

	client_socket.close()
	print(response)

	return response

def count_alive_devices(directory):
	alive_count = 0

	# Iterate through files in the directory
	for filename in os.listdir(directory):
		if filename.endswith(".json") and "_" in filename:
			file_path = os.path.join(directory, filename)

			# Read JSON file
			with open(file_path, 'r') as file:
				data = json.load(file)

			# Extract MAC address
			user_id = data.get("id")
			username = data.get('username')
			mac_address = data.get("mac_address")

			if mac_address:
				# Check if device is alive
				if is_device_alive(mac_address) == "OK":
					alive_count += 1
					response = is_device_alive(mac_address)
					print(mac_address)
					if mac_address not in alive_mac_addresses:
						alive_mac_addresses.add(mac_address)
				elif is_device_alive(mac_address) != "OK":
					if mac_address in alive_mac_addresses:
						alive_mac_addresses.discard(mac_address)
	return alive_count, is_device_alive(mac_address)

def recognizeFace(client, image, collection):
	face_matched = False
	try:
		with open(image, 'rb') as file:
			response = client.search_faces_by_image(CollectionId=collection, Image={'Bytes': file.read()}, MaxFaces=1, FaceMatchThreshold=85)

		if not response['FaceMatches']:
			face_matched = False
		else:
			face_matched = True
	except (EndpointConnectionError, BotoCoreError) as e:
		print(f"Error during face recognition: {e}")

	return face_matched, response

face_image_path = ''

def detectFace(frame, face_cascade):
	face_detected = False
	face_image_path = ''  # Initialize the variable

	# Detect faces
	faces = face_cascade.detectMultiScale(frame,
										  scaleFactor=1.1,
										  minNeighbors=5,
										  minSize=(30, 30),
										  flags=cv2.CASCADE_SCALE_IMAGE)

	x1, y1, w1, h1 = 0, 0, 0, 0

	print("Found {0} faces!".format(len(faces)))

	timestr = time.strftime("%Y%m%d-%H%M%S")

	if len(faces) > 0:
		face_detected = True
		x1, y1, w1, h1 = faces[0]

		# Create a mask for the face
		mask = np.zeros_like(frame)
		mask[y1:y1 + h1, x1:x1 + w1] = frame[y1:y1 + h1, x1:x1 + w1]

		# Extract the face region
		cropped_face = mask[y1:y1 + h1, x1:x1 + w1]

		# Save the cropped face
		face_image_path = '{0}/face_{1}.png'.format(directory, timestr)
		try:
			cv2.imwrite(face_image_path, cropped_face)
			print('Cropped face saved to %s' % face_image_path)
		except Exception as e:
			print('Error saving cropped face:', e)

	return face_detected, face_image_path, x1, y1

def collect_data(frame):
	parser = argparse.ArgumentParser(description='Facial recognition')
	parser.add_argument('--collection', help='Collection Name', default='NetSecShow-faces')
	parser.add_argument('--face_cascade', help='Path to face cascade.', default='./haarcascade_frontalface_alt2.xml')
	args = parser.parse_args()

	face_cascade_name = args.face_cascade
	face_cascade = cv2.CascadeClassifier()

	if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
		print('--(!)Error loading face cascade')
		exit(0)

	pre_name = ''
	if frame is None:
		print('--(!)NO IMAGE FOUND')

	face_detected, image, x, y = detectFace(frame, face_cascade)
	if not face_detected:
		pre_name = ''

	if face_detected:
		try:
			setdelname = pre_name
			with concurrent.futures.ThreadPoolExecutor() as executor:
				future = executor.submit(recognizeFace, client, image, args.collection)
				try:
					face_matched, response = future.result(timeout=5)  # Set a timeout value
					if face_matched:
						name = '%s' % (response['FaceMatches'][0]['Face']['ExternalImageId'])
						name += "=" * ((4 - len(name) % 4) % 4)
						base64_bytes = name.encode('utf-8')
						message_bytes = base64.b64decode(base64_bytes)
						message = message_bytes.decode('utf-8')
						print ('Identity matched ' + message +
							  ' with %r similarity and %r confidence...' % (round(response['FaceMatches'][0]['Similarity'], 1),
																			 round(response['FaceMatches'][0]['Face']['Confidence'], 2)))
						name = '%s' % (response['FaceMatches'][0]['Face']['ExternalImageId'])
						base64_bytes = name.encode('utf-8')
						message_bytes = base64.b64decode(base64_bytes)
						message = message_bytes.decode('utf-8')
						if message == pre_name:
							print('')
						else:
							print('Hello ' + message)
							pre_name = message
					else:
						alertNofi()
						print("FAILED TO DETECT")

				except concurrent.futures.TimeoutError:
					print("Recognition timed out")
				except Exception as e:
					print("Recognition failed:", e)

		except Exception:
			pass


class VideoStream:
	"""Camera object that controls video streaming from the Picamera"""
	def __init__(self,resolution=(640,480),framerate=30):
		# Initialize the PiCamera and the camera image stream
		self.stream = cv2.VideoCapture(0)
		ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
		ret = self.stream.set(3,resolution[0])
		ret = self.stream.set(4,resolution[1])
			
		# Read first frame from the stream
		(self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
		self.stopped = False

	def start(self):
	# Start the thread that reads frames from the video stream
		Thread(target=self.update,args=()).start()
		return self

	def update(self):
		# Keep looping indefinitely until the thread is stopped
		while True:
			# If the camera is stopped, stop the thread
			if self.stopped:
				# Close camera resources
				self.stream.release()
				return

			# Otherwise, grab the next frame from the stream
			(self.grabbed, self.frame) = self.stream.read()

	def read(self):
	# Return the most recent frame
		return self.frame

	def stop(self):
	# Indicate that the camera and thread should be stopped
		self.stopped = True

image_filename = ''


def main():
	dir_json = "/home/tri/Desktop/doan/web"  # Replace with your actual directory path
	output_file = "/home/tri/Desktop/doan/findevices/alive_count.txt"
	# Define and parse input arguments
	parser = argparse.ArgumentParser()
	# parser.add_argument('--hostip', help='Access Point default IP',
	# 					default='10.42.0.1')
	parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
						default="./")
	parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
						default='detect.tflite')
	parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
						default='labelmap.txt')
	parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
						default=0.6)
	parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
						default='1280x720')
	parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
						action='store_true')

	args = parser.parse_args()
	# host = args.hostip
	MODEL_NAME = args.modeldir
	GRAPH_NAME = args.graph
	LABELMAP_NAME = args.labels
	min_conf_threshold = float(args.threshold)
	resW, resH = args.resolution.split('x')
	imW, imH = int(resW), int(resH)
	use_TPU = args.edgetpu
	num = 0

	# port = 12345

	# # Create a socket object
	# client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

	# # Connect to the server
	# client_socket.connect((host, port))
	# Import TensorFlow libraries
	# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
	# If using Coral Edge TPU, import the load_delegate library
	pkg = importlib.util.find_spec('tflite_runtime')
	if pkg:
		from tflite_runtime.interpreter import Interpreter
		if use_TPU:
			from tflite_runtime.interpreter import load_delegate
	else:
		from tensorflow.lite.python.interpreter import Interpreter
		if use_TPU:
			from tensorflow.lite.python.interpreter import load_delegate

	# If using Edge TPU, assign filename for Edge TPU model
	if use_TPU:
		# If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
		if (GRAPH_NAME == 'detect.tflite'):
			GRAPH_NAME = 'edgetpu.tflite'       

	# Get path to current working directory
	CWD_PATH = os.getcwd()

	# Path to .tflite file, which contains the model that is used for object detection
	PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

	# Path to label map file
	PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

	# Load the label map
	with open(PATH_TO_LABELS, 'r') as f:
		labels = [line.strip() for line in f.readlines()]

	# Have to do a weird fix for label map if using the COCO "starter model" from
	# https://www.tensorflow.org/lite/models/object_detection/overview
	# First label is '???', which has to be removed.
	if labels[0] == '???':
		del(labels[0])

	# Load the Tensorflow Lite model.
	# If using Edge TPU, use special load_delegate argument
	if use_TPU:
		interpreter = Interpreter(model_path=PATH_TO_CKPT,
								experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
		print(PATH_TO_CKPT)
	else:
		interpreter = Interpreter(model_path=PATH_TO_CKPT)

	interpreter.allocate_tensors()

	# Get model details
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
	height = input_details[0]['shape'][1]
	width = input_details[0]['shape'][2]

	floating_model = (input_details[0]['dtype'] == np.float32)

	input_mean = 127.5
	input_std = 127.5

	# Initialize frame rate calculation
	frame_rate_calc = 1
	freq = cv2.getTickFrequency()

	# Initialize video stream

	time.sleep(1)
	videostream = VideoStream(resolution=(1280,720),framerate=30).start()
	# Create window
	#cv2.namedWindow('Object detector', cv2.WINDOW_NORMAL)

	#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
	while True:
		
		# Start timer (for calculating frame rate)
		t1 = cv2.getTickCount()

		# Grab frame from video stream
		frame1 = videostream.read()
		# Acquire frame and resize to expected shape [1xHxWx3]
		frame = frame1.copy()
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame_resized = cv2.resize(frame_rgb, (width, height))
		input_data = np.expand_dims(frame_resized, axis=0)

		# Normalize pixel values if using a floating model (i.e. if model is non-quantized)
		if floating_model:
			input_data = (np.float32(input_data) - input_mean) / input_std

		# Perform the actual detection by running the model with the image as input
		interpreter.set_tensor(input_details[0]['index'],input_data)
		interpreter.invoke()

		# Retrieve detection results
		boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
		classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
		scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
		#num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

		# Loop over all detections and draw detection box if confidence is above minimum threshold
		for i in range(len(scores)):
			if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

				# Get bounding box coordinates and draw box
				# Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
				ymin = int(max(1,(boxes[i][0] * imH)))
				xmin = int(max(1,(boxes[i][1] * imW)))
				ymax = int(min(imH,(boxes[i][2] * imH)))
				xmax = int(min(imW,(boxes[i][3] * imW)))
				
				cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
				
				# Draw label
				object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
				if object_name != "person":
					continue
				label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
				labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
				label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
				cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
				cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
				# client_socket.send(mac_address.encode('utf-8'))
				# Draw circle in center
				xcenter = xmin + (int(round((xmax - xmin) / 2)))
				ycenter = ymin + (int(round((ymax - ymin) / 2)))
				cv2.circle(frame, (xcenter, ycenter), 5, (0,0,255), thickness=-1)
				alive_count, response = count_alive_devices(dir_json)
				#print(f"Server response: {response}")
				if alive_count > num:
					num = alive_count
					print("OK")
					time.sleep(5)

				else:
					# Capture and save an image
					image_filename = f"./static/human/person_captured_{time.strftime('%Y%m%d%H%M%S')}.jpg"
					collect_data(frame)
					cv2.imwrite(image_filename, frame)

				# client_socket.close()
				# client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
				# client_socket.connect((host, port))
				# Print info
				#print('Object ' + str(i) + ': ' + object_name + ' at (' + str(xcenter) + ', ' + str(ycenter) + ')')

		# Draw framerate in corner of frame
		#cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

		# All the results have been drawn on the frame, so it's time to display it.
		#cv2.imshow('Project_SHID', frame)

		# Calculate framerate
		t2 = cv2.getTickCount()
		time1 = (t2-t1)/freq
		frame_rate_calc= 1/time1

		# Press 'q' to quit
		if cv2.waitKey(1) == ord('q'):
			client_socket.close()
			break

	# Clean up
	cv2.destroyAllWindows()
	videostream.stop()
	

if __name__ == '__main__':
	main()
	
