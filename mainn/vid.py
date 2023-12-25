import os
import cv2
from threading import Thread
import time
from flask import jsonify, url_for
from flask import Flask, render_template, Response

app = Flask(__name__)

class VideoStream:
	"""Camera object that controls video streaming from the Picamera"""
	def __init__(self,resolution=(640,480),framerate=30):
		# Initialize the PiCamera and the camera image stream
		self.stream = cv2.VideoCapture(1)
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

# Đường dẫn thư mục chứa hình ảnh
image_folder1 = "./static/faces"
image_folder2 = "./static/human"
latest_image1 = None  # Biến lưu trữ tên hình ảnh mới nhất từ folder1
latest_image2 = None  # Biến lưu trữ tên hình ảnh mới nhất từ folder2
latest_image_time1 = None  # Thời gian hình ảnh mới nhất từ folder1 được cập nhật
latest_image_time2 = None  # Thời gian hình ảnh mới nhất từ folder2 được cập nhật

# Hàm đọc hình ảnh mới nhất từ thư mục
def update_latest_images():
    global latest_image1, latest_image_time1, latest_image2, latest_image_time2
    while True:
        images1 = [f for f in os.listdir(image_folder1) if f.endswith(('.jpg', '.jpeg', '.png'))]
        images2 = [f for f in os.listdir(image_folder2) if f.endswith(('.jpg', '.jpeg', '.png'))]

        if images1:
            latest_image1 = max(images1, key=lambda x: os.path.getctime(os.path.join(image_folder1, x)))
            latest_image_time1 = os.path.getctime(os.path.join(image_folder1, latest_image1))
        else:
            latest_image1 = None
            latest_image_time1 = None

        if images2:
            latest_image2 = max(images2, key=lambda x: os.path.getctime(os.path.join(image_folder2, x)))
            latest_image_time2 = os.path.getctime(os.path.join(image_folder2, latest_image2))
        else:
            latest_image2 = None
            latest_image_time2 = None

        time.sleep(1)  # Ngủ 1 giây để giảm gánh nặng của vòng lặp  

# Bắt đầu thread để cập nhật hình ảnh mới nhất
update_thread = Thread(target=update_latest_images)
update_thread.daemon = True
update_thread.start()

def generate_frames():
    cam = VideoStream(resolution=(1280,720),framerate=30).start()
    while True:
        # Đọc frame từ webcam
        frame = cam.read()

        # Chuyển đổi frame thành định dạng JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Gửi frame đến client
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cam.stop() 

@app.route('/')
def index():
    return render_template('index.html', latest_image1=latest_image1, latest_image_time1=latest_image_time1,
                           latest_image2=latest_image2, latest_image_time2=latest_image_time2)

@app.route('/latest_image1')
def latest_image_info1():
    return jsonify({
        'latest_image_src': url_for('static', filename='faces/' + latest_image1),
        'latest_image_time': latest_image_time1
    })

@app.route('/latest_image2')
def latest_image_info2():
    return jsonify({
        'latest_image_src': url_for('static', filename='human/' + latest_image2),
        'latest_image_time': latest_image_time2
    })

@app.route('/video_feed')
def video_feed():
    # Trả về video stream
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5100, debug=True)
