import os
import socket
from flask import Flask, render_template, request
import subprocess
import json
import boto3
import requests  # Import the requests library

AWS_ACCESS_KEY_ID = 'AKIAZ553U37JIDZID7UD'
AWS_SECRET_ACCESS_KEY = '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'
S3_BUCKET_NAME = 'zalouserdata'
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

def upload_to_s3(file_path, s3_key):
    try:
        s3_client.upload_file(file_path, S3_BUCKET_NAME, s3_key)
        print(f"File uploaded to S3: {s3_key}")
        return True
    except Exception as e:
        print(f"Failed to upload file to S3: {e}")
        return False

def remove_newline(input_string):
    if '\n' in input_string:
        # Remove the newline character
        output_string = input_string.replace('\n', '')
        return output_string
    else:
        return input_string

app = Flask(__name__)
def get_next_id():
    # Function to get the next available ID for the JSON file
    if not os.path.exists('ids.json'):
        with open('ids.json', 'w') as file:
            json.dump({'next_id': 1}, file)

    with open('ids.json', 'r') as file:
        data = json.load(file)
        next_id = data['next_id']

    return next_id


def is_already_registered(username, mac_address, user_id):
    # Check if the combination of username and MAC address is already registered
    file_path = f'{user_id}_{username}.json'
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
            if data['mac_address'] == mac_address:
                return True
    return False

def call_api(user_id, username):
    # Function to call the API
    api_url = f"https://dho61nl46f.execute-api.ap-southeast-1.amazonaws.com/test/userbasicdata?userID={user_id}&username={username}"
    response = requests.get(api_url)
    if response.status_code == 200:
        print(f"API call successful for user ID {user_id}, username {username}")
    else:
        print(f"API call failed for user ID {user_id}, username {username}. Status code: {response.status_code}")

@app.route('/', methods=['GET', 'POST'])
def index():
    port = 13579
    host = '10.42.0.1'
    # Create a socket object
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    if request.method == 'POST':
        user_ip = request.remote_addr
        client_socket.send(user_ip.encode('utf-8'))
        result = remove_newline(client_socket.recv(1024).decode('utf-8'))

        # Get the submitted username from the form
        username = request.form.get('username')

        if username and result:
            # Get the next available ID
            next_id = get_next_id()

            # Check if the username or MAC address is already registered
            if is_already_registered(username, result, next_id):
                return "This Device is already registered. Thank you!"

            # Log the registered user and MAC address to the console
            print(f"Registered user: {username}, MAC address: {result}")

            # Store the username and MAC address in a JSON file
            file_path = f'{next_id}_{username}.json'
            data = {'id': next_id, 'username': username, 'mac_address': result}
            with open(file_path, 'w') as file:
                json.dump(data, file)

            s3_key = f'{next_id}_{username}.json'
            if upload_to_s3(file_path, s3_key):
                print(f"File uploaded successfully to S3: {s3_key}")

                # Call the API after successful S3 file upload
                call_api(next_id, username)

            next_id += 1
            with open('ids.json', 'w') as file:
                json.dump({'next_id': next_id}, file)
            # Display a simple response
            return f"Registration successful, {username}! Thank you!"

    # Render the HTML template
    return render_template('index.html')

if __name__ == '__main__':
    # Run the web server on all available network interfaces
    app.run(host='0.0.0.0', port=5000, debug=True)
