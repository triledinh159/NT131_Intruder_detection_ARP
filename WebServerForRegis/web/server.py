# server.py

import socket
import subprocess
def remove_newline(input_string):
    if '\n' in input_string:
        # Remove the newline character
        output_string = input_string.replace('\n', '')
        return output_string
    else:
        return input_string

# Define the server address (host and port)
host = '10.42.0.1'  # Replace with your server IP address
port = 13579

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the address
server_socket.bind((host, port))

# Listen for incoming connections (1 connection at a time)
server_socket.listen(1)

print(f"Server listening on {host}:{port}")
check = "failed"
response =""
result = ""
while True:
	# Accept a connection
	client_socket, client_address = server_socket.accept()
	print(f"Accepted connection from {client_address}")

	# Receive MAC address from the client
	ip_address = client_socket.recv(1024).decode('utf-8')
	print(f"Received IP address: {ip_address}")

	# Check device status (You can replace this with your own logic)
	try:
		command = f"arp | grep {ip_address} | awk '{{print $3}}'"
		result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, text=True)
		if result != "":
			response = result
	except subprocess.CalledProcessError:
		response = "chekc"

	# Send the response back to the client
	client_socket.send(response.encode('utf-8'))

	# Close the connection
	client_socket.close()
