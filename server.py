# server.py

import socket
import subprocess

from ping3 import ping, verbose_ping

def is_pingable(ip_address):
    # Ping the IP address once
    response = ping(ip_address)

    # Check if the response is not None (indicating a successful ping)
    return response is not None

# Define the server address (host and port)
host = '10.42.0.1'  # Replace with your server IP address
port = 12345

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
    mac_address = client_socket.recv(1024).decode('utf-8')
    print(f"Received MAC address: {mac_address}")

    # Check device status (You can replace this with your own logic)
    try:
        command = f"arp | grep {mac_address} | awk '{{print $1}}'"
        result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, text=True)
        if result != "":
            if is_pingable(result):
                response = "OK"
        else: response = "failed"
    except subprocess.CalledProcessError:
        response = check

    # Send the response back to the client
    client_socket.send(response.encode('utf-8'))

    # Close the connection
    client_socket.close()
