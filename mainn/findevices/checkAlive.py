import os
import json
import socket
import time
import boto3
import requests

# Set to keep track of alive MAC addresses
alive_mac_addresses = set()

def call_api(user_id, username):
    # Function to call the API
    api_url = f"https://dho61nl46f.execute-api.ap-southeast-1.amazonaws.com/test/userbasicdata?userID={user_id}&username={username}"
    response = requests.get(api_url)
    if response.status_code == 200:
        print(f"API call user ID {user_id}, username {username}")
    else:
        print(f"API call failed for user ID {user_id}, username {username}. Status code: {response.status_code}")



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
                    if mac_address not in alive_mac_addresses:
                        call_api(user_id, username)
                        alive_mac_addresses.add(mac_address)
                elif is_device_alive(mac_address) != "OK":
                    if mac_address in alive_mac_addresses:
                        call_api(user_id, username)
                        alive_mac_addresses.discard(mac_address)

    return alive_count

def main():
    directory = "/home/tri/Desktop/doan/web"  # Replace with your actual directory path
    output_file = "alive_count.txt"

    while True:
        # Count alive devices
        alive_count = count_alive_devices(directory)

        # Store the count in a text file
        with open(output_file, 'w') as file:
            file.write(f"{alive_count}\n")
            file.write("\n".join(alive_mac_addresses))

        print(f"Number of alive devices: {alive_count}")
        print("Alive MAC addresses:", alive_mac_addresses)
        
        # Wait for 15 seconds before checking again
        time.sleep(10)

if __name__ == "__main__":
    main()
