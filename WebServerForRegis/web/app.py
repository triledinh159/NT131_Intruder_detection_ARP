from flask import Flask, render_template, request
import subprocess

app = Flask(__name__)

def is_already_registered(username, mac_address):
    with open('mac_addresses.txt', 'r') as file:
        for line in file:
            if mac_address in line:
                return True
    return False

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the user's MAC address using ARP
        user_ip = request.remote_addr
        command = f"arp | grep {user_ip} | awk '{{print $3}}'"
        result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, text=True).strip()

        # Get the submitted username from the form
        username = request.form.get('username')

        if username and result:
            # Check if the username or MAC address is already registered
            if is_already_registered(username, result):
                return "This Device is already registered. Thank you!"

            # Log the registered user and MAC address to the console
            print(f"Registered user: {username}, MAC address: {result}")

            # Store the username and MAC address in a text file
            with open('mac_addresses.txt', 'a') as file:
                file.write(f"{username}\n")
                file.write(f"{result}\n")

            # Display a simple response
            return f"Registration successful, {username}! Thank you!"

    # Render the HTML template
    return render_template('index.html')

if __name__ == '__main__':
    # Run the web server on all available network interfaces
    app.run(host='0.0.0.0', port=5000, debug=True)
