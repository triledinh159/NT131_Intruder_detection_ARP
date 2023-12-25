def read_mac_addresses():
    with open('mac_addresses', 'r') as file:
        lines = file.readlines()

    mac_addresses_dict = {}
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            label, mac_address = parts[0], parts[1]
            mac_addresses_dict[label] = mac_address
    return mac_addresses_dict

print(read_mac_addresses())

