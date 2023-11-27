# Assuming each float is represented with 8 characters, and you want the third float
target_pressure = 5
target_temperature = 110

filename = "nitrogen.txt"


float_width = 8
minimum_pressure = 0
maximum_pressure = 20   
minimum_temperature = 100
maximum_temperature = 400

num_rows = maximum_temperature - minimum_temperature + 1
num_columns = maximum_pressure - minimum_pressure + 1

with open(filename, "r") as file:
    content = file.read()

print(num_rows, num_columns)
start_position = (target_temperature - minimum_temperature) * (
    num_columns * float_width
) + (target_pressure - minimum_pressure) * float_width
print(start_position)

end_position = start_position + float_width  # The end position of the bytes

desired_float_str = content[start_position:end_position]
desired_float = float(desired_float_str)

print("Float:", desired_float)

# Print raw bytes
raw_bytes = [ord(char) for char in desired_float_str]
print("Raw Bytes:", raw_bytes)
