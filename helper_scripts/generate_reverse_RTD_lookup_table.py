import pandas as pd
import numpy as np
import os

# PT111 Data, supplied by Lakeshore Cryogenics
temperature_data = [
    28.000,
    30.000,
    32.000,
    36.000,
    38.000,
    40.000,
    42.000,
    46.000,
    52.000,
    58.000,
    65.000,
    75.000,
    85.000,
    105.000,
    140.000,
    180.000,
    210.000,
    270.000,
    315.000,
    355.000,
    400.000,
    445.000,
    490.000,
    535.000,
    585.000,
    630.000,
    675.000,
    715.000,
    760.000,
    800.000,
    810.000,
]

resistance_data = [
    3.40500,
    3.82000,
    4.23500,
    5.14600,
    5.65000,
    6.17000,
    6.72600,
    7.90900,
    9.92400,
    12.1800,
    15.0150,
    19.2230,
    23.5250,
    32.0810,
    46.6480,
    62.9800,
    75.0440,
    98.7840,
    116.270,
    131.616,
    148.652,
    165.466,
    182.035,
    198.386,
    216.256,
    232.106,
    247.712,
    261.391,
    276.566,
    289.830,
    293.146,
]


def linear_interpolation(x, x1, y1, x2, y2):
    # Linear interpolation formula
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1)


def generate_table(
    start_resistance, end_resistance, interval, temperature_data, resistance_data
):
    # Create lists to store the interpolated values
    resistance_list = []
    temperature_list = []

    # Perform linear interpolation for each desired resistance value
    current_resistance = start_resistance
    while current_resistance <= end_resistance:
        # Find the two closest resistance values in the original data
        for i in range(len(resistance_data) - 1):
            if resistance_data[i] <= current_resistance <= resistance_data[i + 1]:
                x1, y1 = resistance_data[i], temperature_data[i]
                x2, y2 = resistance_data[i + 1], temperature_data[i + 1]

                # Perform linear interpolation
                interpolated_temperature = linear_interpolation(
                    current_resistance, x1, y1, x2, y2
                )

                # round resistance to account for floating-point errors
                rounded_resistance = round(current_resistance, 1)

                # round temperature
                rounded_temperature = round(interpolated_temperature, 4)

                # Add the rounded values to the lists
                resistance_list.append(rounded_resistance)
                temperature_list.append(rounded_temperature)
                break

        current_resistance += interval

    return resistance_list, temperature_list


# Define the desired range and interval
start_resistance = 3.5
end_resistance = 193.3
interval = 0.1

# Generate the interpolated lists
resistance_list, temperature_list = generate_table(
    start_resistance, end_resistance, interval, temperature_data, resistance_data
)

# Create a DataFrame from the lists
df = pd.DataFrame(
    {"Resistance (ohms)": resistance_list, "Temperature (K)": temperature_list}
)

# Print the DataFrame
print(df)

# Export Dataframe to CSV
df.to_csv("helper_scripts/PT111.csv", index=False)

# Create a NumPy array from the temperature list
temperature_array = np.array(temperature_list, dtype=np.float64)

# Export the temperature data to a fixed-width file
temperature_array.tofile("helper_scripts/PT111.bin")

# Print the size of the binary file
binary_file_path = "helper_scripts/PT111.bin"
if os.path.exists(binary_file_path):
    binary_file_size = os.path.getsize(binary_file_path)
    print(f"The size of the binary file is: {binary_file_size} bytes")
else:
    print("Binary file not found.")
