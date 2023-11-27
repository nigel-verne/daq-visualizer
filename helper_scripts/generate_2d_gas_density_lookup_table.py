import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI as cp
from pathlib import Path


# =====================================================================================
# Lookup Table Parameters (adjust as necessary)
# =====================================================================================
# Set fluid name
fluid_name = "nitrogen"

# Set pressure parameters (in Bar)
P_min = 0
P_max = 20
dP = 1  # changing this value will cause errors in the PLC code.

# Set temperature parameters (in Kelvin)
T_min = 100  # don't set this below the boiling point or CoolProp will throw an error
T_max = 400
dT = 1  # changing this value will cause errors in the PLC code.


# =====================================================================================
# Code
# =====================================================================================

# Set numpy print settings for printing long arrays
np.set_printoptions(threshold=np.inf)

# Create the filepath, accounting for PLC file name length restrictions
path_variable = str(Path("helper_scripts") / fluid_name[:8])


# Density calculation function
def calculate_density(P, T):
    if P == 0:
        return 0.0
    else:
        return cp("D", "P", P * 10**5, "T", T, fluid_name)


# Vectorize the density calculation function
P_vect = np.arange(P_min, P_max + 1, dP)
T_vect = np.arange(T_min, T_max + 1, dT)
P_array, T_array = np.meshgrid(P_vect, T_vect)
vectorized_density = np.vectorize(calculate_density)

print("\nCalculating densities.  This may take a few minutes...")
density_array = vectorized_density(P_array, T_array)

# Create DataFrame
density_df = pd.DataFrame(density_array, index=T_vect, columns=P_vect)
density_df = density_df.astype(np.float64)  # convert all values to 8-byte floats

# flatten dataframe to a single array of floats
flattened_array = density_df.to_numpy().flatten().astype(np.float64)

# Export the temperature data to a fixed-width file
flattened_array.tofile(path_variable + ".bin")

# Save to CSV
density_df.to_csv((path_variable + ".csv"), float_format="%08.4f")


# Generate fixed-width string
fixed_width_str = density_df.to_string(
    index=False, col_space=8, header=False, float_format=lambda x: "{:08.4f}".format(x)
)

# print(fixed_width_str)

# Eliminate any extraneous characters
fixed_width_str = fixed_width_str.replace(" ", "")
fixed_width_str = fixed_width_str.replace("\n", "")

# Curtail filename to 8 characters to comply with Arduino naming requirements
fixed_width_filename = fluid_name[:8]

# Save the string to a fixed-width file.
with open((fixed_width_filename + ".txt"), "w") as file:
    file.write(fixed_width_str)


# Print dimensions
print("Lookup tables successfully generated.")
print(
    f"CSV file: {fluid_name}.csv\nFixed-width file (for PLC): {fixed_width_filename}.txt"
)


num_rows, num_columns = density_df.shape
print(f"Number of Rows: {num_rows}, Number of Columns: {num_columns}")
