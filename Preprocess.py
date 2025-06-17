# Configuration stored in a dictionary
config = {
    "first_run": False,
    "mesh_top": 29.0,  # mm
    "height": 59.2,    # mm,
    "boundary_nodes": {
        "dx": 0.2,
        "dy": 0.2,
        "left": -10.2 + 0.2,  # calculated value
        "right": 12.3 - 0.2,  # calculated value
        "top": 29.0 - 0.2,    # calculated value
        "bottom": 29.0 - 59.2 + 0.2  # calculated value
    },
    "small_specimen_region": False,
    "equilateral_elements": True,
    "plots": True,
    "export": False,
    "coord_part": True,
    "export_coord": False
}

# Accessing values example
print(config["mesh_top"])  # Output: 29.0
print(config["boundary_nodes"]["left"])  # Output: -10.0

#-----------------------------------------------------------------------------------------------------------------------------------------
# Import data
#----------------------------------------------------------------------------------------------------------------------------------------
# Import necessary libraries
import sys
import pandas as pd
import numpy as np
import os
import csv
import matplotlib.pyplot as plt

### Experiment number
experiment_number = 5 # Change this to the desired experiment number (1, 2, 3, 4, or 5)

# Results folder
# Define base directory for results
base_dir = 'results_preprocessing'

# Check if the base directory exists, if not, create it
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# Define experiment-specific directory
experiment_dir = os.path.join(base_dir, f'experiment_{experiment_number}')

# Check if the experiment-specific directory exists, if not, create it
if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir)

# Initialize variables
force = None
time = None
nodes = []

# Define base file names
file_name_prefix = "Flächenkomponente 1_"
file_name_suffix = ".000 s.csv"

# Data import based on experiment number
#-----------------------------------------------------------------------------------------------------------------------------------------
# Experiment 1
#-----------------------------------------------------------------------------------------------------------------------------------------
if experiment_number == 1:
    # Load force data using csv
    force_path = "C:\\Users\\undamatl\\Nextcloud2\\EUCLID-viscoelasticity-TD\\experiments\\1\\EUCLID_lve_pp_rec_1.csv"
    with open(force_path, 'r', encoding='latin1') as file:
        reader = csv.reader(file, delimiter=';')
        next(reader)  # Skip the header row
        force_data = list(reader)
    force = np.array([float(row[2]) for row in force_data if row[2] != 'Kraft 10kN [kN]'])  # Extracting force data
    time = np.array([float(row[8]) for row in force_data if row[8] != 'Zeit [s]'])  # Extracting time data

    # Initialize nodes list
    nodes = [None] * len(time)  
    file_name_prefix = "Flächenkomponente 1_"
    file_name_suffix = ".000 s.csv"
    folder_x = "C:\\Users\\undamatl\\Nextcloud2\\EUCLID-viscoelasticity-TD\\experiments\\1\\EUCLID_lve_pp_rec_1_dX\\"
    folder_y = "C:\\Users\\undamatl\\Nextcloud2\\EUCLID-viscoelasticity-TD\\experiments\\1\\EUCLID_lve_pp_rec_1_dY\\"

    for t in range(len(time)):
        file_path_x = os.path.join(folder_x, f"{file_name_prefix}{t}{file_name_suffix}")
        file_path_y = os.path.join(folder_y, f"{file_name_prefix}{t}{file_name_suffix}")

        with open(file_path_x, 'r', encoding='utf-8') as file_x, open(file_path_y, 'r', encoding='utf-8') as file_y:
            reader_x = csv.reader(file_x, delimiter=',')
            reader_y = csv.reader(file_y, delimiter=',')
            
            rows_x = list(reader_x)
            rows_y = list(reader_y)
            
            start_index_x = next(i for i, row in enumerate(rows_x) if row and row[0].startswith('id'))
            start_index_y = next(i for i, row in enumerate(rows_y) if row and row[0].startswith('id'))
            
            data_x = np.array([row[0:3] + [row[4]] for row in rows_x[start_index_x + 1:] if len(row) >= 5], dtype=float)
            data_y = np.array([row[4] for row in rows_y[start_index_y + 1:] if len(row) >= 5], dtype=float)
            # Ensure data_x and data_y are not empty	
            if data_x.size == 0 or data_y.size == 0:
                print(f"Warning: Missing data at time step {t}.")
                continue

            if data_x.shape[0] != data_y.shape[0]:
                print(f"Error: Mismatched data rows at time step {t}.")
                continue


            combined_data = np.hstack((data_x, data_y[:, None]))  # Add displacement_y as a new column
            nodes[t] = combined_data

    # Calculate time differences
    dt = np.zeros(len(time) - 1)
    for t in range(1, len(time)):
        dt[t - 1] = time[t] - time[t - 1]

#-----------------------------------------------------------------------------------------------------------------------------------------
# Experiment 2
#-----------------------------------------------------------------------------------------------------------------------------------------
if experiment_number == 2:
    # Load force data using csv
    force_path = "C:\\Users\\undamatl\\Nextcloud2\\EUCLID-viscoelasticity-TD\\experiments\\2\\20241213_EUCLID_lve_pa6_rec_LaserCut_Sand_1Hz.csv"
    with open(force_path, 'r', encoding='latin1') as file:
        reader = csv.reader(file, delimiter=';')
        # Skip the header row (adjust number of skips if there are multiple header lines)
        next(reader)  # Skip one header row. Add more next(reader) if there are more header lines.
        force_data = list(reader)
    force = np.array([float(row[2]) for row in force_data if row[2] != 'Kraft 10kN [kN]'])  # Extracting force data, skip non-numeric
    time = np.array([float(row[8]) for row in force_data if row[8] != 'Zeit [s]'])  # Extracting time data, skip non-numeric

    # Initialize nodes list
    nodes = [None] * len(time)  # Preallocate list for nodes

    file_name_prefix = "Flächenkomponente 1_"
    file_name_suffix = ".000 s.csv"
    folder = "C:\\Users\\undamatl\\Nextcloud2\\EUCLID-viscoelasticity-TD\\experiments\\2\\APA6LCRectSand\\"

    for t in range(len(time)):
        file_path = os.path.join(folder, f"{file_name_prefix}{t}{file_name_suffix}")
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=';')
            rows = list(reader)
            
            # Finding the start of the actual data
            start_index = next(i for i, row in enumerate(rows) if row and row[0].startswith('id'))
            
            # Extracting data while skipping the 'z' column
            data = np.array([row[0:3] + row[4:6] for row in rows[start_index + 1:] if len(row) >= 6], dtype=float)
            nodes[t] = data

    # Calculate time differences
    dt = np.zeros(len(time) - 1)
    for t in range(1, len(time)):
        dt[t - 1] = time[t] - time[t - 1]
    

#-----------------------------------------------------------------------------------------------------------------------------------------
# Experiment 3  
#-----------------------------------------------------------------------------------------------------------------------------------------
if experiment_number == 3:
    # Load force data using csv
    force_path = "C:\\Users\\undamatl\\Nextcloud2\\EUCLID-viscoelasticity-TD\\experiments\\3\\PA6_WJ_ellipsoid.csv"
    with open(force_path, 'r', encoding='latin1') as file:
        reader = csv.reader(file, delimiter=';')
        next(reader)  # Skip the header row
        force_data = list(reader)
    force = np.array([float(row[2]) for row in force_data if row[2] != 'Kraft 10kN [kN]'])  # Extracting force data
    time = np.array([float(row[8]) for row in force_data if row[8] != 'Zeit [s]'])  # Extracting time data

    # Initialize nodes list
    nodes = [None] * len(time)  # Preallocate list for nodes
    file_name_prefix = "Flächenkomponente 1_"
    file_name_suffix = ".000 s.csv"
    folder = "C:\\Users\\undamatl\\Nextcloud2\\EUCLID-viscoelasticity-TD\\experiments\\3\\"

    for t in range(len(time)):
        file_path = os.path.join(folder, f"{file_name_prefix}{t}{file_name_suffix}")
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=';')
            rows = list(reader)
            start_index = next(i for i, row in enumerate(rows) if row and row[0].startswith('id'))
            data = np.array([row[0:3] + row[4:6] for row in rows[start_index + 1:] if len(row) >= 6], dtype=float)
            nodes[t] = data

    # Calculate time differences
    dt = np.zeros(len(time) - 1)
    for t in range(1, len(time)):
        dt[t - 1] = time[t] - time[t - 1]
    

#-----------------------------------------------------------------------------------------------------------------------------------------
# Experiment 4  
#-----------------------------------------------------------------------------------------------------------------------------------------
if experiment_number == 4:
    # Load force data using csv
    force_path = "C:\\Users\\undamatl\\Nextcloud2\\EUCLID-viscoelasticity-TD\\experiments\\4\\PA6_WJ_three_holes.csv"
    with open(force_path, 'r', encoding='latin1') as file:
        reader = csv.reader(file, delimiter=';')
        next(reader)  # Skip the header row
        force_data = list(reader)
    force = np.array([float(row[2]) for row in force_data if row[2] != 'Kraft 10kN [kN]'])  # Extracting force data
    time = np.array([float(row[8]) for row in force_data if row[8] != 'Zeit [s]'])  # Extracting time data

    # Initialize nodes list
    nodes = [None] * len(time)  # Preallocate list for nodes
    file_name_prefix = "Flächenkomponente 1_"
    file_name_suffix = ".000 s.csv"
    folder = "C:\\Users\\undamatl\\Nextcloud2\\EUCLID-viscoelasticity-TD\\experiments\\4\\"

    for t in range(len(time)):
        file_path = os.path.join(folder, f"{file_name_prefix}{t}{file_name_suffix}")
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=';')
            rows = list(reader)
            start_index = next(i for i, row in enumerate(rows) if row and row[0].startswith('id'))
            data = np.array([row[0:3] + row[4:6] for row in rows[start_index + 1:] if len(row) >= 6], dtype=float)
            nodes[t] = data

    # Calculate time differences
    dt = np.zeros(len(time) - 1)
    for t in range(1, len(time)):
        dt[t - 1] = time[t] - time[t - 1]
    
#-----------------------------------------------------------------------------------------------------------------------------------------
# Experiment 5
#-----------------------------------------------------------------------------------------------------------------------------------------
if experiment_number == 5:
    # Load force data using csv
    force_path = "C:\\Users\\undamatl\\Nextcloud2\\EUCLID-viscoelasticity-TD\\experiments\\5\\PA6_WJ_two_bells.csv"
    with open(force_path, 'r', encoding='latin1') as file:
        reader = csv.reader(file, delimiter=';')
        next(reader)  # Skip the header row
        force_data = list(reader)
    force = np.array([float(row[2]) for row in force_data if row[2] != 'Kraft 10kN [kN]'])  # Extracting force data
    time = np.array([float(row[8]) for row in force_data if row[8] != 'Zeit [s]'])  # Extracting time data

    # Initialize nodes list
    nodes = [None] * len(time)  # Preallocate list for nodes
    file_name_prefix = "Flächenkomponente 1_"
    file_name_suffix = ".000 s.csv"
    folder = "C:\\Users\\undamatl\\Nextcloud2\\EUCLID-viscoelasticity-TD\\experiments\\5\\"

    for t in range(len(time)):
        file_path = os.path.join(folder, f"{file_name_prefix}{t}{file_name_suffix}")
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=';')
            rows = list(reader)
            start_index = next(i for i, row in enumerate(rows) if row and row[0].startswith('id'))
            data = np.array([row[0:3] + row[4:6] for row in rows[start_index + 1:] if len(row) >= 6], dtype=float)
            nodes[t] = data

    # Calculate time differences
    dt = np.zeros(len(time) - 1)
    for t in range(1, len(time)):
        dt[t - 1] = time[t] - time[t - 1]
    
    

# Check the loaded data
print("Force data shape:", force.shape)
print("Time data shape:", time.shape)
print("Number of nodes:", len(nodes))
# Print samples of the data
print("Force data sample (first 5):", force[:5])
print("Time data sample (first 5):", time[:5])
print("Nodes sample (first 5 entries of the first node):", nodes[0][:5] if nodes[0] is not None else "No data")

#-----------------------------------------------------------------------------------------------------------------------------------------
#Output files
#-----------------------------------------------------------------------------------------------------------------------------------------
# Save each array individually in binary format in the specific experiment directory
np.save(os.path.join(experiment_dir, 'force_data.npy'), force)
np.save(os.path.join(experiment_dir, 'time_data.npy'), time)
np.save(os.path.join(experiment_dir, 'nodes_data.npy'), np.array(nodes, dtype=object))

# Optional data save complete data
np.savez(os.path.join(experiment_dir, 'experiment_data.npz'), force=force, time=time, nodes=np.array(nodes, dtype=object))

#------------------------------------------------------------
# Data processing 
#------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

# Set this to True when you want to run the plotting section
first_run = True

if first_run:
    # nodes[0][:, 1] -> X coordinates
    # nodes[0][:, 2] -> Y coordinates
    # Check if nodes[0] is not None and has data
    if nodes[0] is not None:
        reference = nodes[0]  # Reference is the initial node data
        
        plt.figure(figsize=(10, 5))  # Create a figure with a specified size
        plt.plot(reference[:, 1], reference[:, 2], '.b', label='dic') # Plot all points as blue dots
        plt.plot(reference[0, 1], reference[0, 2], 'or', label='first') # Plot the first point in red dot
        plt.plot(reference[1, 1], reference[1, 2], 'xk', label='second') # Plot the second point in black 'x'
        plt.plot(reference[-1, 1], reference[-1, 2], 'ob', label='last') # Plot the last point in a blue circle
        # Adding labels for axes
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend() # Add a legend to the plot
        plt.title(f'Initial Node Data Visualization for Experiment {experiment_number}') # Add a title
        # Define the plot filename dynamically based on the experiment number
        plot_filename = os.path.join(experiment_dir, f'node_data_visualization_experiment_{experiment_number}.png') 
        # Save the plot to the file
        plt.savefig(plot_filename)
        plt.show()  # Show the plot in a window
        plt.close()  # Close the plot to free up memory
        
    else:
        print("Reference data is not available.")

