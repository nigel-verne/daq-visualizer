"""
Verne Visualizer
A DAQ Data Processing and Analysis Program

Author: Nigel Veach
Original Release Date: 2023/10/04
Currently compatible with AST2 PLC DAQ System

Description:
    This program is designed for efficient Data Acquisition (DAQ) processing
    and analysis, specifically for compatibility with the AST2 PLC
    DAQ System. Utilizing a multithreading approach, it concurrently performs
    tasks such as reading data from a serial port connected to the PLC, writing
    commands back to the PLC, inserting acquired data into an SQLite
    database, and writing command info to separate log file.

    Threading enhances the program's responsiveness and performance. The
    threads include a logger for handling log messages, a data reader for
    continuous acquisition, a database inserter for persistent storage, and a
    data writer for sending commands to the PLC. The program also features a
    web-based user interface using Dash, allowing real-time visualization of
    data through dynamic graphs with interactive controls.

    All code with this program is designed


Revision History:
- 2023/10/04:   Initial release.
- 2023/10/12:   Improved edge case handling to allow for serial disconnection
- 2023/11/07:    Major revision completed.  Dynamic controls enabled, allowing for 
                improved flexibility for use with future DAQ systems.
"""

# Standard library imports
import sys
import signal
import time
import threading
import logging
from queue import Queue
from collections import deque
from datetime import datetime
import json
from pathlib import Path

# External library imports
import numpy as np
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import colorsys
import serial

# Third-party library imports
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go


# =====================================================================================
# Program Initialization
# =====================================================================================


def load_config(filename):
    """
    Load program configuration settings from a JSON file.

    Args:
        filename (Path): Path to the JSON configuration file.

    Returns:
        dict: A dictionary containing the program configuration settings.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If there is an issue decoding the JSON.
    """
    try:
        with open(filename, "r") as config_file:
            config = json.load(config_file)
        return config
    except FileNotFoundError:
        print("Config file not found.")
        raise  # Re-raise the exception
    except json.JSONDecodeError:
        print("Unable to decode JSON in config file, check file for errors.")
        raise  # Re-raise the exception


# Load the config file
config = load_config(Path("config.json"))

# Extract all global variables from the configuration file
(
    system,
    com_port,
    baud_rate,
    serial_update_rate_sec,
    database_update_rate_sec,
    database_filename,
    log_filename,
    graph_update_interval_ms,
    graph_display_duration_sec,
) = (
    config.get("system"),
    config.get("com_port"),
    config.get("baud_rate"),
    config.get("serial_update_rate_sec"),
    config.get("database_update_rate_sec"),
    config.get("database_filename"),
    config.get("log_filename"),
    config.get("graph_update_interval_ms"),
    config.get("graph_display_duration_sec"),
)

# Extract system-specific visualizer configuration
(
    sensor_labels,
    controls_layout,
    graph_layout,
) = (
    config.get(system)["sensor_labels"],
    config.get(system)["controls_layout"],
    config.get(system)["graph_layout"],
)

# define global variables to store control parameters
checkbox_inputs = []
numeric_inputs = []

# Define the maximum number of points in the graph traces
graph_trace_length = int(graph_display_duration_sec * 1000 / graph_update_interval_ms)

# configure logging
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
)

# Initialize log message queue
log_message_queue = Queue(maxsize=20)


def log_message_handler():
    """
    Continuously handles log messages from a queue and logs them.

    This function runs in a loop until the stop_flag is set. It retrieves log messages from
    the log_message_queue, extracts the log level and message, and logs the message using the
    corresponding logging level.

    Parameters:
        None

    Returns:
        None

    Raises:
        None
    """
    while not stop_flag.is_set():
        if log_message_queue.empty():
            time.sleep(0.005)
        else:
            log_message = log_message_queue.get()
            log_level, message = log_message
            getattr(logging, log_level)(message)


def initialize_database():
    """
    Initialize the SQLite database.

    This function establishes a connection to the SQLite database specified by
    DATABASE_FILENAME and creates the 'sensor_data' table if it does not exist.
    The table structure is dynamically defined based on the keys in ARDUINO_CONFIG.

    Parameters:
        None

    Returns:
        None

    Raises:
        None
    """
    with sqlite3.connect(database_filename) as conn:
        cursor = conn.cursor()

        # Define columns dynamically based on the dictionary keys
        column_definitions = [
            f"{key} REAL" if key != "Timestamp" else f"{key} TEXT"
            for key in sensor_labels
        ]

        columns = ", ".join(column_definitions)

        # Create or verify the existence of the 'sensor_data' table
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS sensor_data (
                {columns},
                PRIMARY KEY (Timestamp)
            )
            """
        )

        conn.commit()


# Initialize database
initialize_database()


# =====================================================================================
# Thread Management Functions
# =====================================================================================


# Create stop_flag to end all threads upon program exit
stop_flag = threading.Event()


def start_thread(target, *args, daemon=True):
    """
    Function to start a thread with the specified target function and settings.

    Args:
        target (function): The target function for the thread.
        *args: Additional arguments to pass to the target function.
        daemon (bool): Whether the thread should be a daemon thread.

    Returns:
        threading.Thread: The started thread.
    """
    thread = threading.Thread(target=target, args=args)
    thread.daemon = daemon
    thread.start()
    return thread


def exit_handler(sig, frame, threads):
    """
    Signal handler to close resources and exit gracefully.

    This function is intended to be used as a signal handler to close resources
    and perform cleanup operations when a specific signal is received.

    Args:
        sig (int): The signal number.
        frame (frame): The current stack frame.

    Returns:
        None

    Raises:
        None
    """

    # Set stop flag to end all threads
    stop_flag.set()

    # Clean up threads
    for thread in threads:
        try:
            thread.join()
        except Exception as e:
            logging.error(f"Failure occured when joining thread: {e}")

    # Close the serial connection
    try:
        serial_connection.close()
    except Exception as e:
        logging.error(f"Failure closing serial connection: {e}")

    # Close the program and log it
    logging.info("Exiting and closing visualizer.")
    print("\nExiting and closing visualizer.")
    sys.exit(0)


# =====================================================================================
# Serial Communication Functions
# =====================================================================================

# Initialize and open serial communication
try:
    # Attempt to open a serial connection with specified COM port and baud rate
    serial_connection = serial.Serial(com_port, baud_rate)

    # Check if the serial connection is successfully opened
    if not serial_connection.is_open:
        serial_connection.open()


except Exception as e:
    # Handle and log errors if unable to open the serial connection
    message = "Error opening serial. Check connection, COM port, & baud rate in config.json file"
    message = f"{message} ({com_port}). Full Error Message: {e}"
    print(message)  # Print the error message to the console
    logging.error(message)  # Log the error message
    sys.exit(0)  # Exit the program if serial connection cannot be established


# give time for the serial connection to initialize
time.sleep(0.05)

# Initialize deque for storing serial reads
latest_plc_values_deque = deque(maxlen=5)

# Initialize Queue for handling serial write commands
plc_command_queue = Queue()


def read_from_serial():
    """
    Background thread function to continuously read serial data from Arduino.
    It decodes the data, processes it, and appends the
    result to a global buffer for access from other programs.

    Raises:
        Exception: If an error occurs during serial communication.

    Notes:
        - Uses a lock (SER_LOCK) to synchronize access to the serial port.
        - Handles exceptions, logs errors, and attempts to reconnect after a delay.
        - Uses a delay to control the thread execution rate.
        - The thread should be started using:
            threading.Thread(target=read_serial_and_print).

    """
    # calculated the expected number of values to be recieved
    expected_serial_length = len(sensor_labels) - 1

    while not stop_flag.is_set():
        # Record the start time
        start_time = time.perf_counter()
        try:
            # Attempt to read from serial
            available_bytes = serial_connection.in_waiting

            # Check to see if there is data from the serial to read
            if available_bytes > 0:
                arduino_data = serial_connection.readline().decode().strip()

                try:
                    # Split the received data into a list of floats
                    sensor_values = list(map(float, arduino_data.split(",")))

                    # Log error if incorrect number of values received
                    if len(sensor_values) != expected_serial_length:
                        if not log_message_queue.full():
                            log_message_queue.put_nowait(
                                (
                                    "error",
                                    (
                                        f"Expected {expected_serial_length} sensor values from PLC, "
                                        f"got {sensor_values}. This error is not an issue if it "
                                        "occurs a single time when the visualizer or PLC initializes."
                                    ),
                                )
                            )

                    # Append the list of values to the deque
                    else:
                        latest_plc_values_deque.append([datetime.now()] + sensor_values)

                # If the visualizer receives an error message from the PLC on the serial
                # log this error to the log file
                except Exception as e:
                    if not log_message_queue.full():
                        log_message_queue.put_nowait(
                            (
                                "info",
                                f"Message from PLC: {str(arduino_data)}",
                            )
                        )

            # If the serial is connected but there are no available bytes to read,
            # append a new timestamp with the most recent data.  This occurs when the
            # visualizer is sending commands to the PLCpr
            elif latest_plc_values_deque:
                latest_plc_values_deque.append(
                    [datetime.now()] + latest_plc_values_deque[-1][1:]
                )

        # Log an error and attempt to reconnect if the serial connection is lost
        except Exception as e:
            if not log_message_queue.full():
                log_message_queue.put_nowait
                (
                    (
                        "error",
                        f"Arduino PLC serial disconnected. Check serial connection. Full Error Message: {str(e)}",
                    )
                )

            # Append null data so that the visualizer graphs continue updating properly
            latest_plc_values_deque.append(
                [datetime.now()] + list(np.full(expected_serial_length, np.nan))
            )

            # attempt to reconnect to the serial
            reconnect_serial()

            # Try again in 1 second to avoid flooding log file
            time.sleep(1)

        # Add a delay to accurately control loop execution time
        execution_time = time.perf_counter() - start_time
        time.sleep(max(serial_update_rate_sec - execution_time, 0))


def write_to_serial():
    """
    Continuously writes serial commands based on valve states.

    Retrieves valve states from SERIAL_COMMAND_QUEUE, builds a serial command
    using build_serial_command function, and writes it to the serial port.

    Notes:
        - The function runs in a loop until the stop flag is set.
        - If SERIAL_COMMAND_QUEUE is empty, it sleeps for a short duration.
        - Handles exceptions and logs errors.

    Returns:
        None
    """
    # Initialize default valve states (with all valves shut)
    control_states = ([0] * len(checkbox_inputs), [0] * len(numeric_inputs))
    plc_command_queue.put(
        control_states,
        timeout=5,
    )
    previous_control_states = control_states

    # Main loop for continuously writing to serial port
    while not stop_flag.is_set():
        if plc_command_queue.empty():
            # If no new commands, wait for a short duration
            time.sleep(0.05)
        else:
            try:
                # Retrieve valve states from the command queue

                new_control_states = plc_command_queue.get()

                # Build serial command based on valve states
                serial_command = build_serial_command(
                    previous_control_states, new_control_states
                )

                # Write the serial command to the port
                serial_connection.write(str(serial_command).encode())

                previous_control_states = new_control_states

            # Handle exceptions, log errors, and attempt to reconnect to serial
            except Exception as e:
                # Log an error and attempt to reconnect if the serial connection is lost
                log_message_queue.put(
                    ("error", f"Error processing serial command: {str(e)}"), timeout=0.5
                )
                reconnect_serial()
                # Wait for a short duration before attempting to reconnect again
                time.sleep(1)


def build_serial_command(previous_control_states, new_control_states):
    """
    Build a serial command based on solenoid and hanbay percentage states.

    Compares the current states with the
    previous states, updates the previous states, and returns a formatted serial
    command.

    Args:
        valve_states: tuple of all valve states

    Returns:
        str: Formatted serial command.
    """

    new_checkbox_states, new_numeric_states = new_control_states
    previous_checkbox_states, previous_numeric_states = previous_control_states

    for index, (new_checkbox_state, previous_checkbox_state) in enumerate(
        zip(new_checkbox_states, previous_checkbox_states)
    ):
        if new_checkbox_state != previous_checkbox_state:
            if new_checkbox_state == 1:
                log_message_queue.put(
                    ("info", f"{checkbox_inputs[index]} set to True"), timeout=0.5
                )
            elif new_checkbox_states == 0:
                log_message_queue.put(
                    ("info", f"{checkbox_inputs[index]} set to False (Closed)"),
                    timeout=0.5,
                )

    for index, (new_numeric_state, previous_numeric_state) in enumerate(
        zip(new_numeric_states, previous_numeric_states)
    ):
        if new_numeric_state != previous_numeric_state:
            log_message_queue.put(
                ("info", f"{numeric_inputs[index]} set to {new_numeric_state}"),
                timeout=0.5,
            )

    command_string = ",".join(
        [str(value) for array in new_control_states for value in array]
    )

    # print(command_string)
    return command_string


def reconnect_serial():
    """
    Attempt to reconnect to the serial port.
    """
    # Log the attempt to reconnect
    log_message_queue.put(
        ("info", "Attempting to reconnect to the serial port..."), timeout=0.5
    )

    try:
        # Restart the serial connection
        serial_connection.close()
        serial_connection.open()

        # Check if the connection is successfully reopened
        if serial_connection.is_open:
            log_message_queue.put(
                ("info", "Reconnected to the serial port successfully."), timeout=0.5
            )
    except Exception as e:
        # Log reconnection failure
        log_message_queue.put(
            ("error", f"Failed to reconnect to serial: {str(e)}"), timeout=0.5
        )


# =====================================================================================
# Database Function
# =====================================================================================


def insert_data_into_database():
    """
    Background thread function to continuously insert data into the database.

    This function runs as a background thread to continuously insert data into
    an SQLite database. It retrieves the most recent data from a buffer and inserts
    it into the 'sensor_data' table.

    Notes:
        - The function operates in an infinite loop, inserting data at regular intervals
        - The thread is started by creating an instance of `threading.Thread`

    Raises:
        Exception: If there is an error during the database insertion process.

    """
    while not stop_flag.is_set():
        start_time = time.perf_counter()  # Record the start time

        try:
            # Create a new connection in this thread
            with sqlite3.connect(database_filename) as local_conn:
                local_cursor = local_conn.cursor()

                # Get the most recent PLC data from the buffer
                if latest_plc_values_deque:
                    most_recent_data = latest_plc_values_deque[-1].copy()

                    # Format the timestamp
                    timestamp = most_recent_data[0].strftime("%Y-%m-%d %H:%M:%S")
                    most_recent_data[0] = timestamp

                    # Insert most_recent_data into the database
                    placeholders = ", ".join(["?" for _ in sensor_labels])
                    query = f"""
                        INSERT OR REPLACE INTO sensor_data ({", ".join(sensor_labels)})
                        VALUES ({placeholders})
                    """

                    try:
                        local_cursor.execute(query, most_recent_data)
                        local_conn.commit()
                    except Exception as e:
                        log_message_queue.put(
                            ("error", f"Error executing query: {str(e)}"), timeout=0.5
                        )

        except Exception as e:
            # Handle and log errors
            log_message_queue.put(
                ("error", f"Error inserting data into the database: {str(e)}"),
                timeout=0.5,
            )

        # Add a delay to accurately control loop execution time
        execution_time = time.perf_counter() - start_time
        time.sleep(max(database_update_rate_sec - execution_time, 0))


# =====================================================================================
# Dash App Functions
# =====================================================================================


def create_layout():
    """
    Create the layout for the Verne Data Visualizer v2.

    Returns:
        dash.html.Div: The Dash HTML Div representing the layout.

    Notes:
        This function defines the layout structure for the Verne Data Visualizer v2.
        It includes an H1 heading, buttons, input field, graphs container,and an
        interval component.
    """
    # Return HTML Div containing various components for the Verne Data Visualizer v2
    return html.Div(
        [
            # Header with the visualizer title
            html.H1(
                "Verne Data Visualizer",
                style={"textAlign": "center", "fontSize": "36px"},
            ),
            # Dynamically generate all sensor indicators
            html.Div(
                generate_sensor_indicators(sensor_labels),
                style={
                    "justifyContent": "center",
                    "display": "flex",
                    "flexWrap": "wrap",
                    "margin-bottom": "0px",
                },
            ),
            # Dynamically generate all controls
            html.Div(
                generate_controls(controls_layout),
                style={
                    "justifyContent": "center",
                    "display": "flex",
                    "flexWrap": "wrap",
                    "margin-bottom": "0px",
                },
            ),
            # Dynamically generate all graphs
            html.Div(
                id="graphs-container",
                children=initialize_graphs(graph_layout),
                style={"margin": "0px", "margin-bottom": "-80px"},
            ),
            # Interval components for graph updates and numeric indicators
            dcc.Interval(
                id="interval-component",
                interval=graph_update_interval_ms,
                n_intervals=0,
            ),
        ]
    )


def generate_controls(controls_layout):
    """
    Generate Dash components for valve controls based on a JSON configuration.

    Parameters:
    - json_file_path (str): Path to the JSON configuration file.

    Returns:
    list: List of Dash components for valve controls.
    """
    global checkbox_inputs, numeric_inputs

    controls_components = []
    for control in controls_layout:
        control_type = control.get("type")
        control_id = control.get("id")
        control_label = control.get("label")
        if control_type == "checkbox":
            checkbox_inputs.append(control_id)
            control_component = dcc.Checklist(
                id=control_id,
                options=[
                    {
                        "label": "",
                        "value": f"{control_id}",
                    },
                ],
                value=[],
                inline=True,
                style={
                    "display": "inline-block",
                },
            )
        elif control_type == "input":
            numeric_inputs.append(control_id)
            control_min = control.get("min")
            control_max = control.get("max")
            control_component = dcc.Input(
                id=control_id,
                type="number",
                placeholder=f"Enter a value({control_min} - {control_max})",
                min=control_min,
                max=control_max,
                style={
                    "display": "inline-block",
                },
            )
        else:
            # Handle other control types as needed
            continue

        label = html.Label(
            f"{control_label}:", style={"margin-left": "15px", "margin-right": "5px"}
        )
        controls_components.extend([label, control_component])

    return controls_components


def generate_sensor_indicators(sensor_labels):
    """
    Generate HTML Div elements for sensor indicators.

    Parameters:
    - sensor_labels (list): List of sensor labels.

    Returns:
    list: List of HTML Div elements containing sensor indicators.
    """
    indicators = []

    # Iterate through sensor labels, excluding "Timestamp" ([1:])
    # This is because "Timestamp" isn't reported by the PLC
    for label in sensor_labels[1:]:
        # Create an HTML Div for each sensor indicator
        indicator = html.Div(
            [
                # Create a label for the sensor indicator
                html.Label(
                    label, style={"margin-bottom": "0px", "text-align": "right"}
                ),
                # Create H2 element to display the value of the sensor
                html.H2(
                    id=f"{label}-indicator",
                    children="0",
                    style={
                        "border": "2px solid black",
                        "padding": "0px",
                        "margin-bottom": "5px",
                        "margin-top": "0px",
                        "width": "72px",
                        "text-align": "right",
                    },
                ),
            ],
            style={"margin-right": "20px", "text-align": "right"},
        )
        indicators.append(indicator)

    return indicators


def initialize_graphs(graphs):
    """
    Initialize Dash Graph components based on the provided graph configuration.
    Each graph includes traces with specific colors and layout settings.

    Args:
        graphs (list): List of dictionaries containing graph configuration details.

    Returns:
        list: List of initialized Dash Graph components.
    """

    def initialize_trace(sensor, graph_index, trace_index):
        """
        Initialize a Plotly Scatter trace for a sensor.

        Args:
            sensor (str): The sensor name.
            graph_index (int): Index of the graph.
            trace_index (int): Index of the trace within the graph.

        Returns:
            plotly.graph_objects.Scatter: Initialized Scatter trace.
        """
        # Generate trace color based on graph and trace indices
        color_hue = (graph_index * 0.2 + trace_index * 0.3) % 1
        rgb_color = colorsys.hls_to_rgb(color_hue, 0.5, 0.5)
        rgba_color = tuple(int(x * 255) for x in rgb_color) + (1,)

        # Return a Scatter trace with specified attributes
        return go.Scatter(
            x=[], y=[], mode="lines", name=sensor, line=dict(color=f"rgba{rgba_color}")
        )

    def initialize_layout(graph):
        """
        Initialize a Plotly Layout for a graph.

        Args:
            graph (dict): Graph configuration details.

        Returns:
            plotly.graph_objects.Layout: Initialized Layout.
        """
        return go.Layout(
            title=dict(text=graph["title"], x=0.5),
            xaxis=dict(title=graph["x_axis_title"], tickformat="%H:%M:%S"),
            yaxis=dict(title=graph["y_axis_title"]),
            showlegend=True,
        )

    def initialize_graph(graph_index, graph):
        """
        Initialize a Dash Graph component.

        Args:
            graph_index (int): Index of the graph.
            graph (dict): Graph configuration details.

        Returns:
            dash.html.Div: Initialized Dash Graph component.
        """
        # Initialize Scatter traces for each sensor in the graph
        data_traces = [
            initialize_trace(sensor, graph_index, trace_index)
            for trace_index, sensor in enumerate(graph["traces"])
        ]

        # Return a Dash Graph component with specified attributes
        return dcc.Graph(
            id=graph["title"],
            figure=go.Figure(data=data_traces, layout=initialize_layout(graph)),
            style={"width": "33%", "display": "inline-block", "margin": "0px"},
            config={"staticPlot": True},
        )

    # Return a list of initialized Dash Graph components
    return [
        initialize_graph(graph_index, graph) for graph_index, graph in enumerate(graphs)
    ]


def append_new_data(existing_figure, sensor_data_dict, graph_index):
    """
    Append new data to traces in graphs.

    Parameters:
    - existing_figure (dict): Existing figure.
    - sensor_data_dict (dict): Data dictionary.
    - graph_index (int): Index of the graph.

    Returns:
    tuple: Tuple containing new data dictionary, trace indices, and maximum data points.
    """

    # Extract timestamp from sensor_data_dict
    time = sensor_data_dict["Timestamp"]

    # Extract traces and their count from graph layout
    traces = graph_layout[graph_index]["traces"]
    num_traces = len(traces)

    # Create a new data dictionary with x and y values
    new_data_dict = {
        "x": list(zip([time] * num_traces)),
        "y": list(zip(sensor_data_dict[trace] for trace in traces)),
    }

    # Create trace indices corresponding to the number of traces
    trace_indices = list(range(num_traces))

    # Return the tuple with new data dictionary, trace indices, and max data points
    return new_data_dict, trace_indices, graph_trace_length


# Initialize and setup Dash App
app = dash.Dash(__name__)
app.scripts.config.serve_locally = True
app.css.config.serve_locally = True
app.layout = create_layout()

# Use ThreadPoolExecutor to run the update_indicators function callback in a separate thread
indicator_executor = ThreadPoolExecutor(max_workers=1)


@app.callback(
    [Output(f"{label}-indicator", "children") for label in sensor_labels[1:]],
    [Input("interval-component", "n_intervals")]
    + [Input(f"{id}", "value") for id in checkbox_inputs]
    + [Input(f"{id}", "n_submit") for id in numeric_inputs],
    [State(f"{id}", "value") for id in numeric_inputs],
)
def update_indicators(n_intervals, *args):
    """
    Callback function to update numeric indicators based on PLC values.

    Parameters:
    - n_intervals (int): The number of intervals that have passed.

    Returns:
    list: List of formatted values for numeric indicators.
    """
    with indicator_executor:
        # Identify triggered inputs (buttons that have been pressed)
        triggered_inputs = [
            p.split(".")[0]
            for p in dash.callback_context.triggered_prop_ids
            if p != "interval-component.n_intervals"
        ]

        # Put solenoid_state and hanbay_percentage into the command queue if triggered
        if triggered_inputs:
            # Get the callback context to access triggered inputs and states
            checkbox_states = [
                1 if state else 0 for state in args[: len(checkbox_inputs)]
            ]
            numeric_states = list(
                value if value else 0 for value in args[-len(numeric_inputs) :]
            )

            # print(f"Checkbox States: {checkbox_states}")
            # print(f"Numeric States: {numeric_states}")

            plc_command_queue.put(
                (checkbox_states, numeric_states),
                timeout=5,
            )

        # Check if there are latest PLC values available
        if latest_plc_values_deque:
            try:
                # print(latest_plc_values_deque[-1][1:])
                # Extract values from the latest PLC data
                indicator_dict = dict(
                    zip(sensor_labels[1:], latest_plc_values_deque[-1][1:])
                )
                # Format values for display
                formatted_values = [
                    f"{value}" if isinstance(value, float) else str(int(value))
                    for value in indicator_dict.values()
                ]
                return formatted_values
            except Exception as e:
                log_message_queue.put(
                    ("error", f"Error in update_indicators: {e}"), timeout=0.5
                )

        # Return a default value in case of failure
        return ["N/A"] * len(sensor_labels[1:])


# Use ThreadPoolExecutor to run the update_graph function callback in a separate thread
graph_executor = ThreadPoolExecutor(max_workers=1)


@app.callback(
    [Output(graph["title"], "extendData") for graph in graph_layout],
    Input("interval-component", "n_intervals"),
    State("graphs-container", "children"),
)
def update_graphs(n_intervals, existing_graphs):
    """
    Callback function to update data in multiple graphs.

    Parameters:
    - n_intervals (int): Number of intervals.
    - existing_graphs (list): List of existing graphs.

    Returns:
    list: List of updated data for each graph.
    """
    with graph_executor:
        # Check if there are latest PLC values available
        if latest_plc_values_deque:
            # Match the latest PLC values with their sensor labels in a dict
            sensor_data_dict = dict(zip(sensor_labels, latest_plc_values_deque[-1]))

            # Update data for each graph using the append_new_data function
            return [
                append_new_data(existing_figure, sensor_data_dict, graph_index)
                for graph_index, existing_figure in enumerate(existing_graphs)
            ]

        # Return a default value in case of failure
        return [dash.no_update] * len(graph_layout)


# =====================================================================================
# Main Function
# =====================================================================================


if __name__ == "__main__":
    # Start a thread for handling log messages
    log_message_thread = start_thread(log_message_handler)

    # Start a thread for reading from serial communication
    serial_read_thread = start_thread(read_from_serial)

    # Wait for the deque to initialize with values from the PLC
    print("Connecting to PLC...")
    timeout = 10
    start_time = time.time()
    while not latest_plc_values_deque:
        if time.time() - start_time > timeout:
            log_message_queue.put(
                ("error", "Timeout waiting for values from serial"), timeout=0.5
            )
            print("Error: Timeout connecting to PLC. Exiting program.")
            exit(1)  # Exit the program with an error code
        time.sleep(0.05)

    # Start a thread for inserting data into the database
    database_thread = start_thread(insert_data_into_database)

    # Start a thread for writing to serial communication
    write_serial_thread = start_thread(write_to_serial)

    # generate a list of all threaded functions
    threads = [
        log_message_thread,
        serial_read_thread,
        database_thread,
        write_serial_thread,
    ]

    # Set up an exit handler to gracefully handle interrupts (e.g., Ctrl+C)
    signal.signal(signal.SIGINT, lambda sig, frame: exit_handler(sig, frame, threads))

    print("\nVerne Visualizer now running at the link below. Press Ctrl-C to quit.\n")

    # Run the server with debugging enabled
    # (use_reloader set to False to avoid conflicts with threads)
    app.run_server(debug=True, use_reloader=False)
