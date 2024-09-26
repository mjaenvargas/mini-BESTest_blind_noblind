#############
# IMPORTS
#############
# External packages:
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
    
# Internal packages:
import env as env
import numpy as np

#############
# CONSTANTS
#############
OPENSENSE_SENSOR_POSITIONS = ['head', 'neck', 'pelvis', 'torso', 'l_forearm', 'r_forearm', 'l_hand', 'r_hand', 'l_shank', 'r_shank', 'l_foot', 'r_foot']
OPENSENSE_SENSOR_ACC_AXES = ['ax', 'ay', 'az']

OPENSENSE_SENSOR_LABELS = ['pelvis_ax', 'pelvis_ay', 'pelvis_az', 'pelvis_gx', 'pelvis_gy', 'pelvis_gz',
                        'torso_ax', 'torso_ay', 'torso_az', 'torso_gx', 'torso_gy', 'torso_gz',
                        'l_shank_ax', 'l_shank_ay', 'l_shank_az', 'l_shank_gx', 'l_shank_gy', 'l_shank_gz',
                        'l_foot_ax', 'l_foot_ay', 'l_foot_az', 'l_foot_gx', 'l_foot_gy', 'l_foot_gz',
                        'r_shank_ax', 'r_shank_ay', 'r_shank_az', 'r_shank_gx', 'r_shank_gy', 'r_shank_gz',
                        'r_foot_ax', 'r_foot_ay', 'r_foot_az', 'r_foot_gx', 'r_foot_gy', 'r_foot_gz',
                        'l_forearm_ax', 'l_forearm_ay', 'l_forearm_az', 'l_forearm_gx', 'l_forearm_gy','l_forearm_gz', 'l_hand_ax', 'l_hand_ay', 'l_hand_az', 'l_hand_gx', 'l_hand_gy', 'l_hand_gz', 'r_forearm_ax', 'r_forearm_ay', 'r_forearm_az', 'r_forearm_gx', 'r_forearm_gy', 'r_forearm_gz', 'r_hand_ax', 'r_hand_ay', 'r_hand_az', 'r_hand_gx', 'r_hand_gy', 'r_hand_gz', 'neck_ax', 'neck_ay', 'neck_az', 'neck_gx', 'neck_gy', 'neck_gz', 'head_ax', 'head_ay', 'head_az', 'head_gx', 'head_gy', 'head_gz',
                        ]
COLOR_MAP = {'ax': 'red', 'ay': 'green', 'az': 'blue'}

#############
# FUNCTIONS
#############

def plot_opensense_data(df, single=True, title=None, output_path=None, output_type='png'):
    """Plot data from OpenSensors file.

    Args:
        df (pd.DataFrame): Dataframe with the input file.
        single (bool): If True, plot all of one participant only. If False, plot all participants and average plots for normalized duration.
        title (str): Plot title.
        output_path (str): Output file path.
        output_type (str): Output file type. e.g. 'png', 'pdf'.
    """
    if env.VERBOSE:
        print("Plotting data...")
    
    # Create subplot figure
    fig = make_subplots(rows=3, cols=4, subplot_titles=OPENSENSE_SENSOR_POSITIONS)

    # For each sensor position
    for i, sensor in enumerate(OPENSENSE_SENSOR_POSITIONS, start=1):
        # Calculate row and column for subplot
        row = (i-1)//4 + 1
        col = (i-1)%4 + 1

        # Plot ax, ay, and az, and overlay the amount movement with a thicker line
        for v in OPENSENSE_SENSOR_ACC_AXES:
            color = COLOR_MAP[v]
            t = np.arange(0, len(df) / env.OPENSENSE_FS, 1 / env.OPENSENSE_FS)
            fig.add_trace(go.Scatter(x=t, y=df[sensor + '_' + v], mode='lines', line=dict(color=color, width=1.5), opacity=0.9), row=row, col=col)
            
            # Name the traces for the legend
            fig.data[-1].name = sensor + '_' + v
            
        # Add axis titles
        fig.update_xaxes(tick0 = 0.0, rangemode = "nonnegative", range=[0, max(t)], title_text="Time (s)", row=row, col=col)
        fig.update_yaxes(title_text="Amplitude (mV)", row=row, col=col)

    # Add title
    if title:
        fig.update_layout(title_text=title)

    # Show the graph
    fig.show()