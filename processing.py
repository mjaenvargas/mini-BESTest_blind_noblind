## @file processing.py
#  @brief This script is the file that contains the processing functions to load and process IMU data.
#
#  @author Josué Pagán
#  @date 2024-07

# Local includes
import env as env
import util as util

# External includes
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import glob
import numpy as np
import os
import pandas as pd
import re


def get_score(participant=None, test_activity=None, leg_of_study="worst"):
    """ Get the test score and leg for the participant.

    Args:
        participant (int): Participant ID. If none, return the test score and leg for all participants.
        test_activity (str): Test activity.
        leg_of_study (str): leg to use in the classification. If "worst", get the leg with the lowest score

    Returns:
        test_score (float): Test score.
        leg (str): Leg.
    """
    # Load the punctuations from CSV file:
    punctuations = pd.read_csv(env.INPUT_DATA_PATH + env.INPUT_PUNCTUATIONS_FILE_NAME)
    
    # Get participant test score and leg (if needed)
    if (test_activity == "Test 3") | (test_activity == "Test 6"):
        if  (leg_of_study == "worst"):
            if participant is not None:                
                # Get the right and left leg score, and select the lowest one. Save the leg "left" or "right"
                # Extract scores for both legs
                right_leg_score = punctuations.loc[punctuations[env.ID] == participant, test_activity + " right_leg"].values[0]
                left_leg_score = punctuations.loc[punctuations[env.ID] == participant, test_activity + " left_leg"].values[0]
                
                # Determine the worst leg score
                test_score = min(right_leg_score, left_leg_score)
                
                # Determine the worst leg
                leg = "right" if right_leg_score < left_leg_score else "left"
                
                # Create a Dataframe with the ID, test score, and leg
                id_leg_score_df = pd.DataFrame({env.ID: [participant], env.LEG: [leg], test_activity: [test_score]})
            else:
                # Return the test score for all participants
                leg = np.where(punctuations[test_activity + " right_leg"] < punctuations[test_activity + " left_leg"], "right", "left")
                
                # Get the score of the worst leg of each participant
                test_score = np.where(punctuations[test_activity + " right_leg"] < punctuations[test_activity + " left_leg"], punctuations[test_activity + " right_leg"], punctuations[test_activity + " left_leg"])
                
                # Create a Dataframe with the ID, test score, and leg
                id_leg_score_df = pd.DataFrame({env.ID: punctuations[env.ID], env.LEG: leg, test_activity: test_score})
                
        elif (leg_of_study != "both"):
            leg = leg_of_study
            # Get the test score for the selected leg        
            if participant is not None:
                test_score = punctuations.loc[punctuations[env.ID] == participant, test_activity + " right_leg"].values[0] if leg_of_study == "right" else punctuations.loc[punctuations[env.ID] == participant, test_activity + " left_leg"].values[0]
            else:
                # Return the test score for all participants
                test_score = punctuations[test_activity + " right_leg"].values if leg_of_study == "right" else punctuations[test_activity + " left_leg"].values
            
            # Create a Dataframe with the ID, test score, and leg
            id_leg_score_df = pd.DataFrame({env.ID: punctuations[env.ID], env.LEG: leg, test_activity: test_score})
            
        else: # Both legs
            leg = np.nan
            test_score = np.nan
            
            # Create a Dataframe with the ID, test score, and leg
            id_leg_score_df = pd.DataFrame({env.ID: participant, env.LEG: leg, test_activity: test_score})
        
                    
    else:
        leg = np.nan
        
        if participant is not None:
            test_score = punctuations.loc[punctuations[env.ID] == participant, test_activity].values[0]
        else:
            # Return the test score for all participants
            test_score = punctuations[test_activity].values
            
        # Create a Dataframe with the ID, test score, and leg
        id_leg_score_df = pd.DataFrame({env.ID: punctuations[env.ID], env.LEG: leg, test_activity: test_score})
    
    return id_leg_score_df

def compute_imu_raw_data(strategy=env.FEATURES_CLASSIFICATION, test_activity=np.nan, leg_of_study="best"):
    # Load the punctuations from CSV file:
    punctuations = pd.read_csv(env.INPUT_DATA_PATH + env.INPUT_PUNCTUATIONS_FILE_NAME)
    
    # Create the output directory if it does not exist
    if not os.path.exists(env.OUTPUT_DATA_PATH):
        os.makedirs(env.OUTPUT_DATA_PATH)
        
    if strategy == env.FEATURES_CLASSIFICATION:
        # Create a log file to store the processing information
        log_file = env.OUTPUT_DATA_PATH + "log_ward_" + str(int(env.WARD_TIME_SEC*1000)) + "_win_" + str(int(env.WINDOW_SIZE_SEC*1000)) + "_overlap_" + str(int(env.OVERLAP_SEC*1000)) + ".txt"

        # Check if the global features file already exists. 
        file_name = env.OUTPUT_DATA_PATH + env.OUTPUT_FEATURES_FILE_NAME + "_ward_" + str(int(env.WARD_TIME_SEC*1000)) + "_win_" + str(int(env.WINDOW_SIZE_SEC*1000)) + "_overlap_" + str(int(env.OVERLAP_SEC*1000)) + ".csv"
        if (os.path.exists(file_name)):
            print(f'Global features file already exists.')            
            return
    
    elif strategy == env.DTW_CLASSIFICATION:
        # Create a log file to store the processing information
        log_file = env.OUTPUT_DATA_PATH + "log_dtw_ward_" + str(int(env.WARD_TIME_SEC*1000)) + ".txt"

        # Check if the global features file already exists. 
        file_name = env.OUTPUT_DATA_PATH + env.OUTPUT_DTW_FILE_NAME + "_ward_" + str(int(env.WARD_TIME_SEC*1000)) + ".csv"
        if (os.path.exists(file_name)):
            print(f'Global DTW distances file already exists.')
            return
        
    elif strategy == env.DL_CLASSIFICATION:
        # TODO: add ward time to the data processing of raw data for DL classification
        print("DL classification signal processing not tested yet, nor implemented.")        
        log_file = env.OUTPUT_DATA_PATH + "log_dl.txt"
        
    # For each participant, load the IMU data:
    for participant in range(min(punctuations[env.ID]), max(punctuations[env.ID])+1):
        # Convert participant to integer
        participant = int(participant)
         
        # Get participant blindness
        blindness = punctuations.loc[punctuations[env.ID] == participant, env.BLINDNESS].values[0]
        
        # Get participant test score and leg (if needed)
        id_leg_score_df = get_score(participant, test_activity, leg_of_study=leg_of_study)
        test_score = id_leg_score_df[test_activity].values[0]
        leg = id_leg_score_df[env.LEG].values[0]
                
        legs = ["right", "left"] if leg_of_study == "both" else [leg]

        for leg in legs:
            # Retrieve the test score for the participant if it is not available (both legs)
            if (leg_of_study == "both"):
                test_score = punctuations.loc[punctuations[env.ID] == participant, test_activity + " right_leg"].values[0] if leg == "right" else punctuations.loc[punctuations[env.ID] == participant, test_activity + " left_leg"].values[0]
            
            # Load the IMU data
            pattern = f"{env.INPUT_IMU_DATA_PATH}/*{participant:02}*{leg if leg != np.nan else ''}.csv"
            matching_files = glob.glob(pattern)

            if matching_files:
                imu_data = pd.read_csv(matching_files[0])
            else:
                print(f"No files found matching the pattern for participant {participant:02}")
            
            # Check data length in seconds
            data_length = len(imu_data) / env.OPENSENSE_FS
            
            # Remove first and last seconds to avoid noise
            imu_data = imu_data[int(env.WARD_TIME_SEC):int(len(imu_data) - env.WARD_TIME_SEC)]
            
            # Plot
            if env.PLOT_ENABLE:
                util.plot_opensense_data(imu_data, title=f"Participant: {participant:02}. Activity: {test_activity}")        
            
            if (strategy == env.FEATURES_CLASSIFICATION):
                #############################################################################################
                # Compute features form the IMU data
                ############################################################################################## 
                # Compute features for rolling windows of WINDOW_SIZE_S samples overlapping by OVERLAP_S samples
                X_df = compute_features_for_mbesttest_scoring(imu_data, participant, blindness, test_score, leg)
                
                # Append log information about participant ID, blindness, test score, leg, and data length:
                with open(log_file, "a") as file:
                    file.write(f"Participant ID: {participant}, Blindness: {blindness}, Test Score: {test_score}, Leg: {leg}, Data Length: {data_length} seconds\n")

                # Save the features. If the file already exists, append the features.
                if (os.path.exists(file_name)):
                    X_df.to_csv(file_name, mode='a', header=False, index=False)
                else:
                    X_df.to_csv(file_name, header=True, index=False)
                    
            elif (strategy == env.DTW_CLASSIFICATION):    
                # Dataframe containing the DTW distances and other information
                dtw_distances_df = pd.DataFrame(columns=[env.SUBJECT_ID + '_frst', env.SUBJECT_ID + '_scnd', env.BLINDNESS + '_frst', env.BLINDNESS + '_scnd', env.LEG, env.CLASS + '_frst', env.CLASS + '_scnd', "duration_sec_frst", "duration_sec_scnd"] + ['dtw_' + s + '_' + ax for s in env.OPENSENSE_SENSOR_POSITIONS for ax in env.OPENSENSE_SENSOR_ACC_AXES])
                
                # For each other participant, compute the DTW distance
                list_of_remaining_participants = list(punctuations[env.ID])                
                list_of_remaining_participants.remove(participant)
                
                for second_participant in range(participant+1, max(punctuations[env.ID])+1):
                    
                    # Get participant blindness
                    blindness = punctuations.loc[punctuations[env.ID] == second_participant, env.BLINDNESS].values[0]
                    
                    # Get participant test score and leg (if needed)
                    if (test_activity == "Test 3") | (test_activity == "Test 6"):
                        if  (leg_of_study == "worst"):
                            # Get the right and left leg score, and select the lowest one. Save the leg "left" or "right"
                            # Extract scores for both legs
                            right_leg_score = punctuations.loc[punctuations[env.ID] == second_participant, test_activity + " right_leg"].values[0]
                            left_leg_score = punctuations.loc[punctuations[env.ID] == second_participant, test_activity + " left_leg"].values[0]
                            
                            # Determine the worst leg score
                            test_score_scn = min(right_leg_score, left_leg_score)
                            
                            # Determine the worst leg
                            leg = "right" if right_leg_score < left_leg_score else "left"

                        elif (leg_of_study != "both"):
                            test_score_scn = punctuations.loc[punctuations[env.ID] == second_participant, test_activity + " right_leg"].values[0] if leg_of_study == "right" else punctuations.loc[punctuations[env.ID] == second_participant, test_activity + " left_leg"].values[0]
                            
                        else: # Both legs
                            test_score_scn = np.nan
                    else:
                        test_score_scn = punctuations.loc[punctuations[env.ID] == second_participant, test_activity].values[0]
                    
                    # Retrieve the test score for the participant if it is not available (both legs)
                    if (leg_of_study == "both"):
                        test_score_scn = punctuations.loc[punctuations[env.ID] == second_participant, test_activity + " right_leg"].values[0] if leg == "right" else punctuations.loc[punctuations[env.ID] == second_participant, test_activity + " left_leg"].values[0]
                    
                    # Load the IMU data
                    pattern = f"{env.INPUT_IMU_DATA_PATH}/*{second_participant:02}*{leg if leg != np.nan else ''}.csv"
                    matching_files = glob.glob(pattern)

                    if matching_files:
                        imu_data_scnd = pd.read_csv(matching_files[0])
                    else:
                        print(f"No files found matching the pattern for second participant {second_participant:02}")
                        exit()
                    
                    data_length_scnd = len(imu_data_scnd) / env.OPENSENSE_FS

                    # Remove first and last seconds to avoid noise
                    imu_data_scnd = imu_data_scnd[int(env.WARD_TIME_SEC):int(len(imu_data_scnd) - env.WARD_TIME_SEC)]
                
                    # Compute DTW distances    
                    print(f"Computing DTW distance between participant {participant:02} and participant {second_participant:02} for leg {leg}")

                    dtw_distances = compute_dtw_distances(imu_data, imu_data_scnd)
                    
                    # Append to dtw distances the information about participants ID, blindness, test score, leg, and data length:
                    new_dtw_data = pd.DataFrame(
                        [[participant, second_participant, blindness, blindness, leg, test_score, test_score_scn, data_length, data_length_scnd] + dtw_distances.values[0].tolist()],
                        columns=dtw_distances_df.columns
                    )

                    # Save the DTW distances. If the file already exists, append the features.
                    if (os.path.exists(file_name)):    
                        new_dtw_data.to_csv(file_name, mode='a', header=False, index=False)
                    else:
                        new_dtw_data.to_csv(file_name, header=True, index=False)
            
            elif (strategy == env.DL_CLASSIFICATION):
                # For Deep Learning classification only raw data is needed. Check that the raw data is already available.
                # Append log information about participant ID, blindness, test score, leg, and data length:
                with open(log_file, "a") as file:
                    file.write(f"Participant ID: {participant}, Blindness: {blindness}, Test Score: {test_score}, Leg: {leg}, Data Length: {data_length} seconds\n")
                
            else:
                print(f"Invalid strategy: {strategy}")
                exit()

                            
                        
 


def compute_features_for_mbesttest_scoring(imu_data, participant, blindness, test_score, leg):
    """ Compute the features for the miniBESTest test scoring.

    Args:
        imu_data (pd.DataFrame): IMU data.
        participant (int): Participant ID.
        blindness (int): Participant blindness.
        test_score (float): Test score.
        leg (str): Leg.
    """
    #############################################################################################
    # Preprocess the IMU data
    ############################################################################################## 
    # Compute features for rolling windows of WINDOW_SIZE_S samples overlapping by OVERLAP_S samples
    window_size = int(env.WINDOW_SIZE_SEC * env.OPENSENSE_FS)
    overlap = int(env.OVERLAP_SEC * env.OPENSENSE_FS)
    
    # Get data length in seconds
    data_length = len(imu_data) / env.OPENSENSE_FS
    
    # Compute the number of windows
    n_windows = int((len(imu_data) - window_size) / overlap) + 1
    
    # Initialize the feature matrix
    feature_columns = [env.SUBJECT_ID, env.BLINDNESS, env.LEG, "window_id"]
    for sensor in env.OPENSENSE_SENSOR_POSITIONS:
        for axis in env.OPENSENSE_SENSOR_ACC_AXES:
            feature_columns += [f"{sensor}_{axis}_mean", f"{sensor}_{axis}_std", f"{sensor}_{axis}_energy", f"{sensor}_{axis}_amplitude"]
    feature_columns += ["duration_sec", env.CLASS]
    
    # Dataframe to store the features for each window
    X_df = pd.DataFrame(columns=feature_columns)

    # For each window
    for i in range(n_windows):
        # Get the window
        window = imu_data.iloc[i * overlap: i * overlap + window_size]
        features = []

        # Append participant ID
        features.append(participant)
        
        # Append blindness
        features.append(blindness)
        
        # Append the leg, if nan, append "-"
        features.append(leg if leg is not np.nan else "-")   

        # Append the window ID
        features.append(i)
        
        # Compute the features
        for sensor in env.OPENSENSE_SENSOR_POSITIONS:
            for axis in env.OPENSENSE_SENSOR_ACC_AXES:
                axis_data = window[sensor + '_' + axis]
                # Mean
                features.append(axis_data.mean())
                # Standard Deviation
                features.append(axis_data.std())
                # Energy
                features.append(np.sum(np.square(axis_data)))                    
                # Amplitude
                amplitude = axis_data.max() - axis_data.min()
                features.append(amplitude)
        
        # Add duration
        features.append(data_length)
        
        # Add test score
        features += [test_score]

        # Append to DataFrame
        X_df.loc[len(X_df)] = features
                        
    return X_df        

def compute_dtw_distances(imu_data, imu_data_scnd):
    """ Compute the DTW distance for the miniBESTest test scoring.

    Args:
        imu_data (pd.DataFrame): IMU data.
        imu_data_scnd (pd.DataFrame): IMU data from another participant.
    """
    
    # Get column names
    X_df = pd.DataFrame(columns=[('dtw_' + s + '_' + ax) for s in env.OPENSENSE_SENSOR_POSITIONS for ax in env.OPENSENSE_SENSOR_ACC_AXES])        
    
    for s in env.OPENSENSE_SENSOR_POSITIONS:
        for ax in env.OPENSENSE_SENSOR_ACC_AXES:
            s1 = imu_data[s + '_' + ax].values
            s2 = imu_data_scnd[s + '_' + ax].values
            
            # Compute the DTW distance
            if env.PLOT_ENABLE:                
                dtw_distance, paths = dtw.warping_paths(s1, s2)
                best_path = dtw.best_path(paths)
                dtwvis.plot_warpingpaths(s1, s2, paths, best_path)
            
            else:
                # Compute DTW distance
                dtw_distance = dtw.distance(s1, s2)
                
            X_df.at[0, 'dtw_' + s + '_' + ax] = dtw_distance
                
    return X_df
    

    
    
