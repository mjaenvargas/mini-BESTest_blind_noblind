## @file main.py
#  @brief This script is the main entry point for the classification of activities of the miniBESTest balance test.
#
#  @author Josué Pagán
#  @date 2024-06

# Local includes
import classification_ml as clf
import env as env
import modeling as mod
import processing as proc
import util as util

# External includes
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import subprocess
import sys

if __name__ == "__main__":

    #############################################################################################
    # Process the IMU data for each participant to get either the features or the DTW distances
    #############################################################################################
    proc.compute_imu_raw_data(strategy=env.MODELING_STRATEGY, test_activity=env.MBESTEST_TEST, leg_of_study=env.LEG_SELECTION)        

    if env.MODELING_STRATEGY == env.FEATURES_CLASSIFICATION:
        print("Features classification")
        print('=======================')
        
        #############################################################################################
        # Load features to make feature selection (depending on the sensors selected)
        #############################################################################################   
        knn_dtw_assignments_file_name = env.OUTPUT_DATA_PATH + env.OUTPUT_FEATURES_FILE_NAME + "_ward_" + str(int(env.WARD_TIME_SEC*1000)) + "_win_" + str(int(env.WINDOW_SIZE_SEC*1000)) + "_overlap_" + str(int(env.OVERLAP_SEC*1000)) + ".csv"
        data = pd.read_csv(knn_dtw_assignments_file_name)
        
        #############################################################################################
        # Classification models
        #############################################################################################
        
        # Random labels for classification
        tasks = ['test_score']

        results_dfs_list = []
        importances_dfs_list = []
        
        # Create the dataframe to store the experimental setup in env.py and save it to a csv file
        experimental_setup = pd.DataFrame(columns=['experiment_id', 'test_activity', 'multiclass', 'leg', 'window_size_sec', 'overlap_sec', 'ward_time_sec', 'sensors', 'correct_class_imbalance', 'rf_grid', 'feature_selection', 'perc_features_selec'])

        # Check if the experimental setup file exists
        file_path = os.path.join(env.OUTPUT_DATA_PATH, env.OUTPUT_EXPERIMENTAL_SETUP_FILE_NAME)
        if os.path.exists(file_path):
            # Read the existing file to determine the next experiment ID
            prev_experiments = pd.read_csv(file_path)
            num_experiment = prev_experiments.shape[0]
        else:
            num_experiment = 0

        # Create a new row for the experimental setup
        new_row = pd.DataFrame([{
            'experiment_id': num_experiment,
            'test_activity': env.MBESTEST_TEST,
            'multiclass': not(env.BINARY_CLASS),
            'leg': env.LEG_SELECTION,
            'window_size_sec': env.WINDOW_SIZE_SEC,
            'overlap_sec': env.OVERLAP_SEC,
            'ward_time_sec': env.WARD_TIME_SEC,
            'sensors': env.SENSORS_SELECTION,
            'correct_class_imbalance': env.CORRECT_CLASS_IMBALANCE,
            'rf_grid': env.RF_GRID,
            'feature_selection': int(env.FEATURE_SELECTION),
            'perc_features_selec': env.PERC_FEATURES_SELEC,
        }])

        # Append the new row to the file
        if num_experiment == 0:
            new_row.to_csv(file_path, index=False)
        else:
            new_row.to_csv(file_path, mode='a', header=False, index=False)
        
        print()
        
        for t in tasks:
            # Obtain labels of each classification task, i.e., risk to be predicted
            if t == env.CLASS:
                labels = data[t]
            else:
                raise ValueError(f'Unknown task: {t}')

            # Create datasets, i.e. specific sets of features 
            # Select only rows for selected leg if needed
            data_filtered = data if env.MBESTEST_TEST not in ['Test 3', 'Test 6'] or env.LEG_SELECTION == 'both' else data[data['leg'] == env.LEG_SELECTION]
            
            # Remove informative columns
            data_filtered = data_filtered.drop(columns=env.FEATURES_INFORMATIVE_COLUMNS_ONLY)
            
            # Select all features only whose names contain the selected sensors. Also select the labels and the participant ID, keeping the 'leg'
            data_filtered = data_filtered[[env.SUBJECT_ID] + [col for col in data_filtered.columns if any(sensor in col for sensor in env.SENSORS_SELECTION)] + [env.CLASS] + [env.LEG]] 
            
            # Adapt labels if binary classification is needed
            if env.BINARY_CLASS:
                labels = mod.multiclass_to_binary_labels(data_filtered[env.CLASS], env.CLASS_TO_KEEP)
                data_filtered[env.CLASS] = labels
                
                print("Binary classification selected. Class to keep: " + str(env.CLASS_TO_KEEP))
            
            print(f'Task: {t} (# classes: {len(labels.unique())})')
            print(f'Dataset: {env.MBESTEST_TEST} (# features: {data_filtered.shape[1]-2})')

            # Perform nested CV with specific test, task, and dataset
            knn_assignments, results_imp_tmp = clf.perform_cv(data_filtered, task=t, dataset=env.MBESTEST_TEST, experiment_id=num_experiment, correct_class_imbalance=env.CORRECT_CLASS_IMBALANCE)

            # Append results to end lists
            results_dfs_list.append(knn_assignments)
            importances_dfs_list.append(results_imp_tmp)

            print()

        results_clf = pd.concat(results_dfs_list)
        results_imp = pd.concat(importances_dfs_list)
        
        results_clf.to_csv(env.OUTPUT_DATA_PATH + 'rf_classification_results_' + str(num_experiment) + '.csv', index=None)
        results_imp.to_csv(env.OUTPUT_DATA_PATH + 'rf_importances_results_' + str(num_experiment) + '.csv', index=None)

        # Compute the mean and standard deviation of the metrics and create a new dataframe with the mean and std only in alternate columns
        res_avg_df = pd.DataFrame(columns=['bal_acc_avg', 'bal_acc_std', 'pre_avg', 'pre_std', 'rec_avg', 'rec_std', 'f1_avg', 'f1_std', 'auc_avg', 'auc_std', 'mcc_avg', 'mcc_std'])
        res_avg_df['bal_acc_avg'] = results_clf['bal_acc'].mean()
        res_avg_df['bal_acc_std'] = results_clf['bal_acc'].std()
        res_avg_df['pre_avg'] = results_clf['pre'].mean()
        res_avg_df['pre_std'] = results_clf['pre'].std()
        res_avg_df['rec_avg'] = results_clf['rec'].mean()
        res_avg_df['rec_std'] = results_clf['rec'].std()
        res_avg_df['f1_avg'] = results_clf['f1'].mean()
        res_avg_df['f1_std'] = results_clf['f1'].std()
        res_avg_df['auc_avg'] = results_clf['auc'].mean()
        res_avg_df['auc_std'] = results_clf['auc'].std()
        res_avg_df['mcc_avg'] = results_clf['mcc'].mean()
        res_avg_df['mcc_std'] = results_clf['mcc'].std()        
        
        if num_experiment is not np.nan:
            # Move the experiment ID to the first column of the dataframe
            cols = res_avg_df.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            res_avg_df = res_avg_df[cols]
            
        res_avg_df.to_csv(env.OUTPUT_DATA_PATH + 'classification_results_avg_' + str(num_experiment) + '.csv', index=None)
        
    
    elif env.MODELING_STRATEGY == env.DTW_CLASSIFICATION:
        print("DTW classification")
        print('==================')
        
                
        #############################################################################################
        # Load DTW distances to make feature selection (depending on the sensors selected)
        #############################################################################################   
        knn_dtw_assignments_file_name = env.OUTPUT_DATA_PATH + env.OUTPUT_DTW_FILE_NAME + "_ward_" + str(int(env.WARD_TIME_SEC*1000)) + ".csv"
        data = pd.read_csv(knn_dtw_assignments_file_name)

        labels = data[env.CLASS + '_frst']
        
        # Select only rows for selected leg if needed
        data_filtered = data if env.MBESTEST_TEST not in ['Test 3', 'Test 6'] or env.LEG_SELECTION == 'both' else data[data['leg'] == env.LEG_SELECTION]
        
        # Remove informative columns if exist
        columns_to_drop = [col for col in data_filtered.columns if any(col.startswith(prefix) for prefix in env.FEATURES_INFORMATIVE_COLUMNS_ONLY)]
        data_filtered = data_filtered.drop(columns=columns_to_drop)
        print(f'# classes: {len(labels.unique())}')
        
        # Check if the experimental setup file exists
        file_path = os.path.join(env.OUTPUT_DATA_PATH, env.OUTPUT_EXPERIMENTAL_DTW_SETUP_FILE_NAME)
        if os.path.exists(file_path):
            # Read the existing file to determine the next experiment ID
            prev_experiments = pd.read_csv(file_path)
            num_experiment = prev_experiments.shape[0]
        else:
            num_experiment = 0

        # Create a new row for the experimental setup
        new_row = pd.DataFrame([{
            'experiment_id': num_experiment,
            'test_activity': env.MBESTEST_TEST,
            'multiclass': not(env.BINARY_CLASS),
            'leg': env.LEG_SELECTION,
            'ward_time_sec': env.WARD_TIME_SEC,
            'sensors': env.SENSORS_SELECTION,
            'correct_class_imbalance': env.CORRECT_CLASS_IMBALANCE,
            'feature_selection': int(env.FEATURE_SELECTION),
            'perc_features_selec': env.PERC_FEATURES_SELEC,
            'normalization': int(env.NORMALIZE_DTW_DISTANCES),
        }])
        
        # Append the new row to the file
        if num_experiment == 0:
            new_row.to_csv(file_path, index=False)
        else:
            new_row.to_csv(file_path, mode='a', header=False, index=False)

        
        # Perform nested CV with specific test, task, and dataset
        knn_assignments, class_results = clf.perform_dtw_knn_cv(data_filtered, dataset=env.MBESTEST_TEST, experiment_id=num_experiment, correct_class_imbalance=env.CORRECT_CLASS_IMBALANCE)

        # Save the results to a csv file. If the experiment ID is not available, save the results to a file with a default name
        if num_experiment is not np.nan:
            # If file already exists, append the results to the file. Else, create a new file
            knn_dtw_assignments_file_name = env.OUTPUT_DATA_PATH + env.OUTPUT_DTW_KNN_ASSIGNMENTS_FILE_NAME + '_' + str(num_experiment) + '.csv'
            knn_class_results_file_name = env.OUTPUT_DATA_PATH + env.OUTPUT_DTW_KNN_RESULTS_FILE_NAME + '_' + str(num_experiment) + '.csv'
            
            # Save the knn assignments and the classification results to separate files
            if os.path.exists(knn_dtw_assignments_file_name):
                knn_assignments.to_csv(knn_dtw_assignments_file_name, mode='a', header=False, index=None)
            else:
                knn_assignments.to_csv(knn_dtw_assignments_file_name, index=None)
                
            if os.path.exists(knn_class_results_file_name):
                class_results.to_csv(knn_class_results_file_name, mode='a', header=False, index=None)
            else:
                class_results.to_csv(knn_class_results_file_name, index=None)
        else:
            knn_assignments.to_csv(env.OUTPUT_DATA_PATH + env.OUTPUT_DTW_KNN_ASSIGNMENTS_FILE_NAME + '.csv', index=None)
            class_results.to_csv(env.OUTPUT_DATA_PATH + env.OUTPUT_DTW_KNN_RESULTS_FILE_NAME + '.csv', index=None)

        print()
        
    elif env.MODELING_STRATEGY == env.DL_CLASSIFICATION:
        try:
            result = subprocess.run([sys.executable, 'classification_dl.py'], check=True)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running classification_dl.py: {e}")
    else:
        raise ValueError(f'Unknown modeling strategy: {env.MODELING_STRATEGY}')
            

