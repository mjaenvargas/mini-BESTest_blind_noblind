## @file classification.py
#  @brief This script is the file that contains the classification functions to classify the activities of the miniBESTest balance test.
#
#  @author Josué Pagán
#  @date 2024-07

# Local includes
import env as env
import modeling as mod

# External includes
import numpy as np
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

def compute_classification_metrics_from_dtw_knn(y_real, y_predicted, y_predicted_prob):
    """
    Compute classification metrics from the results of the DTW k-NN algorithm.
    
    Parameters
    ----------
    y_real : array-like of shape (n_samples,)
        Ground truth (correct) target values.
        
    y_predicted : array-like of shape (n_samples,)
        Predicted target values.
    
    Returns
    -------
    res : array-like of shape (8,)
        Array with the classification metrics: Balanced accuracy, precision, recall, F1-Score, AUC, Matthews Correlation Coefficient, Cohen's Kappa and the confusion matrix.
    """
    # Convert to series of integers
    y_real = pd.Series(y_real).astype(int)
    y_predicted = pd.Series(y_predicted).astype(int)

    # Evaluate the aggregated predictions
    try:
        acc = metrics.balanced_accuracy_score(y_real, y_predicted)                                # Balanced accuracy
        pre = metrics.precision_score(y_real, y_predicted, average='weighted', zero_division=0)   # Precision (weighted)
        rec = metrics.recall_score(y_real, y_predicted, average='weighted', zero_division=0)      # Recall (weighted)
        f1  = metrics.f1_score(y_real, y_predicted, average='weighted')                           # F1-Score (weighted)
        
        # Check if the number of classes in y_predicted_prob is the same as the number of classes in y_real for the AUC calculation
        y_score_n_classes = y_predicted_prob.shape[1] if y_predicted_prob.ndim == 2 else 2
        classes_y_real = np.unique(y_real) 
        n_classes = len(classes_y_real)

        if n_classes != y_score_n_classes:
            # Remove the column of probabilities which index is not in the classes of y_real
            y_predicted_prob = y_predicted_prob[:, classes_y_real]
        if n_classes > 2: # Multiclass
            auc = metrics.roc_auc_score(y_real, y_predicted_prob, multi_class='ovr')                       # Area Under ROC Curve
        else: # Binary
            auc = metrics.roc_auc_score(y_real, y_predicted_prob[:, 1])                       # Area Under ROC Curve
            
        mcc = metrics.matthews_corrcoef(y_real, y_predicted)                                      # Matthews Correlation Coefficient (MCC)
        cm  = metrics.confusion_matrix(y_real, y_predicted)                                       # Confusion Matrix
        kappa = metrics.cohen_kappa_score(y_real, y_predicted)                                    # Cohen's Kappa
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        
        acc = np.nan
        pre = np.nan
        rec = np.nan
        f1  = np.nan
        auc = np.nan
        mcc = np.nan
        kappa = np.nan
        cm  = np.nan

    res = [acc, pre, rec, f1, auc, mcc, kappa, cm]
    
    # Convert res to DataFrame with the same columns as res_df
    res_df = pd.DataFrame([res], columns=['bal_acc', 'pre', 'rec', 'f1', 'auc', 'mcc', 'kappa', 'cm'])    

    # Print results
    print(f'Balanced Accuracy: {acc:.2f}')
    print(f'Precision: {pre:.2f}')
    print(f'Recall: {rec:.2f}')
    print(f'F1-Score: {f1:.2f}')
    print(f'AUC: {auc:.2f}')
    print(f'MCC: {mcc:.2f}')
    print(f'Kappa: {kappa:.2f}')
    print(f'Confusion Matrix:\n{cm}')
    
    return res_df
          
          
def compute_classification_metrics_from_features(model, x_test, y_test, task, dataset, k, experiment_id=np.nan, subject_wise=False):   
    prob = model.predict_proba(x_test)
    pred = model.predict(x_test)

    if not subject_wise:
        real = y_test
        predictions = pred
        
        if env.BINARY_CLASS:
            probabilities = prob[:, 1]
        else:
            probabilities = prob
        
    else:
        y_test['predictions'] = pred
        
        for c in range(0, prob.shape[1]):
            y_test['prob_' + str(int(c))] = prob[:, c]                    
        
        # Group by participant and label
        gr = y_test.groupby([env.SUBJECT_ID, env.CLASS, env.LEG, 'predictions']).size().reset_index(name='count')
        
        # Convert the count of each subject and label to percentage
        gr['count_perc'] = gr.groupby([env.SUBJECT_ID, env.CLASS, env.LEG])['count'].transform(lambda x: x/x.sum())
        
        # Compute and average the probabilities of each subject and label. In this way, we can get the most probable label for each participant in case of a tie in count_perc
        gr['avg_prob'] = gr.apply(lambda row: y_test[(y_test[env.SUBJECT_ID] == row[env.SUBJECT_ID]) & (y_test[env.LEG] == row[env.LEG]) & (y_test['predictions'] == row['predictions'])]['prob_' + str(int(row['predictions']))].mean(), axis=1)
                          
        # Create an array with the probabilities for each participant and label
        array_prob = gr.pivot_table(index=[env.SUBJECT_ID, env.CLASS, env.LEG], columns='predictions', values='count_perc').reset_index()
        array_prob = array_prob.fillna(0) # Convert nan to 0
        
        # Get the probabilities of the most frequent label for each participant. Try-except is used to avoid errors when there is no label 0, 1 or 2
        for c in range(0, prob.shape[1]):
            try:
                array_prob[c] = array_prob[c].values
            except:
                array_prob[c] = 0
        
        if env.BINARY_CLASS:
            predictions = array_prob[1].values > array_prob[0].values
            probabilities = array_prob[1].values
        else:
            probabilities = array_prob[range(0, prob.shape[1])].values

        # Group by participant and label and get all original labels
        real = y_test.groupby([env.SUBJECT_ID, env.CLASS, env.LEG]).size().reset_index(name='count')[env.CLASS]
        
        # Get predictions of the most frequent label for each participant. If there is a tie, get the first one
        idx = gr.groupby([env.SUBJECT_ID, env.CLASS, env.LEG])['count_perc'].transform(max) == gr['count_perc']
        idx = gr.groupby([env.SUBJECT_ID, env.CLASS, env.LEG])['avg_prob'].transform(max) == gr['avg_prob']
        
        predictions = gr[idx]['predictions']
        
    # Evaluate the aggregated predictions
    try:
        acc = metrics.balanced_accuracy_score(real, predictions)                                # Balanced accuracy
        pre = metrics.precision_score(real, predictions, average='weighted', zero_division=0)   # Precision (weighted)
        rec = metrics.recall_score(real, predictions, average='weighted', zero_division=0)      # Recall (weighted)
        f1  = metrics.f1_score(real, predictions, average='weighted')                           # F1-Score (weighted)
        auc = metrics.roc_auc_score(real, probabilities, multi_class='ovr')                     # Area Under ROC Curve
        mcc = metrics.matthews_corrcoef(real, predictions)                                      # Matthews Correlation Coefficient (MCC)
        cm  = metrics.confusion_matrix(real, predictions)                                       # Confusion Matrix
        kappa = metrics.cohen_kappa_score(real, predictions)                                    # Cohen's Kappa
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        
        acc = np.nan
        pre = np.nan
        rec = np.nan
        f1  = np.nan
        auc = np.nan
        mcc = np.nan
        kappa = np.nan
        cm  = np.nan

    if experiment_id is not np.nan:
        res = [experiment_id, acc, pre, rec, f1, auc, mcc, kappa, cm]
    else:
        res = [acc, pre, rec, f1, auc, mcc, kappa, cm]
    res = res + [task, dataset, model, k]

    # Print results
    print(f'Balanced Accuracy: {acc:.2f}')
    print(f'Precision: {pre:.2f}')
    print(f'Recall: {rec:.2f}')
    print(f'F1-Score: {f1:.2f}')
    print(f'AUC: {auc:.2f}')
    print(f'MCC: {mcc:.2f}')
    print(f'Kappa: {kappa:.2f}')
    print(f'Confusion Matrix:\n{cm}')
    
    return res

def compute_feature_importances(model, train, task, dataset, k, experiment_id=np.nan):

    values = model.feature_importances_

    if experiment_id is not np.nan:
        importances_fold = pd.DataFrame(columns=['feature', 'importance', 'task', 'dataset', 'model', 'outer_fold', 'experiment_id'])
        importances_fold['experiment_id'] = experiment_id
    else:
        importances_fold = pd.DataFrame(columns=['feature', 'importance', 'task', 'dataset', 'outer_fold'])
    importances_fold['feature'] = train.columns
    importances_fold['importance'] = values
    importances_fold['task'] = task
    importances_fold['dataset'] = dataset
    importances_fold['outer_fold'] = k

    return importances_fold

def force_balanced_5_cv_fold_for_mbesttest_test3():
    """
    This function forces the selection of folds to make them balanced. There are too few participants in the dataset to perform nested cross-validation randomly. There should be an optimization here to ensure that there are the same number of labels of each class in each fold. But it is very unbalanced in Test 3.
    
    The folds have been created manually in the "punctuation_allparticipants_B_final.xlsx" file.
    """
    
    print('Forcing balanced 5-fold cross-validation for Test 3...')
    
    # Load the punctuations from CSV file:
    punctuations = pd.read_csv(env.INPUT_DATA_PATH + env.INPUT_PUNCTUATIONS_FILE_NAME)
    manual_folders = pd.read_csv(env.INPUT_DATA_PATH + 'manual_stratification.csv')
    
    # Create the folds as a cross-validation object does
    folds = {
        0: manual_folders['fold_0'].dropna().astype(int).values,
        1: manual_folders['fold_1'].dropna().astype(int).values,
        2: manual_folders['fold_2'].dropna().astype(int).values,
        3: manual_folders['fold_3'].dropna().astype(int).values,
        4: manual_folders['fold_4'].dropna().astype(int).values
    }

    # For each fold get participants IDs and from punctuations count the number of labels of each class for Test 3 right_leg and left_leg
    for k in range(5):
        fold = folds[k]
        print(f'Fold {k}: {fold}')
        points = punctuations.loc[punctuations['ID'].isin(fold), ['ID', 'Test 3 right_leg', 'Test 3 left_leg']]
        
        # Get how many labels of each class are
        if env.BINARY_CLASS:
            for label in [0, 1]:
                if label == 0:
                    right_leg = points.loc[points['Test 3 right_leg'] != env.CLASS_TO_KEEP].shape[0]
                    left_leg = points.loc[points['Test 3 left_leg'] != env.CLASS_TO_KEEP].shape[0]
                        
                else:
                    right_leg = points.loc[points['Test 3 right_leg'] == env.CLASS_TO_KEEP].shape[0]
                    left_leg = points.loc[points['Test 3 left_leg'] == env.CLASS_TO_KEEP].shape[0]
                
                # Print absolute sum and percentage of labels
                print(f'\tLabel {label}: {right_leg+left_leg} ({(right_leg+left_leg)/(2*points.shape[0])*100:.2f}%)')
        else:
            for label in np.unique(points[['Test 3 right_leg', 'Test 3 left_leg']].values):
                right_leg = points.loc[points['Test 3 right_leg'] == label].shape[0]
                left_leg = points.loc[points['Test 3 left_leg'] == label].shape[0]
                
                # Print absolute sum and percentage of labels
                print(f'\tLabel {label}: {right_leg+left_leg} ({(right_leg+left_leg)/(2*points.shape[0])*100:.2f}%)')
        
    return folds
            
            
def perform_cv(data, task, dataset, correct_class_imbalance=False, experiment_id=np.nan):

    '''Perform patient-wise cross-validation with Random Forest'''
        
    # Create CV folds
    if env.MBESTEST_TEST in ['Test 3']:
        cv_folds = force_balanced_5_cv_fold_for_mbesttest_test3()
    else:
        raise ValueError(f'This code is not ready to do CV for a test activity different from Test 3')
               
    # Define scoring
    scoring = 'f1_weighted' # If binary classification, use 'f1' instead of 'f1_weighted'

    res_list = []
    imp_list = []

    # Perform LOSO CV splits. Select outer & inner splits based on participants IDs and test scores
    k = 1
    for f in range(len(cv_folds)):
        # Get train & test participants IDs
        test_participants = cv_folds[f]
        train_participants = np.concatenate([cv_folds[i] for i in range(len(cv_folds)) if i != f])
        
        # Get train & test data
        X_train = data[data[env.SUBJECT_ID].isin(train_participants)]
        X_train = X_train.drop(columns=[env.LEG])
        X_test = data[data[env.SUBJECT_ID].isin(test_participants)] # Keep LEG column for y_test
        
        # Get train & test labels
        y_train = pd.DataFrame(X_train[env.CLASS], columns=[env.CLASS])
        y_test = X_test[[env.SUBJECT_ID, env.CLASS, env.LEG]]
        
        X_test = X_test.drop(columns=[env.LEG]) # Drop LEG column from X_test

        # Drop ID and CLASS columns
        X_train = X_train.drop(columns=[env.SUBJECT_ID, env.CLASS])
        X_test = X_test.drop(columns=[env.SUBJECT_ID, env.CLASS])
                
        if env.FEATURE_SELECTION:
            features_selected = mod.feature_selection(X_train, y_train.values.ravel(), env.PERC_FEATURES_SELEC)
            X_train = X_train[features_selected]
            X_test = X_test[features_selected]        
        
        # Correct class imbalance in training data
        if correct_class_imbalance:
            X_train, y_train, msg = mod.correct_class_imbalance(X_train, y_train, X_test, y_test[env.CLASS], [], to_categorical=False, undersampling_majority=False)
            print(msg)
            class_weights = 'balanced'
        else:
            print('Class imbalance correction disabled.')    
            class_weights = 'balanced'
        
        # Classifier: Random Forest
        rf = RandomForestClassifier(random_state=1, class_weight=class_weights, n_jobs=-1)

        # Inner loop - Grid search CV hyperparameter search
        print(f'* Performing GridSearch CV on fold #{k}...')
        param_grid = env.RF_GRID
        rf_gs = GridSearchCV(rf, param_grid, scoring=scoring, refit=True)
        rf_gs.fit(X_train, y_train.values.flatten())

        # Get best estimator from grid search
        rf_best = rf_gs.best_estimator_
        print(f'* Best estimator: {rf_best}')

        # Compute classification metrics and feature importances
        if experiment_id is not np.nan:
            res_fold = compute_classification_metrics_from_features(rf_best, X_test, y_test, task, dataset, k, experiment_id, subject_wise=True)
            imp_fold = compute_feature_importances(rf_best, X_train, task, dataset, k, experiment_id)
        else:
            res_fold = compute_classification_metrics_from_features(rf_best, X_test, y_test, task, dataset, k, subject_wise=True)
            imp_fold = compute_feature_importances(rf_best, X_train, task, dataset, k)
        res_list.append(res_fold)
        imp_list.append(imp_fold)

        k += 1


    # Make & save end dataframes (results & importances)
    res_df = pd.DataFrame(res_list, columns=['experiment_id', 'bal_acc', 'pre', 'rec', 'f1', 'auc', 'mcc', 'kappa', 'cm', 'task', 'dataset', 'model', 'outer_fold'])

    imp_df = pd.concat(imp_list)
    
    return res_df, imp_df

def perform_dtw_knn_cv(data, dataset, correct_class_imbalance=False, experiment_id=np.nan):

    '''Perform patient-wise cross-validation with Random Forest'''
        
    # Create CV folds
    if env.MBESTEST_TEST in ['Test 3']:
        cv_folds = force_balanced_5_cv_fold_for_mbesttest_test3()
    else:
        raise ValueError(f'This code is not ready to do CV for a test activity different from Test 3')
               
    # Perform LOSO CV splits. Select outer & inner splits based on participants IDs and test scores
    labels_counter = {} # Dictionary to store the number of occurrences of each label
    
    knn_dtw_distances_assignments_df = pd.DataFrame(columns=['fold', 'activity', 'leg', 'test_part_id', 'test_part_blindness', 'features_selected', 'test_score_original', 'dtw_distance_normalized', 'overall_dtw_distances', 'train_part_ids', 'k', 'test_part_score_classified'])

    knn_res_df = pd.DataFrame(columns=['fold', 'activity', 'leg', 'k', 'bal_acc', 'pre', 'rec', 'f1', 'auc', 'mcc', 'kappa', 'cm'])

    for f in range(len(cv_folds)):
        # Get train & test participants IDs
        test_participants = cv_folds[f]
        train_participants = np.concatenate([cv_folds[i] for i in range(len(cv_folds)) if i != f])            
        
        for l in ['right', 'left']:
            # Force distances_l to be the whole dataset (distances_l = data) if you want to compute the distances for both legs
            distances_l = data[data[env.LEG] == l]            
            
            # Compute for all k nearest neighbors the number of occurrences of each label
            for k in range(1, len(train_participants)+1):

                for sp in test_participants:
                    max_label_repetitions = 0 # Maximum number of occurrences of a label

                    # Distances and labels from participant sp to all other participants in the training set for leg l
                    distances_l_sp_training = distances_l[
                        ((distances_l[env.SUBJECT_ID + '_frst'] == sp) | (distances_l[env.SUBJECT_ID + '_scnd'] == sp)) &
                        ((distances_l[env.SUBJECT_ID + '_frst'].isin(train_participants)) | (distances_l[env.SUBJECT_ID + '_scnd'].isin(train_participants)))
                    ]
                    
                    # Distances                    
                    # Get test_score of reference participants. Get labels from frst if scnd is sp, otherwise get from scnd
                    test_scores_train_part = []
                    test_scores_test_part = []
                    train_ids = []
                    for i in range(len(distances_l_sp_training)):
                        if distances_l_sp_training[env.SUBJECT_ID + '_frst'].values[i] == sp:
                            test_scores_train_part.append(distances_l_sp_training[env.CLASS + '_scnd'].values[i])
                            test_scores_test_part.append(distances_l_sp_training[env.CLASS + '_frst'].values[i])
                            train_ids.append(distances_l_sp_training[env.SUBJECT_ID + '_scnd'].values[i])
                        else:
                            test_scores_train_part.append(distances_l_sp_training[env.CLASS + '_frst'].values[i])
                            test_scores_test_part.append(distances_l_sp_training[env.CLASS + '_scnd'].values[i])
                            train_ids.append(distances_l_sp_training[env.SUBJECT_ID + '_frst'].values[i])
                    
                    # Get the score of the test participant sp. Is always the same, so we can get it from the first row
                    leg_values = distances_l_sp_training[env.LEG].tolist()
                    test_score_sp = [score for score, leg in zip(test_scores_test_part, leg_values) if leg == l][0]
                                                               
                    if env.BINARY_CLASS:
                        test_scores_train_part = [1 if score == env.CLASS_TO_KEEP else 0 for score in test_scores_train_part]
                        test_scores_test_part = [1 if score == env.CLASS_TO_KEEP else 0 for score in test_scores_test_part]         
                        test_score_sp = 1 if test_score_sp == env.CLASS_TO_KEEP else 0          
                    
                    # The overall distance is the sqrt of the sum of the square of each dtw_ distance:
                    features = ['dtw_' + s + '_' + ax for s in env.OPENSENSE_SENSOR_POSITIONS for ax in env.OPENSENSE_SENSOR_ACC_AXES]
                    
                    if env.FEATURE_SELECTION:
                        features_selected = mod.feature_selection(distances_l_sp_training[features], test_scores_train_part, env.PERC_FEATURES_SELEC)
                    else:
                        features_selected = features

                    
                    if env.NORMALIZE_DTW_DISTANCES:
                        mean = distances_l_sp_training[features_selected].mean()
                        std = distances_l_sp_training[features_selected].std()                        
                        norm_distances = (distances_l_sp_training[features_selected] - mean)/std                        
                        overall_distances = list(np.sqrt(np.sum(np.power(norm_distances, 2), axis=1)))
                    else:
                        overall_distances = list(np.sqrt(np.sum(np.power(distances_l_sp_training[features_selected], 2), axis=1)))

                    # Get the k nearest neighbors (NN). Sort the distances and get the indexes of the k smallest distances
                    NN = sorted(range(len(overall_distances)), key=lambda i: overall_distances[i], reverse=False)

                    # Reset the counters
                    labels_counter = {lb: 0 for lb in np.unique(data[env.CLASS + '_frst'])}

                    max_label_repetitions = 0
                    for r in NN[:k]:
                        lb = test_scores_train_part[r]
                        labels_counter[lb] += 1
                        max_label_repetitions = max(max_label_repetitions, labels_counter[lb])

                    # Find the label(s) with the highest frequency
                    test_scores_classified = [k for k, v in labels_counter.items() if v == max_label_repetitions]
        
                    # In case of a tie, return one at random
                    test_score_sp_classified = random.choice(test_scores_classified)
                    
                    # Compute label probabilities as the number of occurrences of each label divided by the total number of labels
                    labels_probabilities = {lb: v/k for lb, v in labels_counter.items()}
                    
                    # Save in a dataframe 
                    new_row = pd.DataFrame([{
                        'fold': f,
                        'activity': env.MBESTEST_TEST,
                        'leg': l,
                        'test_part_id': sp,
                        #'test_part_blindness': data[env.BLINDNESS + '_scnd'],
                        'features_selected': features_selected,
                        'test_score_original': int(test_score_sp),
                        'dtw_distance_normalized': int(env.NORMALIZE_DTW_DISTANCES),
                        'overall_dtw_distances': overall_distances,
                        'train_part_ids': train_ids,
                        'k': k,                        
                        'test_part_score_classified': int(test_score_sp_classified),
                        'labels_probabilities': labels_probabilities,
                    }])
                    
                    # Append the new row to the dataframe
                    knn_dtw_distances_assignments_df = pd.concat([knn_dtw_distances_assignments_df, new_row], ignore_index=True)

                # Compute classification metrics for k nearest neighbors, fold f and leg l
                filtered_df = knn_dtw_distances_assignments_df[(knn_dtw_distances_assignments_df['k'] == k) & (knn_dtw_distances_assignments_df['fold'] == f) & (knn_dtw_distances_assignments_df['leg'] == l)]
                all_label_probabilities = np.array([list(d.values()) for d in filtered_df['labels_probabilities'].values]) # Extract the labels_probabilities values and convert to a NumPy array
                              
                res_fold = compute_classification_metrics_from_dtw_knn(filtered_df['test_score_original'].values, filtered_df['test_part_score_classified'].values, all_label_probabilities)
                
                # Create a dataframe with the results of the classification metrics and f, activity, leg, k
                res_fold['fold'] = f
                res_fold['activity'] = env.MBESTEST_TEST
                res_fold['leg'] = l
                res_fold['k'] = k
                
                # Concat res_fold to res_df
                knn_res_df = pd.concat([knn_res_df, res_fold], ignore_index=True)
    
    # End of cross-validation
    return knn_dtw_distances_assignments_df, knn_res_df

           


        
