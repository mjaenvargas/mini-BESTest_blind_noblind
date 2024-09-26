# IMPORT
# Internal imports
import env as env
import processing as proc

# External imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import os

from keras import Sequential
from keras.layers import Flatten, Dense, Dropout, LSTM, TimeDistributed
from keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef


#############
#METHODS
############# 
def sliding_window(x, y, window_size, dataset_size):
    X = np.array([x[i:(i+window_size)] for i in range(dataset_size - window_size)])
    y = np.array([y[i] for i in range(dataset_size - window_size)])
    
    if env.VERBOSE:
        print("shape(X) =", X.shape)
        print("shape(y) =", y.shape)

    return X, y

def create_cnn_lstm_model(input_shape, num_classes, learning_rate):
    # Define architecture
    model = Sequential()
    
    # Conv1D layer to extract temporal features, now wrapped in TimeDistributed
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(input_shape)))
    model.add(TimeDistributed(Flatten()))
    
    # LSTM layer to learn longer temporal patterns
    model.add(LSTM(50, activation='relu'))
    
    # Dropout layer for regularization
    model.add(Dropout(0.5))
    
    # Fully connected layer (Dense) for classification
    model.add(Dense(num_classes, activation='softmax'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
     # Show the model architecture 
    if env.VERBOSE:
        model.summary()
    
    return model

def load_participants_data(participant_ids, legs, scaler=None):
    # Load and append the data of all participants in the list
    data = []

    # Iterate over the list of participants and load their data
    for participant_id in participant_ids:
        # Search for files in the directory with a partial match
        leg = legs[participant_id]
        matching_files = [file for file in os.listdir(env.INPUT_IMU_DATA_PATH) if f"S{str(participant_id).zfill(2)}_raw_acc_{leg}" in file]

        if env.VERBOSE:
            print(f"For participant {participant_id}, matching files are: {matching_files}")

        if matching_files:
            # Assuming there's only one matching file, use the first one
            file_name = matching_files[0]

            # Load the data of the participant
            participant_data = pd.read_csv(os.path.join(env.INPUT_IMU_DATA_PATH, file_name))

            # Append the data of the participant to the list
            data.append(participant_data)
        else:
            print(f"Error: No matching file found for participant {participant_id}")

            
    # Concatenate the data of all participants into a single dataframe
    data = pd.concat(data)
    
    # Extract the features and target variable
    X_df = data.drop(columns = env.RAW_IMU_INFORMATIVE_COLUMNS_ONLY).reset_index(drop=True)
    
    if env.BINARY_CLASS:
        y_list = [1 if x == env.CLASS_TO_KEEP else 0 for x in data[env.LABEL]]
    else:
        y_list = data[env.LABEL]
       
    # Scale the data if a scaler is provided
    if (scaler is not None):
        X_df = scaler.fit_transform(X_df)
        
    # Convert labels to one-hot encoded format (assuming you have multi-class classification)
    y_one_hot = to_categorical(y_list)    
    
    return X_df, y_one_hot, scaler
    
def my_confussion_matrix(y_real, y_pred):
    # Refortmat
    y_real = np.argmax(y_real, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    
    return confusion_matrix(y_real, y_pred)

def plot_model_history_and_roc(history, y_real, y_pred, plot=True, plot_accuracy=True):
    # Plot the training and validation loss
    if env.PLOT_ENABLE:
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.show()
        
    if plot_accuracy:
        plt.plot(history.history['accuracy'])  # Change 'accuracy' to the appropriate metric name
        plt.plot(history.history['val_accuracy'])  # Change 'val_accuracy' to the appropriate metric name
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.show()        
        
    # Convert multi-class labels to binary labels. Force to the number of classes of y_real. Do with for-loops to ensure that the code works for any number of classes:
    y_pred_one_hot = np.zeros((y_real.shape[0], 3))
    
    for i in range(y_pred_one_hot.shape[0]):
        # Get the column index of the maximum value of the row
        idx = np.argmax(y_pred[i])
        y_pred_one_hot[i, idx] = 1    
    
    # Get the ROC curve
    fpr = dict()
    tpr = dict()
    th = dict()
    roc_auc = dict()
    for i in range(y_real.shape[1]):
        fpr[i], tpr[i], th[i] = roc_curve(y_real[:, i], y_pred_one_hot[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Plot the ROC curve
    if env.PLOT_ENABLE:
        plt.figure()
        for i in range(y_real.shape[1]):
            plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()
    
    return roc_auc

def plot_model_history_and_roc_total(history, y_real, y_pred, plot=True, plot_accuracy=True):
    # Plot the training and validation loss
    if env.PLOT_ENABLE:
        plt.figure()
        plt.plot(history.history['loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train'], loc='upper left')
        plt.show()
        
    if plot_accuracy:
        plt.plot(history.history['accuracy'])  # Change 'accuracy' to the appropriate metric name
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train'], loc='upper left')
        plt.show()   
        # Convert multi-class labels to binary labels. Force to the number of classes of y_real. Do with for-loops to ensure that the code works for any number of classes:
    y_pred_one_hot = np.zeros((y_real.shape[0], 3))
    
    for i in range(y_pred_one_hot.shape[0]):
        # Get the column index of the maximum value of the row
        idx = np.argmax(y_pred[i])
        y_pred_one_hot[i, idx] = 1    
    
    # Get the ROC curve
    fpr = dict()
    tpr = dict()
    th = dict()
    roc_auc = dict()
    for i in range(y_real.shape[1]):
        fpr[i], tpr[i], th[i] = roc_curve(y_real[:, i], y_pred_one_hot[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Plot the ROC curve
    if env.PLOT_ENABLE:
        plt.figure()
        for i in range(y_real.shape[1]):
            plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()
    
    return roc_auc

#############
# VARIABLES
#############
# Data processing
scaler = StandardScaler()

#############
# MAIN
#############
if __name__ == "__main__":
    id_leg_score_df = proc.get_score(test_activity=env.MBESTEST_TEST, leg_of_study=env.LEG_SELECTION)

    if env.VERBOSE:
        print(id_leg_score_df)

    test_scores = id_leg_score_df[env.MBESTEST_TEST].values
    legs = id_leg_score_df[env.LEG]
    ids_idx_df = pd.DataFrame(id_leg_score_df[env.ID].values, columns=["participant_id"])

    # Load punctuation data from CSV file
    punctuations_df = pd.read_csv(env.INPUT_DATA_PATH + env.INPUT_PUNCTUATIONS_FILE_NAME)

    if env.BINARY_CLASS:
        participants_evaluations_values = [1 if x == env.CLASS_TO_KEEP else 0 for x in test_scores]

        if env.VERBOSE:
            print("Binary classification. Keeping class:", env.CLASS_TO_KEEP, "as 1s and the rest are merged as 0.")
            print("New evaluation values:", participants_evaluations_values)
    else:
        participants_evaluations_values = test_scores
    evaluation_idx_df = pd.DataFrame(participants_evaluations_values, columns=["evaluation"])

    # Initialize the outer cross-validation (CV) with 5 folds
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Outer cross-validation loop            
    results_outer_loop_df = pd.DataFrame(columns=['iteration', 'train_val_ids', 'test_idxs',  'epochs', 'batch_size', 'score_test', 'auc_binary', 'cm'])

    out_it = 1
    for train_val_idx, test_idx in outer_cv.split(ids_idx_df, evaluation_idx_df):
        if env.VERBOSE:
            print("Outer CV fold: " + str(out_it)) 
            print("Train-validation indexes: ", train_val_idx)
            print("Test indexes: ", test_idx)
            
        train_val_ids, test_ids = ids_idx_df.iloc[train_val_idx], ids_idx_df.iloc[test_idx]
        y_train_val_evaluations = evaluation_idx_df.iloc[train_val_idx] # This is to split the inner CV
        
        # At this point we have the training and test indexes and names of the files of the participants for the outer CV fold. We still need to load the data and train the model!!! We will do it later.
        
        # Initialize the inner cross-validation with 5 folds
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Inner cross-validation loop
        results_inner_loop_df = pd.DataFrame(columns=['iteration', 'train_ids', 'val_idxs',  'epochs', 'batch_size', 'score'])

        inn_it = 1
        for train_index, val_index in inner_cv.split(train_val_ids, y_train_val_evaluations):
            if env.VERBOSE:
                print("Inner CV fold: " + str(inn_it))
                print("Train indexes: ", train_index)
                print("Validation indexes: ", val_index)                

            train_ids, val_ids = train_val_ids.iloc[train_index], train_val_ids.iloc[val_index]            
            train_ids = train_ids['participant_id'].tolist()
            val_ids = val_ids['participant_id'].tolist()
            
            # NOW we can load the TRAIN and VALIDATION data. "y" is one-hot encoded
            X_train, y_train, scaler_fit = load_participants_data(train_ids, legs)
            X_val, y_val, _ = load_participants_data(val_ids, legs, scaler_fit)
                       
            # Apply sliding window to your data
            window_size = env.SLIDING_WINDOW_SIZE
            dataset_size = X_train.shape[0]  # Use the size of your training data
            X_train, y_train = sliding_window(X_train, y_train, window_size, dataset_size)
            dataset_size = X_val.shape[0]  # Use the size of your training data
            X_val, y_val = sliding_window(X_val, y_val, window_size, dataset_size)
                    
            # Get the number of classes (i.e., the number of different evaluations)
            num_classes = y_train.shape[1] # y is one-hot encoded   

            # Compute the class weights (y is one-hot encoded)
            class_weights = {}
            for i in range(num_classes):
                class_weights[i] = 1.0 / np.sum(y_train[:, i])
            
            #Reshaping into subsequences
            n_steps, n_length = env.SLIDING_WINDOW_STEP, env.SLIDING_WINDOW_SIZE
            n_features = X_train.shape[2]
            X_train= X_train.reshape((X_train.shape[0], n_steps, n_length, n_features))
            X_val= X_val.reshape((X_val.shape[0], n_steps, n_length, n_features))
            input_shape = (None, n_length, n_features)
            
            # Create a CNN model using Keras:            
            model = create_cnn_lstm_model((input_shape), num_classes=num_classes, learning_rate=env.DL_LEARNING_RATE)
            
            # Perform MANUAL grid search to find the best hyperparameters without using the GridSearchCV class
            # Initialize a list to store the results. We have two lists (i) a temporary list to store the results of the inner loop and (ii) a list to store the results of the outer loop with the best hyperparameters and results of the inner loops
            temp_results_inner_loop_df = pd.DataFrame(columns=['epochs', 'batch_size', 'score'])
            
            # Add a for-loop on each combination of hyperparameters
            for epochs in env.DL_GRID['model__epochs']:
                for batch_size in env.DL_GRID['model__batch_size']:
                    # Add hyperparameters to the model and fit it.
                    # Train the model on the training data and validate it on the validation data
                    model.fit(X_train, y_train, validation_data=(X_val, y_val), class_weight=class_weights, epochs=epochs, batch_size=batch_size, verbose=0)
                    
                    # Evaluate the model on the validation data
                    score = model.evaluate(X_val, y_val)[1]
                    
                    # Concat the result to the results dataframe  
                    new_row = {'epochs': epochs, 'batch_size': batch_size, 'score': score}                  
                    temp_results_inner_loop_df = pd.concat([temp_results_inner_loop_df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
            
            # Get the best hyperparameters
            # Get the index of the best score
            best_idx = np.argmax(temp_results_inner_loop_df['score'])
            best_epochs = temp_results_inner_loop_df.iloc[best_idx]['epochs']
            best_batch_size = temp_results_inner_loop_df.iloc[best_idx]['batch_size']
            best_score = temp_results_inner_loop_df.iloc[best_idx]['score']
            
            if env.VERBOSE:
                print("Best hyperparameters [epochs, batch_size, score]:", best_epochs, best_batch_size, best_score)
            
            new_row = {'iteration': inn_it, 'train_ids': [train_index], 'val_idxs': [val_index], 'epochs': best_epochs, 'batch_size': best_batch_size, 'score': best_score}
            results_inner_loop_df = pd.concat([results_inner_loop_df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
            
            # Increment the inner loop counter
            inn_it += 1

        # In the OUTER LOOP... train the model again with the best hyperparameters and evaluate it on the test data
        # NOW we load the TRAIN+VALIDATION and TEST data
        train_val_ids, test_ids = train_val_ids['participant_id'].tolist(), test_ids['participant_id'].tolist()
        
        X_train_val, y_train_val, scaler_fit = load_participants_data(train_val_ids, legs)
        X_test, y_test, _ = load_participants_data(test_ids, legs, scaler_fit)
        
        if env.VERBOSE:
            print("Evaluating the model on the test data... whose participants are:", test_ids)
        
        window_size = env.SLIDING_WINDOW_SIZE
        dataset_size = X_train_val.shape[0]  # Train_val
        X_train_val, y_train_val = sliding_window(X_train_val, y_train_val, window_size, dataset_size)
        dataset_size = X_test.shape[0]  # Test data
        X_test, y_test = sliding_window(X_test, y_test, window_size, dataset_size)
        
        # Get the best hyperparameters and the results of the inner loops
        best_idx = np.argmax(results_inner_loop_df['score'])
        best_epochs = results_inner_loop_df.iloc[best_idx]['epochs']
        best_batch_size = results_inner_loop_df.iloc[best_idx]['batch_size']
        best_score = results_inner_loop_df.iloc[best_idx]['score']
        print("Best hyperparameters [epochs, batch_size, score]:", best_epochs, best_batch_size, best_score)        
        
        # Set the best hyperparameters
        # Get the number of classes (i.e., the number of different evaluations)
        num_classes = y_train_val.shape[1] # y is one-hot encoded                        

        # Compute the class weights (y is one-hot encoded)
        class_weights = {}
        for i in range(num_classes):
            class_weights[i] = 1.0 / np.sum(y_train_val[:, i])
            
        #Reshaping into subsequences
        n_steps, n_length = env.SLIDING_WINDOW_STEP, env.SLIDING_WINDOW_SIZE
        n_features = X_train_val.shape[2]
        X_train_val= X_train_val.reshape((X_train_val.shape[0], n_steps, n_length, n_features))
        X_test= X_test.reshape((X_test.shape[0], n_steps, n_length, n_features))
        input_shape = (None, n_length, n_features) 
               
        model = create_cnn_lstm_model((input_shape), num_classes=num_classes, learning_rate=env.DL_LEARNING_RATE)

        # Train the model on the training+validation data data and "validate" it on the test data
        history_test = model.fit(X_train_val, y_train_val, validation_data=(X_test, y_test), class_weight=class_weights, epochs=best_epochs, batch_size=best_batch_size, verbose=0)

        # Retrieve the score of the model on the test data
        score_test = model.evaluate(X_test, y_test)[1]
        if env.VERBOSE:
            print("Score on the TEST data:", score_test)        
               
        # Plot the model history and ROC curve
        y_pred = model.predict(X_test)
        auc_binary = plot_model_history_and_roc(history_test, y_test, y_pred, plot=env.PLOT_ENABLE,  plot_accuracy=True)

        # Get F1-score, precision, recall and MCC
        y_test_labels = np.argmax(y_test, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)
        f1 = f1_score(y_test_labels, y_pred_labels, average='weighted')
        precision = precision_score(y_test_labels, y_pred_labels, average='weighted')
        recall = recall_score(y_test_labels, y_pred_labels, average='weighted')
        mcc = matthews_corrcoef(y_test_labels, y_pred_labels)

        # Get the confusion matrix
        cm = my_confussion_matrix(y_test, y_pred)
        
        if env.VERBOSE:
            print("Test performance metrics:")
            print("F1-score:", f1)
            print("Precision:", precision)
            print("Recall:", recall)
            print("MCC:", mcc)            
           
            print("Test confusion matrix:")
            print(cm)
    
        # Concatenate the result to the results dataframe
        new_row = {'iteration': out_it, 'train_val_ids': [train_val_idx], 'test_idxs': [test_idx], 'epochs': best_epochs, 'batch_size': best_batch_size, 'score_test': score_test, 'auc_binary': [auc_binary], 'cm': [cm]}
        results_outer_loop_df = pd.concat([results_outer_loop_df, pd.DataFrame(new_row, index=[0])], ignore_index=True)

        # Increment the outer loop counter
        out_it += 1    
    
    # END OF THE OUTER LOOP. COMPUTE THE AVERAGE PERFORMANCE METRIC ACROSS ALL OUTER CV FOLDS
    if env.VERBOSE:
        print("End of the outer loop")
        print("Results of the outer loop:")
        print(results_outer_loop_df)
            
    # Calculate the average performance metric across all outer CV folds
    average_performance = results_outer_loop_df['score_test'].mean()
    std_performance = results_outer_loop_df['score_test'].std()
    
    if env.VERBOSE:
        print("Average Performance:", average_performance)
        print("Standard Deviation of the Performance:", std_performance)
    
    # If you need the model for production, you can train it again with the best hyperparameters and all the data
    # NOW we load ALL the data
    if env.TRAIN_PRODUCTION_DL_MODEL:
        if env.VERBOSE:
            print("Training the model for production...")
        
        ids_idx_df = ids_idx_df['participant_id'].tolist()
        X, y, _ = load_participants_data(ids_idx_df)
        
        window_size = env.SLIDING_WINDOW_SIZE
        dataset_size = X.shape[0]  # Use the size of your training data
        X, y= sliding_window(X, y, window_size, dataset_size)
        
        # Get the best hyperparameters and the results of the inner loops
        best_idx = np.argmax(results_outer_loop_df['score_test'])
        best_epochs = results_outer_loop_df.iloc[best_idx]['epochs']
        best_batch_size = results_outer_loop_df.iloc[best_idx]['batch_size']
        print("Best hyperparameters [epochs, batch_size, score]: ", best_epochs, best_batch_size, best_score)
        
        # Set the best hyperparameters
        # Compute the class weights (y is one-hot encoded)
        num_classes = y.shape[1] # y is one-hot encoded
        class_weights = {}
        for i in range(num_classes):
            class_weights[i] = 1.0 / np.sum(y[:, i])
            
        #Reshaping into subsequences
        n_steps, n_length = env.SLIDING_WINDOW_STEP, env.SLIDING_WINDOW_SIZE
        n_features = X.shape[2]
        X= X.reshape((X.shape[0], n_steps, n_length, n_features))
        input_shape = (None, n_length, n_features)       

        production_model = create_cnn_lstm_model((input_shape), num_classes=num_classes, learning_rate=env.DL_LEARNING_RATE)
        history_production = production_model.fit(X, y, class_weight=class_weights, epochs=best_epochs, batch_size=best_batch_size, verbose=0)

        # Retrieve the score of the model on the WHOLE dataset. We are not using the test data anymore
        score_production = production_model.evaluate(X, y)[1]
        
        if env.VERBOSE:
            print("Score on the WHOLE dataset:", score_production)
        
        plot_model_history_and_roc_total(history_production, y, production_model.predict(X), plot=env.PLOT_ENABLE, plot_accuracy=True)
        
        # Save the model if needed
        production_model.save(env.OUTPUT_DATA_PATH + env.OUTPUT_DL_PRODUCTION_MODEL + ".h5")
    