import os

# MODELING STRATEGY
FEATURES_CLASSIFICATION = 'FEATURES_CLASSIFICATION' # Classification using features and RF or other classifiers
DTW_CLASSIFICATION = 'DTW_CLASSIFICATION' # Classification using DTW distance
DL_CLASSIFICATION = 'DL_CLASSIFICATION' # Classification using Deep Learning
MODELING_STRATEGY = DL_CLASSIFICATION

# EXECUTION
VERBOSE = True # print output of main functions
PLOT_ENABLE = False # plot results
MBESTEST_TEST = "Test 3" # Test to process

# PATHS
INPUT_DATA_PATH = os.path.join(os.path.dirname(__file__), './data/input/') # Path to the input data
INPUT_IMU_DATA_PATH = os.path.join(INPUT_DATA_PATH, MBESTEST_TEST) # Path to the input IMU data
INPUT_PUNCTUATIONS_FILE_NAME = 'mbestest_punctuation.csv' # File name of the punctuation data
OUTPUT_DATA_PATH = os.path.join(os.path.dirname(__file__), './data/output/') # Path to the output data
OUTPUT_FEATURES_FILE_NAME = 'features_all_participants' # File name of the features data
OUTPUT_DTW_FILE_NAME = 'dtw_all_participants' # File name of the DTW data
OUTPUT_EXPERIMENTAL_SETUP_FILE_NAME = 'rf_experimental_setup.csv' # File name of the experimental setup data
OUTPUT_EXPERIMENTAL_DTW_SETUP_FILE_NAME = 'dtw_experimental_setup.csv' # File name of the DTW experimental setup data
OUTPUT_DTW_KNN_ASSIGNMENTS_FILE_NAME = 'dtw_knn_assignments' # File name of the DTW KNN assignments data
OUTPUT_DTW_KNN_RESULTS_FILE_NAME = 'dtw_knn_classification_results' # File name of the DTW KNN results data
OUTPUT_DL_PRODUCTION_MODEL = 'dl_production_model' # File name of the DL production model

# EXPERIMENT
OPENSENSE_FS = 50 # Sampling frequency
OPENSENSE_SENSOR_POSITIONS = ['head', 'neck', 'pelvis', 'torso', 'l_forearm', 'r_forearm', 'l_hand', 'r_hand', 'l_shank', 'r_shank', 'l_foot', 'r_foot'] # Sensors positions
OPENSENSE_SENSOR_ACC_AXES = ['ax', 'ay', 'az']  # Accelerometer axes

# DATA PROCESSING
WINDOW_SIZE_SEC = 1 # Window size in seconds for features extraction
OVERLAP_SEC = 0.5 # Overlap in seconds for features extraction
WARD_TIME_SEC = 1 # Time to remove from the beginning and end of the data in seconds to avoid noise
SLIDING_WINDOW_SIZE = 25 # Sliding window size in samples for DL classification
SLIDING_WINDOW_STEP = 1 # Sliding window step in samples for DL classification

# COLUMN NAMES
ID = 'ID'
SUBJECT_ID = 'subject_id'
BLINDNESS = 'Blindness'
LEG = 'leg'
CLASS = 'test_score'
LABEL = 'label'

# LABELING
SIGHTED = 0 # Participant is sighted
BLIND = 1 # Participant is blind

# MODELING
LEG_SELECTION = 'worst' # Select the best leg to use in the classification: 'right', 'left', 'both', 'worst'
SENSORS_SELECTION = ['r_foot', 'l_foot', 'r_shank', 'l_shank'] # Sensors to use in the classification
FEATURES_INFORMATIVE_COLUMNS_ONLY = [BLINDNESS, 'window_id', 'duration_sec'] # Informative columns to remove from the features
RAW_IMU_INFORMATIVE_COLUMNS_ONLY = [ID, LABEL, 'timestamp'] # Informative columns to remove from the raw IMU data
CORRECT_CLASS_IMBALANCE = False # Correct class imbalance by oversampling the minority class
FEATURE_SELECTION = True # Select the best features to use in the classification
PERC_FEATURES_SELEC = 0.5 # Percentage (0-1) of features to select if SFS

# CLASSIFICATION
CLASS_BAD = 0 # Bad class
CLASS_MEDIUM = 1 # Medium class
CLASS_GOOD = 2 # Good class
BINARY_CLASS = True # Binary classification
if BINARY_CLASS:
    NUM_CLASSES = 2
else:
    NUM_CLASSES = 3
CLASS_TO_KEEP = CLASS_GOOD # Class to keep in binary classification. The other class will be grouped

# RF GRID TEST
RF_GRID_TEST = {
    'n_estimators': [25],          # balance bt. model performance and training time
    'max_depth': [10],         # help control overfitting, deeper trees might overfit on small datasets
    'min_samples_split': [2],         # controlling the size of the trees
    'min_samples_leaf': [2],           # ensure each leaf has sufficient number of samples, preventing the model from learning noise
    'max_features': ['sqrt'] # control number of features to be considered, balancing trade-off bt. reducing variance and preventing overfitting
}

# RF GRID
RF_GRID = {
    'n_estimators': [25, 50, 500, 1000],          # balance bt. model performance and training time
    'max_depth': [10, 20, 30],         # help control overfitting, deeper trees might overfit on small datasets
    'min_samples_split': [2,5,10],         # controlling the size of the trees
    'min_samples_leaf': [2,5,10],           # ensure each leaf has sufficient number of samples, preventing the model from learning noise
}

# KNN DTW DISTANCES
NORMALIZE_DTW_DISTANCES = True # Normalize DTW distances for KNN

# DEEP LEARNING PARAMETERS
DL_LEARNING_RATE = 0.001 # Learning rate
DL_GRID = {
    'model__epochs': [20, 50, 100],
    'model__batch_size': [32, 64],
}
