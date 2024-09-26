# Blind/ Sighted mBESTest balance evaluation with ML

This folder contains the code for the mBESTest balance evaluation with ML. The code is written in Python.

## Authors

- **Josué Pagán** - [email](mailto:j.pagan@upm.es)
- **Milagros Jaén** - [email](mailto:milagros.jaen@ctb.upm.es)

## Setup

It is recommended to use a virtual environment to run the code. To create a virtual environment `.venv`, and once the virtual environment is active install the required packages listed in `requirements.txt`.

To create the virtual environment with the preferred Python version, e.g. Python 3.8 or higher:
```python3.8 -m venv .venv```

To activate the virtual environment if you are in Linux or MacOS:
```source .venv/bin/activate```

Install the required packages:
```pip install -r requirements.txt```

To deactivate the virtual environment:
```deactivate```

If you are debugging in VSCode, you can set the Python interpreter to the virtual environment by pressing `Ctrl+Shift+P` and typing `Python: Select Interpreter` and selecting the Python interpreter in the `.venv` folder.

## Data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13712755.svg)](https://doi.org/10.5281/zenodo.13712755)

You can test the code with the data provided in Zenodo. The data is available at the following link: [https://zenodo.org/doi/10.5281/zenodo.13704672](https://zenodo.org/doi/10.5281/zenodo.13704672). Create a folder called `data` with a subfolder called `input`. Download the data and extract it to the `data/input/` folder. You can change the path to the data in the `env.py` module.

The dataset consists of the following files:

  1. `Test_3`: IMU data for the mBESTest test 3. Inside the folder you will find 58 CSV files with the accelerometer data for each leg of participant that performed the test 3. The data is sampled at 50 Hz and it was acquired using the OpenSense RT IMU sensors. File naming convention: `ParticipantID_raw_acc_Leg.csv` where `ParticipantID` is the participant ID, and `Leg` is the leg used for the test. Each CSV file contains the following columns:

     - `ID`: The ID of the participant.
     - `timestamp`: The timestamp of the data starting from 0.
     - `pelvis_ax`, `pelvis_ay`, `pelvis_az`: The accelerometer data for the pelvis sensor.
     - `torso_ax`, `torso_ay`, `torso_az`: The accelerometer data for the torso sensor.
     - `l_shank_ax`, `l_shank_ay`, `l_shank_az`: The accelerometer data for the left shank sensor.
     - `l_foot_ax`, `l_foot_ay`, `l_foot_az`: The accelerometer data for the left foot sensor.
     - `r_shank_ax`, `r_shank_ay`, `r_shank_az`: The accelerometer data for the right shank sensor.
     - `r_foot_ax`, `r_foot_ay`, `r_foot_az`: The accelerometer data for the right foot sensor.
     - `l_forearm_ax`, `l_forearm_ay`, `l_forearm_az`: The accelerometer data for the left forearm sensor.
     - `l_hand_ax`, `l_hand_ay`, `l_hand_az`: The accelerometer data for the left hand sensor.
     - `r_forearm_ax`, `r_forearm_ay`, `r_forearm_az`: The accelerometer data for the right forearm sensor.
     - `r_hand_ax`, `r_hand_ay`, `r_hand_az`: The accelerometer data for the right hand sensor.
     - `neck_ax`, `neck_ay`, `neck_az`: The accelerometer data for the neck sensor.
     - `head_ax`, `head_ay`, `head_az`: The accelerometer data for the head sensor.
     - `label`: The evaluation of the mBESTest test.
  
  2. `participants.csv`: The participants' information. The file contains the following columns:

     - `ID`: The ID of the participant.
     - `Blindness`: The blindness status of the participant (blind or non-blind).
     - `Age`: The age of the participant.
     - `Height (m)`: The height of the participant in meters.
     - `Weight (kg)`: The weight of the participant in kilograms.
     - `Gender`: The sex of the participant.

  3. `mbestest_punctuation.csv`: The mBESTest evaluations for the participants. The evaluations were performed by a physiotherapist and can take values from 0 to 2 (0: bad, 1: mild, 2: good performance). The file contains the following columns:

      - `ID`: The ID of the participant.
      - `Blindness`: The blindness status of the participant (blind or non-blind).
      - The evaluation of each mBESTest test from 1 to 14. For Test 3 and Test 6, the evaluation is performed for each leg. The mBESTest score is the worst of the both legs.
      - The aggregation by category of the mBESTest tests: `Anticipatory`, `Reactive Postural Control`, `Sensorial Orientation`, `Dynamic Gait`.
      - `mBESTest Score`: The total mBESTest score.

  4. `manual_stratification.csv`: A shortcut to the stratification of the participants. A manual statification to ensure that the stratification is correct in the cross-validation of the ML classification. The file contains the following columns:

      - `fold_x`: The ID of the participants in the fold x.
  
Please, if you use this data, cite the following paper:

```bibtex
@article{vargas2024automated,
  title={Automated balance assessment for blind and non-blind individuals using mini-BESTest and AI},
  author={Milagros Jaén-Vargas, Josué Pagán, Shiyang Li, María Fernanda Trujillo-Guerrero, Niloufar Kazemi, Alessio Sansó, Benito Codina, Roy Abi Zeid Daou, & José Javier Serrano Olmedo}
  journal={TBD},
  volume={TBD},
  pages={TBD},
  year={2024},
  publisher={TBD},
  doi={TBD}
}
```

## Usage

The main script is `main.py`. To run the script, use the following command. Ensure that you have the correct environment variables set in the `env.py` module before running the script.
```python main.py```

> **Note:**
> The script `classification_dl.py` is an independent script that performs the classification based on the raw data using a deep learning architecture. It can be run independently or from the main script.

This project processes IMU (Inertial Measurement Unit) data for participants to either extract features or compute DTW (Dynamic Time Warping) distances. The main script, main.py, orchestrates the data processing and classification tasks based on the specified modeling strategy.

> **Warning:**
> To run the clustering based on mBESTest evaluations you can use the `clustering.py` script, which is an independent script that performs clustering. Run the script with the following command:
```python clustering.py```

### Script details

The main execution block of the script performs the following key functions **<span style="color: red;">(for Machine Learning classification using Random Forest and KNN from the DTW distances only):</span>**

- It always processes the IMU data for each participant in the data folder for the test `MBESTEST_TEST` to get either the features or the DTW distances depending on the modeling strategy (`MODELING_STRATEGY`). Both `MBESTEST_TEST` and the `MODELING_STRATEGY` are defined in the `env.py` module.

- Depending on the modeling strategy, the script will either perform an mBESTest balance classification of the `MBESTEST_TEST` activity using features and RF or other classifiers, or the classification will be based on the DTW distances.

**<span style="color: blue;">For Deep Learning classification using LSTM + CNN:</span>**

- The script will process the raw IMU data for each participant in the data folder for the test `MBESTEST_TEST` to get the raw IMU data for the DL classification.

#### Classification based on features

If a **classification based on features** is performed (e.g., `MODELING_STRATEGY` is `FEATURES_CLASSIFICATION`), the script will:

1. **Compute the features** depending on the sensors selected (`SENSORS_SELECTION` in `env.py`). The features were computed in the previous step and saved in the output data folder.

2. **Do classification of the mBESTest score** (`env.CLASS = test_score` in `env.py`).

   - `Test_3` and `Test_6` are the only tests that perform the exercise twice, once with each leg. The `LEG_SELECTION` in `env.py` specifies the leg to use for the classification. The `LEG_SELECTION` can be `left`, `right`, `both`, or `worst`. If `LEG_SELECTION` is `worst`, the script will select the leg with the worst mBESTest score.
   - The `BINARY_CLASS` in `env.py` specifies whether to perform a binary classification or a multi-class classification. If `BINARY_CLASS` is `True`, the script will perform a binary classification of the mBESTest score. If `BINARY_CLASS` is `False`, the script will perform a multi-class classification of the mBESTest score. If it is a binary classification, the 3 classes of the mBESTest score are grouped into 2 classes: `0` and `1` keeping unmerged the class `CLASS_TO_KEEP` defined in `env.py`.
   - The script performs a **nested CV**. It will correct class imbalance using the `SMOTE` algorithm if `env.CORRECT_CLASS_IMBALANCE` is `True`. Currently the method `perform_cv` in the `classification.py` module performs **patient-wise cross-validation with Random Forest only**.
   - If `FEATURE_SELECTION` is `True`, the script will perform feature selection using the Sequential Forward Selection in the `feature_selection` method in the `modeling.py` module. The Sequential Forward Selection is based on the Random Forest classifier. The number of features to select is defined as a percentage (0-1) in `PERC_FEATURES_SELEC` in `env.py`.
   - The script generates a Python dataframe with the following columns: `experiment_id`, `bal_acc`, `pre`, `rec`, `f1`, `auc`, `mcc`, `kappa`, `cm`, `task`, `dataset`, `model`, `outer_fold`.
   - At the end of the cross-validation, the main script computes the mean and standard deviation of the metrics and create a new dataframe with the mean and std only.
   - The script saves the results in a CSV file in the output data folder (`env.OUTPUT_DATA_PATH`).

#### Classification based on DTW distances

If the **classification is based on DTW distances** (`MODELING_STRATEGY` is `DTW_CLASSIFICATION`), the script will:

1. Load the DTW distances computed in the previous step and saved in the output data folder (method `proc.compute_imu_raw_data`).

2. Do classification of the mBESTest score (`env.CLASS = test_score` in `env.py`).

    - The script performs a **nested CV**. It will correct class imbalance using the `SMOTE` algorithm if `env.CORRECT_CLASS_IMBALANCE` is `True`. Currently the method `perform_dtw_knn_cv` in the `classification.py` module performs **patient-wise cross-validation with KNN**. It generates a Python dataframe with the following metrics: `bal_acc`, `pre`, `rec`, `f1`, `kappa`, `cm`.
    - At the end of the cross-validation, the main script saves the KNN assignments and the classification results to separate files in the output data folder (`env.OUTPUT_DATA_PATH`).

#### Classification based on raw data (Deep Learning)

If the **classification is based on raw data** (`MODELING_STRATEGY` is `DL_CLASSIFICATION`), the script will:

- Load the raw IMU data and it will create a sliding window of samples with the size defined in `SLIDING_WINDOW_SIZE` and the step defined in `SLIDING_WINDOW_STEP` in `env.py`.

- The script will perform a **nested CV** with an outer and inner loop. The outer loop will perform a patient-wise cross-validation, and the inner loop will perform a leave-one-out cross-validation. The script **does not** correct class imbalance.

## Environment Variables

The script relies on several environment variables defined in the env module. Ensure you edit the following environment variables in the `env.py` module to match your data and modeling strategy:

- Ensure that you add the correct input and output data paths in the `env.py` module.
- **`MODELING_STRATEGY`**: Defines the modeling strategy to use: `FEATURES_CLASSIFICATION`, `DTW_CLASSIFICATION` or `DL_CLASSIFICATION` for classification based on features, DTW distances, or raw data (deep learning), respectively.
- **`MBESTEST_TEST`**: Specifies the test activity to process. The test activity used so far has been `Test_3`.
- **`WARD_TIME_SEC`**: Ward time in seconds to remove from the beginning and end of the activity data.
- **`WINDOW_SIZE_SEC`**: Window size in seconds for the feature extraction on the IMU data.
- **`OVERLAP`**: Overlap window size in seconds.
- **`SLIDING_WINDOW_SIZE`**: Sliding window size in samples for the DL model.
- **`SLIDING_WINDOW_STEP`**: Sliding window step in samples for the DL model.
- **`LEG_SELECTION`**: Indicates the leg selection for processing: `right`, `left`, `both`, or `worst`.
- **`SENSORS_SELECTION`**: Specifies the sensors to use to work with of those defined in `OPENSENSE_SENSOR_POSITIONS`.
- **`FEATURES_INFORMATIVE_COLUMNS_ONLY`**: Specifies the informative columns to filter from the features and DTW distances dataframes.
- **`RAW_IMU_INFORMATIVE_COLUMNS_ONLY`**: Specifies the informative columns to filter from the raw IMU data in the DL classification.
- **`CORRECT_CLASS_IMBALANCE`**: Specifies whether to correct the class imbalance using the SMOTE algorithm.
- **`FEATURE_SELECTION`**: Specifies whether to perform feature selection using the Sequential Forward Selection.
- **`PERC_FEATURES_SELEC`**: Specifies the percentage of features to select in the Sequential Forward Selection (0-1).
- **`BINARY_CLASS`**: Specifies whether to perform a binary classification (`True`) or a multi-class classification (`False`).
- **`CLASS_TO_KEEP`**: Specifies the class to keep in the binary classification. The other classes are merged into the other class.
- **`RF_GRID`**: Specifies the grid of hyperparameters for the Random Forest classifier.
- **`NORMALIZE_DTW_DISTANCES`**: Specifies whether to normalize (`True`) the DTW distances for KNN (default is `True`).
- **`DL_LEARNING_RATE`**: Specifies the learning rate for the DL model.
- **`DL_GRID`**: Specifies the grid of hyperparameters for the DL model.

## License

This project is licensed under the GNU GPLv3 License. See the LICENSE file for details.
