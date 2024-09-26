## @file modeling.py
#  @brief This script is the file that contains some utility functions to model the data.
#
#  @author Josué Pagán
#  @date 2024-07

# Local includes
import env as env

# External includes
from imblearn.over_sampling import SMOTENC, SMOTE
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier

# Correct class imbalance in training data
def correct_class_imbalance(train_features, train_labels, test_features, test_labels, all_categorical_features, to_categorical=False, undersampling_majority = False):

    msg = "\nSynthetic over-sampling works to cause the classifier to build larger decision regions that contain nearby minority class points."    

    # SMOTENC fails when all features are categorical
    # This is a workaround proposed here: https://github.com/scikit-learn-contrib/imbalanced-learn/issues/562
    def sample(X, y, sampler=None):
      if sampler is None:
          sampler = SMOTENC(categorical_features=[], random_state=42)
      X["temp"] = 0
      n_features = X.shape[1] - 1
      indices = range(n_features)
      sampler.set_params(categorical_features=indices)
      X_resampled, y_resampled = sampler.fit_resample(X, y)
      
      X_resampled = pd.DataFrame(X_resampled, columns = X.columns)
      X_resampled = X_resampled.drop(columns="temp")
      
      y_resampled = pd.DataFrame(y_resampled, columns = y.columns)
      return X_resampled, y_resampled

    msg += "\n\nBefore oversampling...:"
    for c in np.unique(train_labels):
        count_train = (train_labels[train_labels.columns] == c)
        msg += '\nTraining Labels class ' + str(c) + ": " + str(sum(count_train[train_labels.columns[0]].values))
        try:
          count_test = (test_labels[test_labels.columns] == c)
          msg += '\nTest Labels class ' + str(c) + ": " + str(sum(count_test[test_labels.columns[0]].values))    
        except:
          count_test = (test_labels == c)
          msg += '\nTest Labels class ' + str(c) + ": " + str(sum(count_test))

        
    if undersampling_majority:
      # Random undersampling of the majority class
      rus = RandomUnderSampler(random_state=42)
      train_features, train_labels = sample(train_features, train_labels, rus)
      msg += "\n\n\nRandom undersampling of the majority class performed correctly."
    else:
      msg += "\n\nWARNING!:The original paper on SMOTE suggested combining SMOTE with random undersampling of the majority class. This is not done here."
  
    
    
    if to_categorical:
      smotecnc = SMOTENC(categorical_features=range(train_features.shape[1]-1), random_state=42)

      # SMOTENC fails when all features are categorical
      # x_df_all, y_df_original = smotecnc.fit_resample(x_df_all, y_df_original)
      original_size = len(train_labels)
      train_features, train_labels = sample(train_features, train_labels, smotecnc)
      train_labels.value_counts(normalize=True)

    else:
      def column_index(df, query_cols):
        cols = df.columns.values
        sidx = np.argsort(cols)
        return sidx[np.searchsorted(cols,query_cols,sorter=sidx)]
        
      original_size = len(train_labels)

      if (len(all_categorical_features) == 0):
        # SMOTE for only numerical
        smote = SMOTE(random_state=42)
        train_features, train_labels = smote.fit_resample(train_features, train_labels)
        msg += "\n\n\nNon-categorical data oversampling performed correctly."
      else:
        # SMOTENC for mix and categorical
        smotecnc = SMOTENC(categorical_features=column_index(train_features, all_categorical_features), random_state=42)
        train_features, train_labels = sample(train_features, train_labels, smotecnc)

        msg += "\n\n\nCategorical data oversampling performed correctly."
    
    msg += "\nThere have been added " + str(len(train_labels)-original_size) + " samples."

    msg += "\n\nAfter oversampling...:"
    for c in np.unique(train_labels):
        count_train = (train_labels[train_labels.columns] == c)
        msg += '\nTraining Labels class ' + str(c) + ": " + str(sum(count_train[train_labels.columns[0]].values))
        try:
          count_test = (test_labels[test_labels.columns] == c)
          msg += '\nTest Labels class ' + str(c) + ": " + str(sum(count_test[test_labels.columns[0]].values))
        except:
          count_test = (test_labels == c)
          msg += '\nTest Labels class ' + str(c) + ": " + str(sum(count_test))
          

    return train_features, train_labels, msg

def multiclass_to_binary_labels(labels, class_to_keep):
  """ Convert multiclass labels to binary labels

  Args:
      labels (_type_): labels to convert
      class_to_keep (_type_): class to keep

  Returns:
      _type_: _description_
  """
  labels = labels.copy()
  labels[labels != class_to_keep] = 0
  labels[labels == class_to_keep] = 1
  return labels
  
def feature_selection(train_features, train_labels, perc_feat_select=1.0):
  knn = KNeighborsClassifier(n_neighbors=3)
  
  if perc_feat_select*len(train_features.columns) < 1: # Select, at least, 1 feature
    perc_feat_select = 1/len(train_features.columns)
      
  sfs_forward = SequentialFeatureSelector(knn, n_features_to_select = perc_feat_select, direction="forward", cv = 3).fit(train_features, train_labels)

  feature_names = np.array(train_features.columns)

  return feature_names[sfs_forward.get_support()]
