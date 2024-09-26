## @file clustering.py
#  @brief This script is the file that contains the main functions to cluster the participants of the miniBESTest balance test. It is not called from the main entry point, it is called directly from the terminal.
#
#  @author Josué Pagán
#  @date 2023-08

# External imports
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, recall_score, precision_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

# Local imports
import env as env

# ##############
# METHODS
# ##############
def standarize(df, cols):
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])

    return df


def feature_selection(X, Y, features=10):
    # Build a logistic regression model
    model = LinearRegression()
    # Define RFE
    rfe = RFE(model, n_features_to_select=features)

    # Use RFE to select the top "n" features
    fit = rfe.fit(X, Y)

    # Create a dataframe for the results
    df_RFE_results = []
    for i in range(X.shape[1]):
        df_RFE_results.append(
            {
                'Feature_names': X.columns[i],
                'Selected':  rfe.support_[i],
                'RFE_ranking':  rfe.ranking_[i],
            }
        )

    df_RFE_results = pd.DataFrame(df_RFE_results)
    df_RFE_results.index.name = 'Columns'

    # Select features
    X = X[df_RFE_results.sort_values(
        'Selected', ascending=False).Feature_names[:features].values]

    return X


# ##############
# DEFINES AND CONSTANTS
# ##############
# Constants 
HIGH_CORRELATION_THRESHOLD = 0.9

# Categorical and numerical variables
categorical_cols = ['Gender', 'Blindness', 'Height (m)', 'Weight (kg)', 'Age']
numerical_cols = ['ID', 'Test 1', 'Test 2', 'Test 3', 'Anticipatory', 'Test 5', 'Test 6', 'Reactive Postural Control', 'Test 7', 'Test 8', 'Test 9', 'Sensorial Orientation', 'Test 10', 'Test 11', 'Test 12', 'Test 13', 'Test 14',
                  'Dynamic Gait', 'mBESTest Score']

columns_to_remove = ['ID', 'Blindness']

# ##############
# MAIN
# ##############
def main():
    # Load the data
    data_df = pd.read_csv(env.INPUT_PUNCTUATIONS_FILE_NAME)

    # ##############
    # Objective 1: clustering blind and non-blind people only with the mBESTest and sociodemographic data
    # ##############
    y_df = data_df['Blindness']
    print(y_df)

    # Remove columns
    x_df = data_df.drop(columns_to_remove, axis=1)

    # Plot histograms by type of blindness superimposed
    # mBESTest Score
    plt.figure()
    plt.hist(data_df[data_df['Blindness'] == 0]
             ['mBESTest Score'], bins=10, alpha=0.5, label='Sighted')
    plt.hist(data_df[data_df['Blindness'] == 1]
             ['mBESTest Score'], bins=10, alpha=0.5, label='Blind')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of mBESTest Score by type of blindness')
    plt.legend()

    # Create the output directory if it does not exist
    if not os.path.exists(env.OUTPUT_DATA_PATH):
        os.makedirs(env.OUTPUT_DATA_PATH)
        
    # Save the plot
    plt.pause(0.5)
    plt.tight_layout()
    plt.savefig(env.OUTPUT_DATA_PATH + 'histogram_of_mBESTest_score.png')
    plt.close()

    # Anticipatory
    fig, axs = plt.subplots(2, 2)
    fig.suptitle('Partial tests')

    axs[0, 0].hist(data_df[data_df['Blindness'] == 0]
                   ['Anticipatory'], bins=10, alpha=0.5, label='Sighted')
    axs[0, 0].hist(data_df[data_df['Blindness'] == 1]
                   ['Anticipatory'], bins=10, alpha=0.5, label='Blind')
    axs[0, 0].set(xlabel='Value', ylabel='Frequency',
                  title='Histogram of Anticipatory by type of blindness')
    axs[0, 0].legend()

    # Reactive Postural Control
    axs[0, 1].hist(data_df[data_df['Blindness'] == 0]
                   ['Reactive Postural Control'], bins=10, alpha=0.5, label='Sighted')
    axs[0, 1].hist(data_df[data_df['Blindness'] == 1]
                   ['Reactive Postural Control'], bins=10, alpha=0.5, label='Blind')
    axs[0, 1].set(xlabel='Value', ylabel='Frequency',
                  title='Histogram of Reactive Postural Control by type of blindness')
    axs[0, 1].legend()

    # Sensorial Orientation
    axs[1, 0].hist(data_df[data_df['Blindness'] == 0]
                   ['Sensorial Orientation'], bins=10, alpha=0.5, label='Sighted')
    axs[1, 0].hist(data_df[data_df['Blindness'] == 1]
                   ['Sensorial Orientation'], bins=10, alpha=0.5, label='Blind')
    axs[1, 0].set(xlabel='Value', ylabel='Frequency',
                  title='Histogram of Sensorial Orientation by type of blindness')
    axs[1, 0].legend()

    # Dynamic Gait
    axs[1, 1].hist(data_df[data_df['Blindness'] == 0]
                   ['Dynamic Gait'], bins=10, alpha=0.5, label='Sighted')
    axs[1, 1].hist(data_df[data_df['Blindness'] == 1]
                   ['Dynamic Gait'], bins=10, alpha=0.5, label='Blind')
    axs[1, 1].set(xlabel='Value', ylabel='Frequency',
                  title='Histogram of Dynamic Gait by type of blindness')
    axs[1, 1].legend()

    # Save the plot
    # Maximize the plot window to full screen
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()  # Maximize the window
    plt.pause(0.5)  # Wait for 1 second
    plt.tight_layout()
    plt.savefig(env.OUTPUT_DATA_PATH + 'histogram_of_partial_tests.png')
    plt.close()

    # Plot correlation matrix
    x_corr = x_df.corr()
    
    # Remove columns with correlation >= HIGH_CORRELATION_THRESHOLD
    high_corr_features = []
    
    list_of_features = x_corr.columns
    
    for i in range(len(list_of_features)):
        for j in range(i+1, len(list_of_features)):
            if (x_corr[list_of_features[i]][list_of_features[j]] >= HIGH_CORRELATION_THRESHOLD):
                        high_corr_features.append(list_of_features[j])
    
    print('High correlation features: ', high_corr_features)    
    
    plt.matshow(x_corr)
    plt.xticks(range(x_df.shape[1]), x_df.columns, fontsize=8, rotation=90)
    plt.yticks(range(x_df.shape[1]), x_df.columns, fontsize=8)
    plt.colorbar()
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()  # Maximize the window
    plt.pause(0.5)  # Wait for 1 second
    plt.tight_layout()
    plt.savefig(env.OUTPUT_DATA_PATH + 'correlation_matrix.png')
    plt.close()

    # Remove columns that have been observed not to be relevant with the correlation matrix nor with the histograms
    x_df = x_df.drop(high_corr_features, axis=1)
    x_df = x_df.drop(['Test 1', 'Test 4', 'Test 5', 'Test 6', 'Reactive Postural Control',
                     'Test 7', 'Test 8', 'Test 9', 'Sensorial Orientation'], axis=1)

    # Standardize numerical variables (mean = 0, std = 1):
    list_of_numerical_features = x_df.columns.tolist()
    list_of_numerical_features.remove('Gender')
    list_of_numerical_features.remove('Practice sports')
    x_std_df = standarize(x_df, list_of_numerical_features)
    
    # Clustering with feature selection 
    results = pd.DataFrame(columns=['features', 'wcss', 'silhouette', 'davies_bouldin', 'calinski_harabasz', 'pearson', 'accuracy', 'precision', 'recall', 'fscore', 'blindness', 'y_kmeans'])
    
    # Feature selection
    for n in range(1, len(x_std_df.columns)+1):
        # Perform feature selection
        X_selected = feature_selection(x_std_df, y_df, n)

        # Evaluate clustering using metrics
        wcss = []
        silhouette = []
        davies_bouldin = []
        calinski_harabasz = []
        pearson = []
        accuracy = []
        fscore = []
        precision = []
        recall = []
        y_kmeans_all = []
                
        for k in range(1, 10):
            kmeans = KMeans(n_clusters=k, init='k-means++',
                            random_state=42).fit(X_selected)
            wcss.append(kmeans.inertia_)
            if len(np.unique(kmeans.labels_)) > 1:
                silhouette.append(silhouette_score(x_std_df, kmeans.labels_))
                davies_bouldin.append(davies_bouldin_score(x_std_df, kmeans.labels_))
                calinski_harabasz.append(calinski_harabasz_score(x_std_df, kmeans.labels_))
            else:
                silhouette.append(0)
                davies_bouldin.append(0)
                calinski_harabasz.append(0)
                

            # Clustering with the selected features
            y_kmeans = kmeans.predict(X_selected)
            
            # pearson correlation coefficient between the cluster labels and the target variable
            pearson_i = pearsonr(y_kmeans, y_df)[0]
            pearson.append(pearson_i)
            
            if (pearson_i < 0) and (k == 2): # If the correlation is negative and we are using 2 clusters, we need to invert the labels
                y_kmeans = 1 - y_kmeans
            accuracy.append(accuracy_score(y_df, y_kmeans))
            fscore.append(f1_score(y_df, y_kmeans, average='weighted'))
            precision.append(precision_score(y_df, y_kmeans, average='weighted'))
            recall.append(recall_score(y_df, y_kmeans, average='weighted', zero_division=0))            
            y_kmeans_all.append(y_kmeans.tolist())        

        # Plot the within cluster sum of squares (WCSS)
        fig, axs = plt.subplots(3, 3)
        fig.suptitle('Metrics for ' + str(n) + ' features: ' + str(X_selected.columns.tolist()))
        
        axs[0, 0].scatter(range(1, 10), wcss)
        axs[0, 0].plot([1, 9], [wcss[0], wcss[len(wcss)-1]])
        axs[0, 0].set(xlabel='Number of clusters', ylabel='WCSS', title='Within cluster sum of squares (WCSS)')
        
        axs[0, 1].plot(range(1, 10), silhouette, 'o-')
        axs[0, 1].set(xlabel='Number of clusters', ylabel='Silhouette', title='Silhouette')
        
        axs[0, 2].plot(range(1, 10), calinski_harabasz, 'o-')
        axs[0, 2].set(xlabel='Number of clusters', ylabel='Calinski Harabasz', title='Calinski Harabasz')
        
        axs[1, 0].plot(range(1, 10), davies_bouldin, 'o-')
        axs[1, 0].set(xlabel='Number of clusters', ylabel='Davies Bouldin', title='Davies Bouldin')
        
        axs[1, 1].plot(range(1, 10), pearson, 'o-')
        axs[1, 1].set(xlabel='Number of clusters', ylabel='Pearson', title='Pearson blind/sight kmeans vs original')
        
        axs[1, 2].plot(range(1, 10), accuracy, 'o-')
        axs[1, 2].set(xlabel='Number of clusters', ylabel='Accuracy', title='Accuracy blind/sight kmeans vs original')
        
        axs[2, 0].plot(range(1, 10), precision, 'o-')
        axs[2, 0].set(xlabel='Number of clusters', ylabel='Precision', title='Precision blind/sight kmeans vs original')
        
        axs[2, 1].plot(range(1, 10), recall, 'o-')
        axs[2, 1].set(xlabel='Number of clusters', ylabel='Recall', title='Recall blind/sight kmeans vs original')
        
        axs[2, 2].plot(range(1, 10), fscore, 'o-')
        axs[2, 2].set(xlabel='Number of clusters', ylabel='Fscore', title='Fscore blind/sight kmeans vs original')
        
        
        # Save the plot
        # Maximize the plot window to full screen
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()  # Maximize the window
        plt.pause(0.5)  # Wait for 1 second
        plt.tight_layout()
        plt.savefig(env.OUTPUT_DATA_PATH + 'clustering_metrics'+ str(n) + '_features.png')
        plt.close()


        results_nl = pd.DataFrame([[X_selected.columns.tolist(), wcss, silhouette, davies_bouldin, calinski_harabasz, pearson, accuracy, precision, recall, fscore, y_df.tolist(), y_kmeans_all]], columns=['features', 'wcss', 'silhouette', 'davies_bouldin', 'calinski_harabasz', 'pearson', 'accuracy', 'precision', 'recall', 'fscore', 'blindness', 'y_kmeans'])
        results = pd.concat([results, results_nl], ignore_index=True)
        
    results.to_csv(env.OUTPUT_DATA_PATH + 'clustering_metrics.csv', index=False, header=True)
   
   
    # #############################################################################
    # END. Plot the metrics for the different number of features manually
    # #############################################################################
    # Histogram of the best feature Test 3
    plt.hist(data_df[data_df['Blindness'] == 0]
             ['Test 3'], bins=10, alpha=0.5, label='Sighted')
    plt.hist(data_df[data_df['Blindness'] == 1]
             ['Test 3'], bins=10, alpha=0.5, label='Blind')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Test 3 by type of blindness')
    plt.legend()

    # Plot the best 3 features
    X_selected = feature_selection(x_std_df, y_df, 3)
    kmeans = KMeans(n_clusters=2, init='k-means++', random_state=42).fit(X_selected)
    y_kmeans = kmeans.predict(X_selected)
    
    fig = plt.figure()
    axs = fig.add_subplot(projection='3d')    
    x = np.array(data_df['Test 3'])
    y = np.array(data_df['Gender'])
    z = np.array(data_df['Age'])

    axs.scatter(x, y, z, zdir=(1,1,0), marker="s", c=y_kmeans, s=40, cmap="RdBu")
    axs.set_xlabel("Test 3")
    axs.set_ylabel("Gender")
    axs.set_zlabel("Age")       

    # Save the plot
    # Maximize the plot window to full screen
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()  # Maximize the window
    plt.pause(0.5)  # Wait for 1 second
    plt.tight_layout()
    plt.savefig(env.OUTPUT_DATA_PATH + '3D_clustering.png')
    plt.close()
    
if __name__ == "__main__":
    main()
