'''
K-Means Clustering Project
unsurpervised algorithm that
clusters Universities into public or private
We usually don't have labels so metrics at the end is not typical
'''

import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report


def gather_data(file):
    college_data = pd.read_csv(file, index_col=0)
    print(college_data.head())
    return college_data

def explore_data(data_frame):
    print(college_data.info())
    print(college_data.describe())

    sns.scatterplot('Grad.Rate', 'Room.Board', hue = 'Private' , data = data_frame)
    plt.show()

    sns.scatterplot('F.Undergrad', 'Outstate', hue = 'Private', data = data_frame)
    plt.show()

    sns.set_style('darkgrid')
    g = sns.FacetGrid(data_frame,hue="Private",palette='coolwarm',height=6,aspect=2)
    g = g.map(plt.hist,'Outstate',bins=20,alpha=0.7)
    plt.show()

    sns.set_style('darkgrid')
    g2 = sns.FacetGrid(data_frame, hue = 'Private', palette = 'coolwarm', height= 6, aspect=2)
    g2 = g2.map(plt.hist, 'Grad.Rate', bins = 20, alpha = 0.7)
    plt.show()


    print(data_frame[data_frame['Grad.Rate']>100])

    data_frame['Grad.Rate']['Cazenovia College'] = 100

    sns.set_style('darkgrid')
    g2 = sns.FacetGrid(data_frame, hue = 'Private', palette = 'coolwarm', height= 6, aspect=2)
    g2 = g2.map(plt.hist, 'Grad.Rate', bins = 20, alpha = 0.7)
    plt.show()

def model_creation(data_frame):
    k_means = KMeans(n_clusters=2)
    k_means.fit(data_frame.drop('Private', axis = 1))
    print(k_means.cluster_centers_)
    return k_means

def converter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0

def evaluation(data_frame, k_means):
    data_frame['Cluster'] = data_frame['Private'].apply(converter)
    print(confusion_matrix(data_frame['Cluster'], k_means.labels_))
    print('\n')
    print(classification_report(college_data['Cluster'], k_means.labels_))


if __name__ == '__main__':
    filename = 'College_Data'
    college_data = gather_data(filename)
    explore = input('Do you want to explore the data before generating model?')
    if explore == 'yes':
        explore_data(college_data)
    k_means = model_creation(college_data)
    evaluation(college_data, k_means)