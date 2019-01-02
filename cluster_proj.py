'''
K-Means Clustering Project
unsurpervised algorithm that
clusters Universities into public or private
We usually don't have labels so metrics at the end is not typical
'''

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

'''
Getting the Data
'''

college_data = pd.read_csv('College_Data', index_col=0)
#print(college_data.head())
#print(college_data.info())
#print(college_data.describe())

'''
Exploratory Data Analysis
'''
sns.scatterplot('Grad.Rate', 'Room.Board', hue = 'Private' , data = college_data)
plt.show()

sns.scatterplot('F.Undergrad', 'Outstate', hue = 'Private', data = college_data )
plt.show()

sns.set_style('darkgrid')
g = sns.FacetGrid(college_data,hue="Private",palette='coolwarm',height=6,aspect=2)
g = g.map(plt.hist,'Outstate',bins=20,alpha=0.7)
plt.show()

sns.set_style('darkgrid')
g2 = sns.FacetGrid(college_data, hue = 'Private', palette = 'coolwarm', height= 6, aspect=2)
g2 = g2.map(plt.hist, 'Grad.Rate', bins = 20, alpha = 0.7)
plt.show()


print(college_data[college_data['Grad.Rate']>100])

college_data['Grad.Rate']['Cazenovia College'] = 100

sns.set_style('darkgrid')
g2 = sns.FacetGrid(college_data, hue = 'Private', palette = 'coolwarm', height= 6, aspect=2)
g2 = g2.map(plt.hist, 'Grad.Rate', bins = 20, alpha = 0.7)
plt.show()

'''
K Means Cluster Creation
'''

from sklearn.cluster import KMeans

k_means = KMeans(n_clusters=2)
k_means.fit(college_data.drop('Private', axis = 1))
print(k_means.cluster_centers_)

'''
Evaluation 
'''

def converter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0

college_data['Cluster'] = college_data['Private'].apply(converter)


from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(college_data['Cluster'], k_means.labels_))
print('\n')
print(classification_report(college_data['Cluster'], k_means.labels_))


