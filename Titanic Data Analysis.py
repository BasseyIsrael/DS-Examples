#import the necessary libraries for data analysis and model building

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

#access the necessary data
print('hi')
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

#understand the data
print(train_data.head(5))
print(test_data.head(5))

print(train_data.describe())
print(test_data.describe())
print(train_data.info())
print(test_data.info())

#understand the relationship between variables

#number of passengers that survived and did not survive
#sns.countplot(x='Survived', data=train_data)

#survival occurence by sex

#sns.countplot('Survived', hue='Sex', data=train_data)

#survival occurence by class

#sns.countplot('Survived', hue='Pclass', data=train_data)

#age distribution of the passengers

#train_data['Age'].plot.hist()

#survival occurence by age and sex

#declare variables
survived ='survived'
not_survived = 'not survived'
women = train_data[train_data['Sex']=='female']
men = train_data[train_data['Sex']=='male']

#set subplot attributes
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4))

#create plots

""""
ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label=survived, ax=axes[0], kde=False)
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label=not_survived, ax=axes[0], kde=False)
ax.legend()
ax.set_title('Female')


ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label=survived, ax=axes[1], kde=False)
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label=not_survived, ax=axes[1], kde=False)
ax.legend()
ax.set_title('Male')
"""


