import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
from sklearn.linear_model import LogisticRegression
print('hi')
new_df = pd.read_csv("HR_Datasheet.csv")
print(new_df.shape)
print(new_df.columns)

print(new_df['left'].unique())

#dependent = left
#independent = satisfaction_level

train = new_df.iloc[:10000,:]
test = new_df.iloc[10001:,:]

#lor = LogisticRegression()

#lor.fit(train['satisfaction_level'].values.reshape(-1,1), train['left'])

#print(lor.predict(test['satisfaction_level'].values.reshape(-1,1)))

#print(lor.intercept_)
#print(lor.coef_)

#lor.fit(train[['satisfaction_level', 'number_project']], train['left'])
#print(lor.predict(test[['satisfaction_level', 'number_project']]))

#print(lor.intercept_)
#print(lor.coef_)



from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

print(new_df.shape)
print(new_df.columns)

print(new_df.head(5))
print(new_df['Work_accident'].unique())
new_df['Work_accident'] = new_df['Work_accident'].astype('category')
print(new_df['salary'].unique())

le = LabelEncoder()
new_df['salary_2'] = le.fit_transform(new_df['salary'])

new_df['salary_2'] = new_df['salary_2'].astype('category')

dt = DecisionTreeClassifier()

x = new_df[['salary_2', 'Work_accident']]
y = new_df['left']
dt.fit(x,y)
dt.predict(new_df[['salary_2', 'Work_accident']])

phe = OneHotEncoder()
temp = phe.fit_transform(new_df[['salary_2']]).toarray()

column_names = ['salary_'+x for x in le.classes_]

temp = pd.DataFrame(temp,columns=column_names)
print(temp.head(5))
new_df2 = pd.concat([new_df, temp], axis = 1)
print(new_df2.head())

export_graphviz(dt, 'test.dot', feature_names=['salary2', 'Work_accident'])
print('end')