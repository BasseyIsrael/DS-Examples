import pandas as pd
#from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
assignment_file = pd.ExcelFile('TakenMind-Python-Analytics-Problem-case-study-1-1.xlsx')
existing_employees1 = pd.read_excel(assignment_file, 'Existing employees')
gone_employees = pd.read_excel(assignment_file, 'Employees who have left')
all_employees = pd.read_excel(assignment_file, 'All Employees')
describe = gone_employees.describe()
print(describe)
all_employees = all_employees.drop('Emp ID', axis = 1)

predictors = all_employees[['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours','time_spend_company', 'Work_accident', 'promotion_last_5years']]
target = all_employees.left
pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, target, test_size = 0.3, random_state=10)


classifier = DecisionTreeClassifier()
classifier = classifier.fit(pred_train, tar_train)

prediction = classifier.predict(pred_test)

print (tar_test)

print(sklearn.metrics.confusion_matrix(tar_test, prediction))
print(sklearn.metrics.accuracy_score(tar_test, prediction))

import pickle 

with open('classifier_pickle', 'wb') as f:
    pickle.dump(classifier, f)

with open('classifier_pickle', 'rb') as f:
    np = classifier = pickle.load(f)

from sklearn.externals import joblib
joblib.dump(classifier, 'model_joblib')

joblib.load('model_joblib')
existing_employees1['salary']=existing_employees1.salary.convert_objects(convert_numeric=True)
existing_employees = existing_employees1.drop('Emp ID', axis = 1)
existing_employees = existing_employees.drop('salary', axis = 1)
existing_employees = existing_employees.drop('dept', axis = 1)
existing_employees = existing_employees.drop('left', axis = 1)
newdata = (existing_employees)

#pred_train = existing_employees
pred = classifier.predict(newdata)
print(pred)

existing_employees1['pred'] = pred
prediction = existing_employees1.drop('left', axis = 1)
#prediction.to_csv('prediction.csv')

prediction2 = prediction[prediction.pred == 1]
prediction2.to_csv('leaving.csv')
#pred.groupby('1').count()
#from sklearn.linear_model import LinearRegression
#clf = LinearRegression()
#clf.fit(pred_train, tar_train)
#clf.predict(pred_test)

#print(sklearn.metrics.confusion_matrix(tar_test, prediction))
#print(sklearn.metrics.accuracy_score(tar_test, prediction))