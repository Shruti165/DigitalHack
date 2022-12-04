#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pip
import seaborn as sns
import warnings

#import Data
hr_data = pd.read_csv("test.csv")

#checks for any empty values
hr_data.isnull().sum()
hr_data.isnull().values.any()

#understand columns , dataypes
hr_data.head()
hr_data.dtypes
hr_data.shape
hr_data.describe()

hr_data['Productivity_Rate'].value_counts()
sns.countplot(hr_data['Productivity_Rate'])
sns.barplot(x = 'Gender' , y = 'Yes', data = hr_data)
sns.barplot(x = 'Department', y = 'Yes', data = hr_data)
import matplotlib.pyplot as plt

fig_dims = (12, 4)
fig, ax = plt.subplots(figsize=fig_dims)
sns.countplot(x='Age', hue='Productivity_Rate', data=hr_data, palette="colorblind", ax=ax,
              edgecolor=sns.color_palette("dark", n_colors=1));

# Print all of the object data types and their unique values
for column in hr_data.columns:
    if hr_data[column].dtype == object:
        print(str(column) + ' : ' + str(hr_data[column].unique()))
        print(hr_data[column].value_counts())
        print("_________________________________________________________________")

hr_data.corr()
# Visualize the correlation
plt.figure(figsize=(14, 14))  # 14in by 14in
sns.heatmap(hr_data.corr(), annot=True, fmt='.0%')

# Transform non-numeric columns into numerical columns
from sklearn.preprocessing import LabelEncoder

for column in hr_data.columns:
    if hr_data[column].dtype == np.number:
        continue
    hr_data[column] = LabelEncoder().fit_transform(hr_data[column])
# Create a new column at the end of the dataframe that contains the same value
hr_data['Age_Years'] = hr_data['Age']
# Remove the first column called age
hr_data = hr_data.drop('Age', axis=1)
# Show the dataframe
hr_data.describe()

hr_data_num = hr_data[['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeNumber',
                       'EnvironmentSatisfaction', 'HourlyRate',
                       'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
                       'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
                       'RelationshipSatisfaction', 'StockOptionLevel',
                       'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
                       'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
                       'YearsWithCurrManager']].copy()

plt.figure(figsize=(10, 10), dpi=100)
sns.heatmap(hr_data_num.corr())
hr_data_uc = hr_data_num[['Age', 'DailyRate', 'DistanceFromHome',
                          'EnvironmentSatisfaction', 'HourlyRate',
                          'JobInvolvement', 'JobLevel', 'JobSatisfaction',
                          'RelationshipSatisfaction', 'StockOptionLevel',
                          'TrainingTimesLastYear']].copy()

hr_data_uc.head()
##Copy categorical data
hr_data_cat = hr_data[['Productivity_Rate', 'BusinessTravel', 'Department',
                       'EducationField', 'Gender', 'JobRole', 'MaritalStatus',
                       'Over18', 'OverTime']].copy()
hr_data_cat.head()
Num_val = {'Yes': 1, 'No': 0}
hr_data_cat['Productivity_Rate'] = hr_data_cat["Productivity_Rate"].apply(lambda x: Num_val[x])
hr_data_cat.head()
hr_data_cat = pd.get_dummies(hr_data_cat)
hr_data_cat.head()

hr_data_final = pd.concat([hr_data_num, hr_data_cat], axis=1)
hr_data_final.head()

from sklearn.cross_validation import train_test_split

target = hr_data_final['Productivity_Rate']
features = hr_data_final.drop('Productivity_Rate', axis=1)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.4, random_state=10)

from sklearn.ensemble import RandomForestClassifier

#forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
model = RandomForestClassifier()
model.fit(X_train, y_train)
model.score(X_train, Y_train)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, forest.predict(X_test))
TN = cm[0][0]
TP = cm[1][1]
FN = cm[1][0]
FP = cm[0][1]

print(cm)
print('Model Testing Accuracy = "{}!"'.format((TP + TN) / (TP + TN + FN + FP)))


from sklearn.metrics import accuracy_score

test_pred = model.predict(X_test)
print(accuracy_score(y_test, test_pred))

importances = pd.DataFrame({'feature': hr_data.iloc[:, 1:hr_data.shape[1]].columns,
                            'importance': np.round(forest.feature_importances_,
                                                   3)})  # Note: The target column is at position 0
importances = importances.sort_values('importance', ascending=False).set_index('feature')
importances
importances.plot.bar()

feat_importances = pd.Series(model.feature_importances_, index=features.columns)
feat_importances = feat_importances.nlargest(20)
feat_importances.plot(kind='barh')
warnings.filterwarnings("ignore")






teams = {a:1,b:3,c:5,d:4}

