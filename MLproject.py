import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle

diabetes_dataset = pd.read_csv('C:\\DBMSLAB\\myprogs\\diabetes.csv')
cols = ['Glucose',
        'BloodPressure',
        'SkinThickness',
        'Insulin',
        'BMI']

for col in cols:
    diabetes_dataset[col] = diabetes_dataset[col].replace(
        0,
        diabetes_dataset[col].median()
    )

# Graph 1
plt.figure(figsize=(6,4))
sns.countplot(x='Outcome', data=diabetes_dataset)
plt.title('Distribution of Diabetes Outcome')
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.show()

# Graph 2
plt.figure(figsize=(10,8))
sns.heatmap(
    diabetes_dataset.corr(),
    annot=True,
    cmap='coolwarm'
)
plt.title('Feature Correlation Heatmap')
plt.show()

# Graph 3
plt.figure(figsize=(6,4))
sns.boxplot(
    x='Outcome',
    y='Glucose',
    data=diabetes_dataset
)
plt.title('Glucose Level vs Diabetes Outcome')
plt.show()

diabetes_dataset.head()
diabetes_dataset.shape
diabetes_dataset.describe()
diabetes_dataset['Outcome'].value_counts()
diabetes_dataset.groupby('Outcome').mean()
# separating the data and labels
X = diabetes_dataset.drop(columns='Outcome')
Y = diabetes_dataset['Outcome']
# print(X)
# print(Y)
scaler = StandardScaler()
scaler.fit(X)
StandardScaler(copy=True, with_mean=True, with_std=True)
standardized_data = scaler.transform(X)
# print(standardized_data)
X = standardized_data
Y = diabetes_dataset['Outcome']
# print(X)
# print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=2),
    "Random Forest": RandomForestClassifier(random_state=2),
    "KNN": KNeighborsClassifier(),
    "SVM": svm.SVC(kernel='linear')
}

print("\nMODEL COMPARISON\n")

for name, model in models.items():

    model.fit(X_train, Y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(Y_test, prediction)

    print(f"{name} : {accuracy:.4f}")

classifier = svm.SVC(kernel='linear')
#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)
pickle.dump(classifier, open('diabetes_model.sav', 'wb'))
pickle.dump(scaler,
            open('scaler.sav','wb'))
# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)
# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data : ', test_data_accuracy)
cm = confusion_matrix(Y_test, X_test_prediction)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(Y_test, X_test_prediction))

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Not Diabetic','Diabetic'],
    yticklabels=['Not Diabetic','Diabetic']
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()
input_data = (1,85,66,29,0,26.6,0.351,31)

input_df = pd.DataFrame(
    [input_data],
    columns=[
        'Pregnancies',
        'Glucose',
        'BloodPressure',
        'SkinThickness',
        'Insulin',
        'BMI',
        'DiabetesPedigreeFunction',
        'Age'
    ]
)

std_data = scaler.transform(input_df)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')




