# Support Vector Machine (SVM)

# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('processed_train.csv')
X = dataset.iloc[:, [1,2,3,4,5,6,7,8,9,10,11,12,13,14]].values
y = dataset.iloc[:, 0].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the SVM model on the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

"""## Accuracy Checking K fold cross validation"""

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 3)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

"""Scoring Model"""
classifier.score(X_test, y_test)


from sklearn.metrics import confusion_matrix
confusionm = confusion_matrix(y_test, y_pred)
confusionm.shape
print(confusionm)

#Saving model in behaviourmodel
import joblib
joblib.dump(classifier,'./behaviorModel/behavior_model.joblib')
