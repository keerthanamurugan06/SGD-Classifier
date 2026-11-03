# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the Iris dataset and separate the input features and target species labels.

2. Split the dataset into training and testing sets, then standardize the features using StandardScaler.

3. Initialize and train the SGDClassifier model using the standardized training data.

4. Predict the species of the test data and evaluate the model using accuracy, classification report, and confusion matrix.
5. 
## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: KEERTHANA M
RegisterNumber:  212225040185
*/
```
```
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


iris = load_iris()
X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=1)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


```

## Output:
<img width="697" height="392" alt="image" src="https://github.com/user-attachments/assets/36117b75-2f2f-4e3a-aa13-68a90e6d2789" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
