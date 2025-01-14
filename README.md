import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


data = pd.read_csv('diabetes.csv')
data.head()


X = data.drop(columns='Outcome').values
y = data['Outcome'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))
    

def knn_predict(X_train, y_train, X_test, k):
 pred = []
 for test_point in X_test:
  distances = [euclidean_distance(train_point, test_point) for train_point in X_train]
  nearest_indices = np.argsort(distances)[:k]
  nearest_labels = y_train[nearest_indices]
# Use majority voting to determine the predicted label
  mejvoting = Counter(nearest_labels).most_common()[0][0] ##(0,3),(1,2)
  pred.append(mejvoting) 
 return np.array(pred)


 # Calculate accuracy
y_pred=knn_predict(X_train, y_train, X_test, k=5)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
tp=cm[0][0]
fp=cm[0][1]
fn=cm[1][0]
tn=cm[1][1]
print(f'TP: {tp}')
print(f'FP: {fp}')
print(f'FN: {fn}')
print(f'TN: {tn}')
accuracy=(tp+tn)/(tp+fp+tn+fn)
print(f'Accuracy: {accuracy}')
