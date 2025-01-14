import pandas as pd 
df = pd.read_csv('diabetes.csv')
print(df.head())
X = df.iloc[:,:-1].copy()
y = df.iloc[:,-1].copy()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


from sklearn.metrics import accuracy_score,confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


# Function to predict class based on user input
def predict_class():
    print("\nEnter the following values for prediction:")
    try:
        Pregnancies = float(input("\nPregnancies:" " "))
        Glucose = float(input("\nGlucose:" " "))
        BloodPressure = float(input("\nBloodPressure:" " "))
        SkinThickness = float(input("\nSkinThickness:" " "))
        Insulin = float(input("\nInsulin:" ""))
        BMI = float(input("\nBMI:" ""))
        DiabetesPedigreeFunction = float(input("\nDiabetesPedigreeFunction:" ""))
        Age = float(input("\nAge:" ""))

        # Combine inputs into a DataFrame with column names matching the dataset
        user_data = pd.DataFrame([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]], columns=X.columns)
        
        # Scale the user input
        user_data_scaled = scaler.transform(user_data)
        
        # Predict the class
        user_pred =clf.predict(user_data_scaled)
        
        print(f"\nPredicted Class: {user_pred[0]}")
    except ValueError:
        print("Invalid input! Please enter numeric values for all features.")

# Call the prediction function
predict_class()
