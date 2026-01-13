import pandas as pd
import numpy as np
import joblib 
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split # Import the function for splitting data into train and test sets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression #model
from sklearn.svm import SVC #model
from sklearn.ensemble import RandomForestClassifier #model
from sklearn.metrics import accuracy_score




CSV_PATH ="features.csv"

try:
    features_df = pd.read_csv(CSV_PATH)
    print("Dataset Loaded Successfully")

    x = features_df.drop('genre_label',axis=1)
    y = features_df['genre_label']

    print("\n--- Label Encoding ---")

    if np.issubdtype(y.dtype,np.integer):
        print("Labels are already numerically encoded. No action needed.")
    else:
        print("Labels are not numerical. Applying Label Encoding...")

    print("\n--- Splitting Data into Training and Testing Sets ---")
    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.25,random_state=42,stratify=y)

    print("\n--- Scaling Features ---")
    scaler = StandardScaler()
    scaler.fit(x_train)
    print("StandardScaler has been fitted to the training data.")


    #print(f"Learned means (u) for the first 5 features: {scaler.mean_[:5]}")
    #print(f"Learned standard deviations (o) for the first 5 features: {scaler.scale_[:5]}")




    #print("\nVerificatipn of scaled data: ")
    #print(f"Mean of first 5 features in x_train_scaled: {x_train_scaled[:,:5].mean(axis=0)}")

    #print(f"Standard deviation of first 5 features in x_train_scaled : {x_train_scaled[:,:5].std(axis=0)}")

    #print(f"\nMean of first 5 features in X_test_scaled: {x_test_scaled[:, :5].mean(axis=0)}")

    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    print("\nFeatures have been scaled.")

    print("\n--- Training Logistic Regression Model ---")
    log_reg= LogisticRegression(max_iter=1000)
    log_reg.fit(x_train_scaled,y_train)
    print("\nLogistic Regression Model has been trained Successfully.")
    #print(f"Model learned the following classes: {log_reg.classes_}")

    print("\n--- Training Support Vector Machine (SVM) Model ---")
    svm_model = SVC(kernel='rbf',C=1.0,random_state=42,probability=True)
    svm_model.fit(x_train_scaled,y_train)

    print("Support Vector Machine model trained successfully!")
    #print(f"Model learned the following classes: {svm_model.classes_}")    

    print("\n--- Training Random Forest Classifier Model ---")
    rf_model = RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1)
    rf_model.fit(x_train_scaled,y_train)

    print("Random Forest Classifier model trained successfully")
    #print(f"Model learned the following classes: {rf_model.classes_}")

    print("\n--- Evaluating Models on the Test set ---")

    #1. Logistic Regression Classifier
    y_pred_log_reg = log_reg.predict(x_test_scaled)
    accuracy_log_reg = accuracy_score(y_test,y_pred_log_reg)
    print(f"Logistic Regression Model Accuracy : {accuracy_log_reg * 100:.2f}%")

    #2. Support Vctor Machine Classifier
    y_pred_svm = svm_model.predict(x_test_scaled)
    accuracy_svm = accuracy_score(y_test,y_pred_svm)
    print(f"Support Vector Machine Accuracy : {accuracy_svm * 100:.2f}%")

    #3. Random Forest Classifier
    accuracy_rf = rf_model.score(x_test_scaled,y_test)
    print(f"Random Forest Classifier Accuracy : {accuracy_rf * 100:.2f}%")


    print("\n--- Saving Models andScaler to Disk ---")

    joblib.dump(scaler,'scaler.joblib')
    joblib.dump(log_reg,'logistic_regression_model.joblib')
    joblib.dump(svm_model,'svm_model.joblib')
    joblib.dump(rf_model,'random_forest_model.joblib')

    print("Scaler and models have been successfully saved to disk.")
    print("The following files have been created in your project directory:")
    print("- scaler.joblib")
    print("- logistic_regression_model.joblib")
    print("- svm_model.joblib")
    print("- random_forest_model.joblib")



    #print("\nVerifying the shapes of the new sets:")
    #print(f"x_train shape: {x_train.shape}")
    #print(f"x_test shape: {x_test.shape}")
    #print(f"y_train shape: {y_train.shape}")
    #print(f"y_test shape: {y_test.shape}")

    #print("\nVerification of target variable 'y' :")
    #print(f"Data Type of y : {y.dtype}")
    #print("First 5 Labels:")
    #print(y.head())

    #print("\nVerifying the separtaion")
    #print(f"Shape of features (x): {x.shape}")
    #print(f"Shape of labels (y): {y.shape}")

    #print("\nFirst 5 rows of features (x):")
    #print(x.head())

    #print("\nFirst 5 values of target (y):")
    #print(y.head())

except FileNotFoundError:
    print(f"Error: The file {CSV_PATH} was not found.")
    print("Please ensure 'features.csv' is in the same directory.")

except Exception as e:
    print(f"An error occurred: {e}")