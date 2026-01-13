import pandas as pd 
import numpy as np
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

print("--- Model Evaluation Script ---")

try:
    print("\n[1/4] Loading and preparing test data...")
    features_df = pd.read_csv("features.csv")

    x = features_df.drop('genre_label',axis=1)
    y = features_df['genre_label']

    _, x_test, _, y_test = train_test_split(x, y, test_size=0.25, random_state=42,stratify=y)

    print("Test data loaded and split successfully")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    print("\n[2/4] Loading scikit-learn models and scaler...")
    scaler = joblib.load('scaler.joblib')

    log_reg_model = joblib.load('logistic_regression_model.joblib')
    rf_model = joblib.load('random_forest_model.joblib')
    svm_model = joblib.load('svm_model.joblib')

    print("Scikit-learn assets loaded successfully.")
    print(f"Scaler: {type(scaler)}")
    print(f"Logistic Regression Model: {type(log_reg_model)}")
    print(f"SVM Model: {type(svm_model)}")
    print(f"Random Forest Model: {type(rf_model)}")

    print("\n[3/4] Loading Keras CNN Model...")
    cnn_model = tf.keras.models.load_model('music_genre_cnn.h5')
    print("Keras CNN Model loaded successfully")
    print(f"CNN Model: {type(cnn_model)}")

    print("\n[4/4] Preparing test data for model predictions...")
    x_test_scaled = scaler.transform(x_test)
    print(f"Shape of x_test_scaled (for scikit-learn): {x_test_scaled.shape}")

    x_test_cnn = np.expand_dims(x_test_scaled,axis=-1)
    print(f"Shape of x_test_cnn (for Keras): {x_test_cnn.shape}")

    print("\nAll models and data are loaded and ready for evaluation!")


    print("\n[5/5] Generating predictions on the test set...")

    y_pred_log_reg = log_reg_model.predict(x_test_scaled)
    y_pred_rf = rf_model.predict(x_test_scaled)
    y_pred_svm = svm_model.predict(x_test_scaled)

    print("Predictions generated for scikit-learn models.")

    y_pred_cnn_probs = cnn_model.predict(x_test_cnn)
    y_pred_cnn = np.argmax(y_pred_cnn_probs, axis=1)
    print("Predictions generated for Keras CNN Model.")

    print("\n--- Verifying Prediction Shapes ---")
    print(f"Logistic Regression Predictions Shape: {y_pred_log_reg.shape}")
    print(f"SVM Predictions Shape: {y_pred_svm.shape}")
    print(f"Random Forest Predictions Shape: {y_pred_rf.shape}")
    print(f"CNN Predictions Shape: {y_pred_cnn.shape}")

    print("\nAll predictions have been generated successfully!")

    genre_names = [
        'blues', 'classical', 'country', 'disco', 'hiphop', 
        'jazz', 'metal', 'pop', 'reggae', 'rock'
    ]
    print("\n"+"="*60)
    print(" Classification Report: Logistic Regression")
    print("="*60)
    print(classification_report(y_test,y_pred_log_reg,target_names=genre_names))

    print("\n" + "="*60)
    print("      Classification Report: Support Vector Machine (SVM)")
    print("="*60)
    print(classification_report(y_test, y_pred_svm, target_names=genre_names))

    print("\n" + "="*60)
    print("      Classification Report: Random Forest")
    print("="*60)
    print(classification_report(y_test, y_pred_rf, target_names=genre_names))

    print("\n" + "="*60)
    print("      Classification Report: Convolutional Neural Network (CNN)")
    print("="*60)
    print(classification_report(y_test, y_pred_cnn, target_names=genre_names))


    # --- 7. Compute the Confusion Matrix for Each Model ---
    print("\n" + "="*60)
    print("           Computing Confusion Matrices")
    print("="*60)

    cm_log_reg = confusion_matrix(y_test, y_pred_log_reg)
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    cm_cnn = confusion_matrix(y_test, y_pred_cnn)

    print("\n--- Logistic Regression Confusion Matrix (raw) ---")
    print(cm_log_reg)
    print(f"Shape: {cm_log_reg.shape}")

    print("\n--- SVM Confusion Matrix (raw) ---")
    print(cm_svm)

    print("\n--- Random Forest Confusion Matrix (raw) ---")
    print(cm_rf)

    print("\n--- CNN Confusion Matrix (raw) ---")
    print(cm_cnn)

    print("\nConfusion matrices computed successfully.")

    def plot_confusion_matrix(cm,labels,title,ax):
        sns.heatmap(
            cm,                  # The confusion matrix data
            annot=True,          # Annotate each cell with its value
            fmt='d',             # Format the annotation as an integer
            cmap='Reds',        # Use the 'Blues' color map
            xticklabels=labels,  # Set the x-axis labels
            yticklabels=labels,  # Set the y-axis labels
            ax=ax                # Plot on the provided subplot axis
        )
        ax.set_title(title,fontsize = 14)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Confusion Matrices for all models',fontsize=20)

    plot_confusion_matrix(cm_log_reg,genre_names,'Logistic Regression', axes[0,0])
    plot_confusion_matrix(cm_svm,genre_names,'Support Vector Machine', axes[0,1])
    plot_confusion_matrix(cm_rf,genre_names,'Random Forest', axes[1,0])
    plot_confusion_matrix(cm_cnn,genre_names,'Convolutional Neural Network', axes[1,1])

    plt.tight_layout(rect=[0,0,1,0.96])
    plt.show()

except FileNotFoundError as e:
    print(f"\nERROR: A required file was not found: {e.filename}")
    print("Please ensure all model files ('scaler.joblib', '*.joblib', 'music_genre_cnn.h5') and 'features.csv' are in the correct directory.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
