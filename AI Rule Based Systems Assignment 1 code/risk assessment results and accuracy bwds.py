import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, log_loss, precision_recall_curve
import numpy as np
from sklearn.preprocessing import LabelBinarizer

# Fetch dataset from UCI repository
def load_data():
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets
    return X, y

def backward_chain_risk_assessment(row, goal):
    # Risk determination using backward chaining rules
    risk = None  # Set to None first

    if goal == "High":
        if row['age'] > 65:
            return "High"
        elif row['sex'] == 1 and row['cp'] == 1 and row['chol'] > 240:
            return "High"
        elif row['thalach'] < (220 - row['age']) * 0.50:
            return "High"
        elif row['exang'] == 1:
            return "High"

    elif goal == "Moderate":
        if 50 <= row['age'] <= 65 and row['chol'] > 200 and row['cp'] != 4:
            return "Moderate"
        elif row['thalach'] < (220 - row['age']) * 0.70:
            return "Moderate"
        elif row['exang'] == 1:
            return "Moderate"

    elif goal == "Low":
        if row['chol'] < 200 and row['cp'] == 3:
            return "Low"
        elif row['thalach'] > (220 - row['age']) * 0.85:
            return "Low"
        else:
            return "Low"

    return "Low"

# Main function to process data and execute risk assessment
def main():
    # Load dataset
    X, y = load_data()

    # Split dataset into training and testing subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply the risk assessment to the training set
    X_train['Risk'] = X_train.apply(lambda row: backward_chain_risk_assessment(row, "High"), axis=1)

    # Apply the backward chaining assessment to the test set
    X_test['Risk'] = X_test.apply(lambda row: backward_chain_risk_assessment(row, "High"), axis=1)

    # Convert 'Risk' to numerical values for log loss calculation (0 for High, 1 for Moderate, 2 for Low)
    risk_map = {'High': 0, 'Moderate': 1, 'Low': 2}
    y_pred = X_test['Risk'].apply(lambda x: risk_map[x])
    
    # True labels (y_test) are already numerical, assuming they map to 0, 1, 2 (for High, Moderate, Low)
    y_true = y_test
    
    # Calculate the predicted probabilities for each risk category
    y_pred_proba = np.zeros((y_pred.shape[0], 3))
    for i, risk in enumerate(X_test['Risk']):
        if risk == 'High':
            y_pred_proba[i] = [0.9, 0.05, 0.05]  # High risk: 90% for High, 5% for Moderate and Low
        elif risk == 'Moderate':
            y_pred_proba[i] = [0.05, 0.9, 0.05]  # Moderate risk: 90% for Moderate, 5% for High and Low
        else:
            y_pred_proba[i] = [0.05, 0.05, 0.9]  # Low risk: 90% for Low, 5% for High and Moderate

    # Calculate performance metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')  # Changed to macro for multiclass
    recall = recall_score(y_true, y_pred, average='macro')  # Changed to macro for multiclass
    precision = precision_score(y_true, y_pred, average='macro')  # Changed to macro for multiclass

    # Log Loss: Now calculating for all classes
    logloss = log_loss(y_true, y_pred_proba, labels=[0, 1, 2])  # Define labels explicitly

    # Precision-Recall Curve using One-vs-Rest strategy
    lb = LabelBinarizer()
    lb.fit([0, 1, 2])  # Ensure all three classes are considered

    # Binarize the true labels
    y_true_bin = lb.transform(y_true)
    
    # Calculate precision-recall curves for each class (One-vs-Rest)
    for i in range(3):  # For each of the 3 classes
        precision_curve, recall_curve, _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
        print(f"Class {i} - Precision-Recall curve:")
        print(f"Precision values: {precision_curve}")
        print(f"Recall values: {recall_curve}")

    # Print the results
    print("Test Dataset with Risk Assessment (Backward Chaining):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (Macro): {f1:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Log Loss: {logloss:.4f}")
    print("\nSample Results:")
    print(X_test[['age', 'sex', 'chol', 'thalach', 'cp', 'exang', 'Risk']].head())

# Run the system
if __name__ == "__main__":
    main()
