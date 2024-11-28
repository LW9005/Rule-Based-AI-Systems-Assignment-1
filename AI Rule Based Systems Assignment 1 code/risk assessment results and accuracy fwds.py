from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, log_loss, precision_recall_curve

# Fetch dataset from UCI repository
def load_data():
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets
    return X, y

# Risk assessment function which is applied in the main function
def risk_assessment(row):
    risk = 0

    # Rule 1- Age 
    if row['age'] > 65:
        risk += 3  # High risk for patients over 65
    else:
        if 50 <= row['age'] <= 65:
           risk += 2  # Moderate risk for patients 50-65

    # Rule 2- Sex 
    if row['sex'] == 1:  # Male
        risk += 2  # Males have higher risk
    else:  # Female
        if row['age'] > 65:  # Older women have similar risks to men
            risk += 2
    
    # Rule 3- Chest pain type 
    if row['cp'] == 1:  # Typical angina
        risk += 3
    elif row['cp'] == 2:  # Atypical angina
        risk += 2
    elif row['cp'] == 3:  # Non-anginal pain
        risk += 1  
    elif row['cp'] == 4:  # Asymptomatic chest pain
        risk += 2

    # Rule 4- Cholesterol 
    if row['chol'] > 240:
        risk += 3  # High cholesterol 
    elif 200 <= row['chol'] <= 240:
        risk += 2  # Borderline cholesterol 
    else:
        risk += 1  # Cholesterol below 200 

    # Rule 5- Maximum heart rate achieved 
    max_heart_rate = row['age'] * 0.7  # Estimated maximum heart rate for a human
    maximum_rate = max_heart_rate - 208

    if row['thalach'] < maximum_rate:
        risk += 2  # Below target heart rate zone
    elif row['thalach'] > maximum_rate:
        risk += 1  # Above target zone
    else:
        risk += 0  # Within the target zone

    # Rule 6- Exercise-induced angina 
    if row['exang'] == 1:
        risk += 2  # Exercise-induced angina present

    # Determine final risk categories
    if risk >= 9:
        return "High"
    elif risk >= 6:
        return "Moderate"
    else:
        return "Low"
    

# Main function to process data and execute risk
def main():
    
    # Load dataset
    X, y = load_data()

    # Split dataset into training and testing subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply the rule-based system to the training set
    X_train['Risk'] = X_train.apply(risk_assessment, axis=1)

    # Apply the rule-based system to the test set for predictions
    X_test['Risk'] = X_test.apply(risk_assessment, axis=1)

    # Map the risk categories to binary values (High = 1, Moderate/Low = 0)
    y_pred = X_test['Risk'].apply(lambda x: 1 if x == 'High' else 0)
    
    # Convert the target values (y_test) to binary labels (1 = heart disease, 0 = no heart disease)
    y_true = y_test.map(lambda x: 1 if x == 1 else 0)  # Here, `y_test` is a Series of 0s and 1s already

    # Calculate performance metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)

    # Calculate Log Loss (requires predicted probabilities, not just the binary predictions)
    # For this, we'll assume the rule-based system provides "probabilities". 
    # As a simple workaround, we can assume 0.9 probability for 'High' and 0.1 for others.
    y_pred_proba = X_test['Risk'].apply(lambda x: 0.9 if x == 'High' else 0.1)

    # Calculate log loss (requires probabilities, not binary outcomes)
    logloss = log_loss(y_true, y_pred_proba)

    # Calculate Precision-Recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)

    # Print the results
    print("Test Dataset : Heart Disease Risk Assessment Results:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Log Loss: {logloss:.2f}")
    print("Precision-Recall curve:")
    print(f"Precision: {precision_curve}")
    print(f"Recall: {recall_curve}")

    # Display some sample test data
    print(X_test[['age', 'sex', 'chol', 'thalach', 'cp', 'exang', 'Risk']].head())

# Run the system
if __name__ == "__main__":
    main()
