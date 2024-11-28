from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split

# Fetch dataset from UCI repository
def load_data():
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets
    return X, y

#Risk assessment function which is applied in main function
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

    # Display some results from the test set
    print("Test Dataset : Heart Disease Risk Assessment:")
    print(X_train[['age','sex', 'chol', 'thalach', 'cp', 'exang', 'Risk']].head())

# Run the system
if __name__ == "__main__":
    main()
