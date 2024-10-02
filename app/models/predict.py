import pickle
import pandas as pd

# Load the model and scaler from the current directory
model_file_path = 'model.pkl'
scaler_file_path = 'scaler.pkl'

# Load the model
with open(model_file_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Load the scaler
with open(scaler_file_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

def predict_shoe_purchase(age, gender, income, shoe_size):
    # Prepare the customer data
    customer = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Income': [income],
        'ShoeSize': [shoe_size]
    })

    # Scale numerical features
    numerical_features = ['Age', 'Income', 'ShoeSize']
    customer[numerical_features] = scaler.transform(customer[numerical_features])

    # Make prediction
    prediction = model.predict(customer)[0]
    probability = model.predict_proba(customer)[0][1]

    return prediction, probability

# Example usage
if __name__ == "__main__":
    # Example predictions
    print("\nExample Predictions:")

    # Example 1: Young person with decent income
    pred1, prob1 = predict_shoe_purchase(age=25, gender=0, income=45000, shoe_size=9)
    print(f"Young person: Will buy = {bool(pred1)}, Probability = {prob1:.2f}")

    # Example 2: Older person with high income
    pred2, prob2 = predict_shoe_purchase(age=55, gender=1, income=80000, shoe_size=7)
    print(f"Older, high income: Will buy = {bool(pred2)}, Probability = {prob2:.2f}")
