# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 23:57:52 2024

@author: John

predict if the customer will buy shoes according to these factors in the
dataset.

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os

# Create larger sample dataset
def create_sample_data(n_samples=15500):
    np.random.seed(42)

    data = {
        'Age': np.random.randint(18, 70, n_samples),
        'Gender': np.random.choice([0, 1], n_samples),  # 0: Male, 1: Female
        'Income': np.random.randint(20000, 100000, n_samples),
        'ShoeSize': np.random.normal(loc=8, scale=2, size=n_samples).round(1)
    }

    df = pd.DataFrame(data)

    # Create target variable with realistic rules
    df['WillBuy'] = (
        ((df['Age'] < 30) & (df['Income'] > 40000)) |  # Young with decent income
        ((df['Income'] > 70000)) |                     # High income
        ((df['Age'] > 50) & (df['Income'] > 60000))    # Older with good income
    ).astype(int)

    # Adding rules for young with bad income and older with bad income
    df['WillBuy'] = (
        df['WillBuy'] |
        ((df['Age'] < 30) & (df['Income'] < 30000)) & (np.random.rand(n_samples) < 0.3) |  # Young with bad income: 30% chance to buy
        ((df['Age'] > 50) & (df['Income'] < 40000)) & (np.random.rand(n_samples) < 0.2)    # Older with bad income: 20% chance to buy
    ).astype(int)

    return df


import pickle

def main():

    df = create_sample_data()
    print("Sample data (first 5 rows):")
    print(df.head())

    # Prepare features and target
    X = df[['Age', 'Gender', 'Income', 'ShoeSize']]
    y = df['WillBuy']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    numerical_features = ['Age', 'Income', 'ShoeSize']
    X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Print metrics
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred):.2f}")

    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")


   # Save the model and scaler to the same directory as the script
    current_directory = os.path.dirname(os.path.abspath(__file__))
    model_file_path = os.path.join(current_directory, 'model.pkl')
    scaler_file_path = os.path.join(current_directory, 'scaler.pkl')

    with open(model_file_path, 'wb') as model_file:
        pickle.dump(model, model_file)

    with open(scaler_file_path, 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    return model, scaler

def predict_shoe_purchase(model, scaler, age, gender, income, shoe_size):
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

if __name__ == "__main__":
    model, scaler = main()

    # Example predictions
    print("\nExample Predictions:")

    # Example 1: Young person with decent income
    pred1, prob1 = predict_shoe_purchase(model, scaler, age=25, gender=0, income=45000, shoe_size=9)
    print(f"Young person: Will buy = {bool(pred1)}, Probability = {prob1:.2f}")

    # Example 2: Older person with high income
    pred2, prob2 = predict_shoe_purchase(model, scaler, age=55, gender=1, income=80000, shoe_size=7)
    print(f"Older, high income: Will buy = {bool(pred2)}, Probability = {prob2:.2f}")
