from flask import Blueprint, render_template, g, request, jsonify
import pickle
import os
import pandas as pd

main = Blueprint('main', __name__)

# Correct the paths to the model and scaler files
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.dirname(current_dir)
model_file_path = os.path.join(app_dir, 'models', 'model.pkl')
scaler_file_path = os.path.join(app_dir, 'models', 'scaler.pkl')

# Load the model and scaler
with open(model_file_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(scaler_file_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


@main.route('/')
def home():
    db = g.db
    cursor = db.cursor(dictionary=True)

    cursor.execute("SELECT * FROM students")
    students = cursor.fetchall()

    cursor.execute("SELECT fname FROM students WHERE fname RLIKE BINARY '^[A-Z]' LIMIT 1")
    fname = cursor.fetchone()

    cursor.execute("SELECT DISTINCT(major) FROM students")
    major = cursor.fetchall()

    cursor.execute("""
     SELECT fname,
           CASE
               WHEN amount = 0 THEN "NULL"
               ELSE amount
           END AS scholarship_amount
    FROM students
     """)
    scholar = cursor.fetchall()

    cursor.execute("SELECT id, fname FROM students WHERE scholarship = 0")
    non_scholar = cursor.fetchall()
    cursor.close()

    return render_template('index.html',
                           students=students,
                           fname=fname,
                           major=major,
                           scholar=scholar,
                           non_scholar=non_scholar)

@main.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = request.form
        age = float(data['age'])
        gender = int(data['gender'])
        income = float(data['income'])
        shoe_size = float(data['shoe_size'])

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

        return render_template('prediction_result.html',
                               will_buy=bool(prediction),
                               probability=float(probability))

    return render_template('prediction.html')
