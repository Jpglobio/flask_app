# app/auth/routes.py

from flask import render_template, request, redirect, url_for, flash, session, g
from . import auth  # Import the auth blueprint
from werkzeug.security import generate_password_hash, check_password_hash
from .forms import LoginForm, RegistrationForm  # Assuming you have a LoginForm in forms.py
from flask import jsonify, request

@auth.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if request.method == 'POST' and form.validate_on_submit():
        username = form.username.data
        password = form.password.data

        # Database connection using g
        cursor = g.db.cursor()

        try:
            cursor.execute("SELECT id, password FROM users WHERE username = %s", (username,))
            user = cursor.fetchone()

            if user and check_password_hash(user[1], password):  # Check hashed password
                flash('Login successful!', 'success')
                session['user_id'] = user[0]  # Store user ID in session
                session['logged_in'] = True
                return redirect(url_for('main.home'))
            else:
                flash('Invalid username or password.', 'danger')
        finally:
            cursor.close()  # Only close the cursor, not the connection

    return render_template('login.html', form=form)

@auth.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if request.method == 'POST':
        print("Form data:", form.data)
        print("Form errors:", form.errors)
        if form.validate_on_submit():
            username = form.username.data
            email = form.email.data
            password = form.password.data
            hashed_password = generate_password_hash(password)

            cursor = g.db.cursor()
            try:
                cursor.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                               (username, email, hashed_password))
                g.db.commit()
                # return jsonify({'message': 'Registration successful! You can now log in.', 'status': 'success'}), 201
                session['success_message'] = 'Registration successful! You can now log in.'
                return redirect(url_for('main.home'))
            except Exception as e:
                g.db.rollback()
                flash('Registration failed. Please try again.', 'error')
                # print(f"Error saving user: {str(e)}")
                # return jsonify({'message': 'Registration failed. Please try again.', 'status': 'error', 'error': str(e)}), 400
        else:
            return jsonify({'message': 'Invalid form submission.', 'errors': form.errors}), 400

    # GET request
    return render_template('signup.html', form=form)


@auth.route('/logout')
def logout():
    session.pop('user_id', None)
    session['logged_in'] = False
    flash('You have been logged out.', 'info')
    return redirect(url_for('auth.login'))
